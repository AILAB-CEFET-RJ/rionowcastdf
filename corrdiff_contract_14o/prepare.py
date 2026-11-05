#!/usr/bin/env python3
"""Immutable, resumable monthly tensors and train-only normalization from 14n."""
import argparse
import fcntl
import json
import sqlite3
from pathlib import Path
import shutil

import duckdb
import numpy as np
import pandas as pd
from pair_contract import CHANNELS, digest, dump, literal, qi, choose
from readers14n import inside, strict_bool, read_channel, read_radar

HERE = Path(__file__).resolve().parent
VERSION = '14o-prepared-v1'

def checked_json(path):
    return json.loads(path.read_text())

def inputs(policy):
    s=checked_json(policy/'summary.json'); c=checked_json(policy/'policy_config.json')
    if s.get('version')!='14n-producer-policy-v1' or not s.get('complete') or s.get('object_error_hours',1):
        raise ValueError('Expected complete error-free 14n')
    for name,h in s['artifact_sha256'].items():
        if digest(inside(policy,name))!=h: raise ValueError('Changed policy artifact: '+name)
    pair=Path(c['pairing_dir']); cache=Path(c['cache_dir'])
    if digest(pair/'run_config.json')!=c['pairing_run_config_sha256']: raise ValueError('Changed pairing configuration')
    r=checked_json(pair/'run_config.json'); ps=checked_json(pair/'summary.json')
    if not ps.get('complete') or ps.get('provisional') or ps.get('era5_error_files',1): raise ValueError('Invalid pairing snapshot')
    for path,h in [(pair/'paired_manifest.parquet',c['paired_manifest_sha256']),
                   (pair/'era5_inventory_snapshot.sqlite',r['era5_snapshot_sha256']),
                   (cache/'config.json',r['radar_config_sha256']),
                   (cache/'geometry.npz',r['geometry_sha256'])]:
        if digest(path)!=h: raise ValueError('Changed source contract: '+str(path))
    with np.load(policy/'coverage.npz',allow_pickle=False) as g:
        lat=g['lat'].copy();lon=g['lon'].copy();mask=g['observed_region_mask'].copy()
    if lat.shape!=(70,84) or lon.shape!=lat.shape or mask.shape!=lat.shape or mask.dtype!=bool:
        raise ValueError('Expected 70x84 coordinate/mask arrays')
    if not mask.any() or not np.isfinite(lat).all() or not np.isfinite(lon).all(): raise ValueError('Invalid spatial arrays')
    palette=checked_json(cache/'config.json')['reference_contract']['palette_dbz']
    if sorted(palette)!=[20,25,30,35,40,45,50]: raise ValueError('Unexpected palette')
    return s,c,r,pair,lat,lon,mask,palette

def select(manifest,pilot_hours):
    df=manifest.copy();df['timestamp_utc']=pd.to_datetime(df.timestamp_utc,utc=True)
    if df.timestamp_utc.duplicated().any(): raise ValueError('Duplicate hours')
    valid=strict_bool(df.policy_pair_candidate)
    if (valid & (~strict_bool(df.in_archive_period)|strict_bool(df.local_filename_ambiguous)|
                  ~strict_bool(df.era5_technical_candidate)|df.observation_pixels.le(0))).any():
        raise ValueError('Contradictory candidate flags')
    # Purge is never a sample. Locked validation/test are prepared, never used to fit statistics.
    df=df[valid & df.split.isin(['train','validation','test_primary','test_stress_ood'])].copy()
    if (df.split.eq('train') & (strict_bool(df.locked)|~strict_bool(df.policy_train_candidate))).any():
        raise ValueError('Locked or inconsistent training sample')
    df=df.sort_values('timestamp_utc')
    if pilot_hours:
        df=df[df.split.eq('train')]
        if len(df): df=df.iloc[np.unique(np.linspace(0,len(df)-1,min(pilot_hours,len(df))).astype(int))]
    if not len(df): raise ValueError('No eligible samples')
    df['month_key']=df.timestamp_utc.dt.strftime('%Y-%m')
    return df.reset_index(drop=True)

class MonthlyReader:
    """Materialize only requested times and two levels once per source file.

    DuckDB may spill temporary tables to disk; per-channel interpolation then
    uses these tables rather than reopening the monthly Parquet for every hour.
    """
    def __init__(self,engine,db,stamps,config):
        self.engine=engine;self.tables={};self.fingerprints={};self.config=config
        wanted=pd.DataFrame({'wanted_time':stamps})
        engine.register('wanted_times',wanted)
        for stamp in stamps:
            for (p,) in db.execute('SELECT DISTINCT path FROM metrics WHERE timestamp_utc=?',(stamp.isoformat(),)):
                if p in self.tables: continue
                fp,status,details=db.execute('SELECT fingerprint,status,details FROM files WHERE path=?',(p,)).fetchone()
                path=Path(p); stat=path.stat()
                if status!='ok' or fp!=f'{stat.st_size}:{stat.st_mtime_ns}': raise ValueError('ERA5 inventory changed: '+p)
                d=json.loads(details); t=d['time_column'];table=f'month_source_{len(self.tables)}'
                columns=d['columns']
                names=[choose(columns,['latitude','lat']),choose(columns,['longitude','lon'])]
                names+=list(d['variable_columns'].values())
                pressure=d.get('pressure_column')
                if pressure in columns:names.append(pressure)
                cols=', '.join(qi(n) for n in dict.fromkeys(names) if n!=t)
                where=''
                if pressure in columns:
                    factor='/100.0' if config['pressure_unit']=='Pa' else ''
                    where=f' AND try_cast(p.{qi(d["pressure_column"])} AS DOUBLE){factor} IN (500,850)'
                query=f'''CREATE TEMP TABLE {table} AS SELECT {cols},
                  try_cast(p.{qi(t)} AS TIMESTAMPTZ) AS {qi(t)}
                  FROM read_parquet({literal(p)},hive_partitioning=false) p
                  WHERE try_cast(p.{qi(t)} AS TIMESTAMPTZ) IN (SELECT wanted_time FROM wanted_times){where}'''
                engine.execute(query)
                engine.execute(f'CREATE INDEX idx_{table} ON {table} ({qi(t)})')
                self.tables[p]=table;self.fingerprints[p]=fp
        engine.unregister('wanted_times')
    def execute(self,sql):
        for path,table in self.tables.items():
            sql=sql.replace(f'read_parquet({literal(path)},hive_partitioning=false)',table)
        return self.engine.execute(sql)
    def close(self):
        for path,fp in self.fingerprints.items():
            s=Path(path).stat()
            if fp!=f'{s.st_size}:{s.st_mtime_ns}': raise ValueError('ERA5 changed during month')
        for table in self.tables.values(): self.engine.execute('DROP TABLE '+table)

def stats_merge(count,mean,m2,fields):
    # Parallel variance merge avoids cancellation and never counts patch overlap.
    flat=np.asarray(fields,dtype=np.float64).reshape(12,-1)
    n=flat.shape[1];mu=flat.mean(axis=1);v=((flat-mu[:,None])**2).sum(axis=1)
    delta=mu-mean;total=count+n
    return total,mean+delta*n/total,m2+v+delta**2*count*n/total

def patch_starts(length,size,stride):
    if size<=0 or stride<=0 or size>length or stride>size:raise ValueError("Invalid patch geometry")
    starts=list(range(0,length-size+1,stride))
    if starts[-1]!=length-size:starts.append(length-size)
    return starts

def finish(out,selected,contract):
    ys=patch_starts(70,32,contract["patch_stride"])
    xs=patch_starts(84,32,contract["patch_stride"])
    per_hour=len(ys)*len(xs)
    count=0;mean=np.zeros(12);m2=np.zeros(12);patches=[];hour_records=[]
    sources={}
    for key,group in selected.groupby('month_key',sort=True):
        folder=out/'shards'/key; meta=checked_json(folder/'complete.json')
        for name,h in meta['sha256'].items():
            if digest(folder/name)!=h: raise ValueError('Shard changed: '+str(folder/name))
        sources.update(meta['sources'])
        x=np.load(folder/'conditioning.npy',mmap_mode='r');m=np.load(folder/'mask.npy',mmap_mode='r')
        y=np.load(folder/'target_code.npy',mmap_mode='r')
        for i,row in enumerate(group.itertuples(index=False)):
            if row.split=='train': count,mean,m2=stats_merge(count,mean,m2,x[i])
            stamp=row.timestamp_utc.isoformat()
            hour_records.append({'timestamp_utc':stamp,'split':row.split,'shard':key,'hour_index':i})
            for iy in ys:
                for ix in xs:
                    mask=m[i,iy:iy+32,ix:ix+32];n=int(mask.sum())
                    if n:
                        patches.append({'timestamp_utc':stamp,'split':row.split,'shard':key,'hour_index':i,
                          'row_start':iy,'col_start':ix,'support_pixels':n,'fully_supported':n==1024,
                          'echo_pixels':int(((y[i,iy:iy+32,ix:ix+32]>0)&mask).sum())})
        del x,m,y
    if count==0: raise ValueError('Cannot normalize without training hours')
    std=np.sqrt(m2/count);scale=np.where(std<1e-8,1.,std)
    dump(out/'normalization.json',{'channels':CHANNELS,'count_per_channel':count,'mean':mean.tolist(),
      'std':std.tolist(),'scale':scale.tolist(),'constant_channels':[c for c,v in zip(CHANNELS,std) if v<1e-8],
      'fit_scope':'eligible unique training hours before patches','pilot_only':contract['pilot_hours']>0})
    pdf=pd.DataFrame(patches);pdf.to_parquet(out/'patches.parquet',index=False)
    pd.DataFrame(hour_records).to_parquet(out/'hours.parquet',index=False)
    dump(out/'era5_fingerprints.json',sources)
    report=pdf.groupby('split').agg(patches=('split','size'),fully_supported=('fully_supported','sum'),
      supported_pixel_occurrences=('support_pixels','sum'),echo_pixel_occurrences=('echo_pixels','sum')).reset_index()
    report.to_csv(out/'patch_summary.csv',index=False)
    names=['contract.json','normalization.json','patches.parquet','hours.parquet','coverage.npz','era5_fingerprints.json']
    names += [f'shards/{key}/complete.json' for key in sorted(selected.month_key.unique())]
    dump(out/'summary.json',{'version':VERSION,'complete':True,'production_ready':False,
      'target_encoding_finalized':True,'coverage_state':contract['coverage_state'],
      'spatial_validity_confirmed':contract['spatial_validity_confirmed'],'corrdiff_native_tested':False,
      'pilot_only':contract['pilot_hours']>0,'selected_hours':len(selected),'split_hours':selected.split.value_counts().to_dict(),
      'patches':len(pdf),'patches_attempted':len(selected)*per_hour,'empty_patches_skipped':len(selected)*per_hour-len(pdf),
      'patch_stride':contract['patch_stride'],'patches_per_hour_max':per_hour,'geometric_grid_coverage_fraction':1.0,
      'fully_supported_patches':int(pdf.fully_supported.sum()),'train_normalization_pixels_per_channel':count,
      'artifact_sha256':{n:digest(out/n) for n in names},
      'note':'Eligible hours are technical candidates under the inherited coverage assumption; no scientific approval inferred.'})

def run(a):
    policy=Path(a.policy_dir).resolve();out=Path(a.output_dir).resolve()
    s,c,r,pair,lat,lon,geom,palette=inputs(policy)
    for root in (policy,pair,Path(c['cache_dir'])):
        if out==root or out in root.parents or root in out.parents: raise ValueError('Use independent output directory')
    selected=select(pd.read_parquet(policy/'policy_manifest.parquet'),a.pilot_hours)
    contract=checked_json(HERE/'contract.json')
    contract.update(policy_dir=str(policy),policy_artifacts=s['artifact_sha256'],pilot_hours=a.pilot_hours,
      selected_hours=len(selected),coverage_state=s['coverage_state'],spatial_validity_confirmed=s.get('spatial_validity_confirmed',False),
      code_sha256={p.name:digest(p) for p in sorted(HERE.glob('*.py'))})
    if out.exists() and not a.resume: raise ValueError('Output exists; use --resume for the same immutable build')
    if out.exists() and not (out/'contract.json').exists():raise ValueError('Existing directory is not a resumable 14o build')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'build.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'contract.json').exists():
            if checked_json(out/'contract.json')!=contract: raise ValueError('Contract changed; use a new directory')
        else: dump(out/'contract.json',contract)
        if (out/'summary.json').exists() and checked_json(out/'summary.json').get('complete'):
            raise ValueError('Snapshot already complete; no changes performed')
        dump(out/'summary.json',{'version':VERSION,'complete':False,'production_ready':False})
        shutil.copyfile(policy/'coverage.npz',out/'coverage.npz')
        selected.to_parquet(out/'selected_hours.parquet',index=False)
        db=sqlite3.connect((pair/'era5_inventory_snapshot.sqlite').as_uri()+'?mode=ro',uri=True)
        engine=duckdb.connect();engine.execute("SET TimeZone='UTC'");engine.execute('SET threads=2')
        engine.execute("SET memory_limit='1GB'");engine.execute('SET temp_directory='+literal(out/'duckdb_tmp'))
        provenance=[];checked=set()
        try:
            for key,group in selected.groupby('month_key',sort=True):
                folder=out/'shards'/key
                if (folder/'complete.json').exists():
                    meta=checked_json(folder/'complete.json')
                    for name,h in meta['sha256'].items():
                        if digest(folder/name)!=h: raise ValueError('Changed completed shard')
                    for p,fp in meta['sources'].items():
                        st=Path(p).stat()
                        if fp!=f'{st.st_size}:{st.st_mtime_ns}': raise ValueError('Changed ERA5 source')
                    print('Reused month '+key,flush=True);continue
                folder.mkdir(parents=True,exist_ok=True)
                monthly=MonthlyReader(engine,db,group.timestamp_utc.tolist(),r['inventory_config'])
                n=len(group); shape=(n,70,84)
                x=np.lib.format.open_memmap(folder/'conditioning.npy',mode='w+',dtype='float32',shape=(n,12,70,84))
                y=np.lib.format.open_memmap(folder/'target_code.npy',mode='w+',dtype='float32',shape=shape)
                m=np.lib.format.open_memmap(folder/'mask.npy',mode='w+',dtype='bool',shape=shape)
                try:
                    for i,row in enumerate(group.itertuples(index=False)):
                        x[i]=np.stack([read_channel(monthly,db,row.timestamp_utc,ch,r['inventory_config'],lat,lon,checked,provenance) for ch in CHANNELS])
                        y[i],m[i]=read_radar(row,policy,lat.shape,geom,palette)
                    x.flush();y.flush();m.flush()
                finally:
                    del x,y,m
                    monthly.close()
                names=['conditioning.npy','target_code.npy','mask.npy']
                dump(folder/'complete.json',{'hours':n,'sources':monthly.fingerprints,'sha256':{name:digest(folder/name) for name in names}})
                print(f'Month {key}: {n} hours committed',flush=True)
        finally: engine.close();db.close()
        finish(out,selected,contract)
        print((out/'summary.json').read_text(),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy-dir',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--pilot-hours',type=int,default=0,help='0=all splits; positive=train-only pilot, never final statistics')
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.pilot_hours<0: p.error('pilot-hours must be >=0')
    run(a)
