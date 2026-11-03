#!/usr/bin/env python3
"""14k: auditable UTC ERA5 inventory and radar pairing; never a training approval."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import fcntl

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

VERSION = '14k-pairing-v1'
ALIASES = {
 'tcwv':['tcwv','total_column_water_vapour'],
 't2m':['t2m','2m_temperature','2_metre_temperature'],
 'u10':['u10','10m_u_component_of_wind','10_metre_u_component_of_wind'],
 'v10':['v10','10m_v_component_of_wind','10_metre_v_component_of_wind'],
 't':['t','temperature'], 'r':['r','relative_humidity'],
 'u':['u','u_component_of_wind'], 'v':['v','v_component_of_wind']}
CHANNELS = ['tcwv','t2m','u10','v10']+[f'{v}_{l}' for l in (850,500) for v in ('t','r','u','v')]

def dump(path, obj):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False,default=str)+'\n')
    tmp.replace(path)

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def choose(names, options, required=True):
    found=[x for x in options if x in names]
    if len(found)>1: raise ValueError('Ambiguous columns: '+str(found))
    if not found and required: raise ValueError('Missing columns: '+str(options))
    return found[0] if found else None

def qi(name): return '"'+name.replace('"','""')+'"'
def literal(s): return "'"+str(s).replace("'","''")+"'"

def utc(value):
    t=pd.Timestamp(value)
    if t.tzinfo is None: raise ValueError('CLI timestamps must include UTC offset / Z')
    return t.tz_convert('UTC')

def connect_inventory(path):
    db=sqlite3.connect(path)
    db.executescript('''
      CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, kind TEXT, fingerprint TEXT,
        status TEXT, details TEXT);
      CREATE TABLE IF NOT EXISTS metrics(path TEXT, timestamp_utc TEXT, channel TEXT,
        n_rows INTEGER,n_coords INTEGER,n_lat INTEGER,n_lon INTEGER,n_finite INTEGER,
        n_invalid_coords INTEGER, lat_min REAL,lat_max REAL,lon_min REAL,lon_max REAL);
      CREATE INDEX IF NOT EXISTS idx_metrics ON metrics(timestamp_utc,channel);
    ''')
    return db

def scan_file(engine, path, kind, start, end, pressure_unit):
    pf=pq.ParquetFile(path)
    schema=pf.schema_arrow
    names=schema.names
    md={k.decode(errors='replace'):v.decode(errors='replace') for k,v in (schema.metadata or {}).items() if k!=b'pandas'}
    if md.get('timezone','UTC').upper() not in ('UTC','ETC/UTC','Z'):
        raise ValueError('Conflicting ERA5 timezone metadata: '+md['timezone'])
    timecol=choose(names,['valid_time','time','datetime','date'])
    # Numeric epochs require an explicit encoding, never guess seconds vs nanoseconds.
    import pyarrow as pa
    if not (pa.types.is_timestamp(schema.field(timecol).type) or pa.types.is_string(schema.field(timecol).type) or pa.types.is_large_string(schema.field(timecol).type)):
        raise ValueError('Unsupported time type: '+str(schema.field(timecol).type))
    lat=choose(names,['latitude','lat']); lon=choose(names,['longitude','lon'])
    level=choose(names,['pressure_level','level','isobaricinhpa','isobaricInhPa','pressure']) if kind=='pressure' else None
    if level:
        units=md.get('units_'+level, (schema.field(level).metadata or {}).get(b'units',b'').decode())
        if units.lower() in ('pa','hpa') and units.lower()!=pressure_unit.lower():
            raise ValueError('Pressure units conflict: metadata='+units+', CLI='+pressure_unit)
    selected={v:choose(names,ALIASES[v],False) for v in (('tcwv','t2m','u10','v10') if kind=='single' else ('t','r','u','v'))}
    details={'columns':names,'time_column':timecol,'time_type':str(schema.field(timecol).type),
      'time_policy':'aware timestamps converted to UTC; naive timestamps interpreted as UTC per supplied ETL',
      'metadata':md,'variable_columns':selected,'pressure_column':level,'rows_in_file':pf.metadata.num_rows}
    # Local file only, no hive inference: partition folders need not be physically in each file.
    src=f"read_parquet({literal(path)}, hive_partitioning=false)"
    t=f'try_cast({qi(timecol)} AS TIMESTAMPTZ)'
    nbad=engine.execute(f'SELECT count(*) FROM {src} WHERE {t} IS NULL').fetchone()[0]
    if nbad: raise ValueError(f'{nbad} null/unparseable timestamps')
    lev=f'try_cast({qi(level)} AS DOUBLE)'+('/100.0' if pressure_unit=='Pa' else '') if level else 'NULL'
    projections=[f'{t} AS ts',f'try_cast({qi(lat)} AS DOUBLE) AS lat',
      f'((try_cast({qi(lon)} AS DOUBLE)+180)%360+360)%360-180 AS lon',f'{lev} AS lev']
    projections += [f'try_cast({qi(col)} AS DOUBLE) AS {qi(v)}' for v,col in selected.items() if col]
    engine.execute('CREATE OR REPLACE TEMP TABLE current_file AS SELECT '+','.join(projections)+
      f' FROM {src} WHERE {t} >= {literal(start.isoformat())}::TIMESTAMPTZ AND {t} <= {literal(end.isoformat())}::TIMESTAMPTZ')
    if level:
        levels=engine.execute('SELECT DISTINCT lev FROM current_file ORDER BY lev').fetchall()
        details['levels_hpa']=[x[0] if x[0] is not None and np.isfinite(x[0]) else None for x in levels]
    rows=[]
    for v,col in selected.items():
        if not col: continue
        for levtarget in ((850,500) if level else (None,)):
            channel=f'{v}_{levtarget}' if level else v
            where=f'WHERE lev={levtarget}' if level else ''
            # Count distinct coordinates, not only rows; a duplicated expver/member is a review issue.
            sql=f'''SELECT ts, count(*),count(DISTINCT (lat,lon)),count(DISTINCT lat),count(DISTINCT lon),
              count(*) FILTER (WHERE isfinite({qi(v)})),
              count(*) FILTER (WHERE lat IS NULL OR lon IS NULL OR NOT isfinite(lat) OR NOT isfinite(lon) OR abs(lat)>90),
              min(lat),max(lat),min(lon),max(lon)
              FROM current_file {where} GROUP BY ts ORDER BY ts'''
            for ts,*m in engine.execute(sql).fetchall():
                # convert nonfinite aggregate bounds to NULL for JSON/SQLite safety
                m=[None if isinstance(x,float) and not np.isfinite(x) else x for x in m]
                rows.append((str(path),pd.Timestamp(ts).isoformat(),channel,*m))
    return rows,details

def index_era5(args, out, start, end):
    roots={k:Path(getattr(args,k+'_dir')).resolve() for k in ('single','pressure')}
    for root in roots.values():
        if not root.is_dir(): raise ValueError('ERA5 directory missing: '+str(root))
    config={'version':VERSION,'start_utc':start.isoformat(),'end_utc':end.isoformat(),
      'roots':{k:str(v) for k,v in roots.items()},'pressure_unit':args.pressure_unit,
      'channels':CHANNELS,'code_sha256':digest(Path(__file__)), 'fingerprint':'size + mtime_ns (not content hash)'}
    cfg=out/'inventory_config.json'
    if cfg.exists() and json.loads(cfg.read_text())!=config: raise ValueError('Inventory contract changed; use a new inventory directory')
    dump(cfg,config)
    dump(out/'inventory_summary.json',{'version':VERSION,'complete':False})
    db=connect_inventory(out/'era5_inventory.sqlite')
    engine=duckdb.connect()
    engine.execute("SET TimeZone='UTC'")
    engine.execute("SET memory_limit='1GB'")
    engine.execute('SET threads=2')
    engine.execute('SET temp_directory='+literal(out/'duckdb_tmp'))
    listing=[]
    for kind,root in roots.items():
        listing.extend((kind,p) for p in sorted(root.rglob('*.parquet')))
    if not listing: raise ValueError('No parquet files found')
    paths=[str(p) for _,p in listing]
    if len(paths)!=len(set(paths)): raise ValueError('Single and pressure directories overlap')
    oldpaths=[r[0] for r in db.execute('SELECT path FROM files')]
    with db:
        for path in set(oldpaths)-set(paths):
            db.execute('DELETE FROM metrics WHERE path=?',(path,));db.execute('DELETE FROM files WHERE path=?',(path,))
    for i,(kind,path) in enumerate(listing,1):
        st=path.stat(); fp=f'{st.st_size}:{st.st_mtime_ns}'
        old=db.execute('SELECT fingerprint,status FROM files WHERE path=?',(str(path),)).fetchone()
        if old==(fp,'ok') and not args.rescan:
            if i%100==0: print(f'ERA5 {i}/{len(listing)} reused',flush=True)
            continue
        try:
            rows,details=scan_file(engine,path,kind,start,end,args.pressure_unit)
            after=path.stat()
            if fp!=f'{after.st_size}:{after.st_mtime_ns}': raise ValueError('File changed while scanning')
            status='ok'
        except Exception as e:
            rows=[];status='error';details={'error':f'{type(e).__name__}: {e}'}
        with db:
            db.execute('DELETE FROM metrics WHERE path=?',(str(path),))
            db.executemany('INSERT INTO metrics VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',rows)
            db.execute('INSERT OR REPLACE INTO files VALUES (?,?,?,?,?)',(str(path),kind,fp,status,json.dumps(details)))
        print(f'ERA5 {i}/{len(listing)} {status} {path}',flush=True)
    files=pd.read_sql_query('SELECT * FROM files ORDER BY path',db)
    files.to_csv(out/'era5_files.csv',index=False)
    summary={'version':VERSION,'complete':True,'files':len(files),'error_files':int((files.status=='error').sum()),
      'metrics_rows':db.execute('SELECT count(*) FROM metrics').fetchone()[0],
      'channels':CHANNELS,'production_ready':False,'values_read':True}
    dump(out/'inventory_summary.json',summary)
    engine.close();db.close()
    return summary

def split_labels(times, purge_hours):
    year=times.year
    nominal=np.select([year<=2020,year<=2022,year==2023],['train','validation','test_primary'],default='test_stress_ood')
    final=nominal.copy(); reasons=np.full(len(times),'',dtype=object)
    for b in ('2021-01-01','2023-01-01','2024-01-01'):
        boundary=pd.Timestamp(b,tz='UTC');delta=pd.Timedelta(hours=purge_hours)
        mask=(times>=boundary-delta)&(times<boundary+delta)
        final[mask]='excluded_purge';reasons[mask]=boundary.isoformat()
    return nominal,final,reasons

def snapshot(cache,out,allow_partial):
    config=json.loads((cache/'config.json').read_text())
    if config.get('version')!='14j-intermediate-cache-v1': raise ValueError('Expected 14j cache')
    reported=json.loads((cache/'summary.json').read_text())
    if not reported.get('complete') and not allow_partial:
        raise ValueError('14j not complete: wait or use --allow-partial for a provisional snapshot')
    src=sqlite3.connect((cache/'manifest.sqlite').as_uri()+'?mode=ro',uri=True)
    dest=sqlite3.connect(out/'radar_manifest_snapshot.sqlite')
    src.backup(dest);src.close()
    frames=pd.read_sql_query('SELECT f.*,s.decoded_path,s.decoded_sha256 FROM frames f LEFT JOIN sources s ON f.source_sha256=s.sha256',dest)
    dest.close()
    dump(out/'radar_config_snapshot.json',config)
    dump(out/'radar_summary_snapshot.json',reported)
    return frames,config,reported

def pair(args,out,start,end):
    cache=Path(args.radar_cache_dir).resolve()
    inventory=Path(args.inventory_dir).resolve()
    iconfig=json.loads((inventory/'inventory_config.json').read_text())
    if utc(iconfig['start_utc'])>start or utc(iconfig['end_utc'])<end: raise ValueError('Inventory interval does not cover requested pairing interval')
    if iconfig['channels']!=CHANNELS: raise ValueError('Channel contract mismatch')
    frames,rconfig,rsummary=snapshot(cache,out,args.allow_partial)
    if rconfig['cadence_minutes']!=60: raise ValueError('This delivery pairs an hourly cache only')
    if utc(rconfig['start_utc'])>start or utc(rconfig['end_utc'])<end: raise ValueError('Radar cache interval does not cover requested interval')
    times=pd.date_range(start,end,freq='h')
    frames['timestamp_utc']=pd.to_datetime(frames.timestamp_utc,utc=True)
    if frames.timestamp_utc.duplicated().any(): raise ValueError('Duplicate radar UTC timestamp')
    result=pd.DataFrame({'timestamp_utc':times}).merge(frames,on='timestamp_utc',how='left',validate='one_to_one')
    result['status']=result.status.fillna('not_processed_in_snapshot')
    result['local_filename_ambiguous']=result.local_filename_ambiguous.fillna(0).astype(bool)
    with np.load(cache/'geometry.npz',allow_pickle=False) as g:
        lat=g['lat'];lon=g['lon'];bounds=(float(lat.min()),float(lat.max()),float(lon.min()),float(lon.max()))
    # Snapshot inventory too, so exports never depend on a concurrently advancing scan.
    with (inventory/'.writer.lock').open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
        isummary=json.loads((inventory/'inventory_summary.json').read_text())
        if not isummary.get('complete'): raise ValueError('ERA5 inventory incomplete: rerun inventory first')
        source=sqlite3.connect((inventory/'era5_inventory.sqlite').as_uri()+'?mode=ro',uri=True)
        db=sqlite3.connect(out/'era5_inventory_snapshot.sqlite');source.backup(db);source.close()
    metrics=pd.read_sql_query('SELECT * FROM metrics',db)
    nerrors=db.execute("SELECT count(*) FROM files WHERE status!='ok'").fetchone()[0]
    db.close()
    details=[]
    if len(metrics):
        metrics['timestamp_utc']=pd.to_datetime(metrics.timestamp_utc,utc=True)
        metrics=metrics[metrics.timestamp_utc.between(start,end)]
        a,b,c,d=bounds
        metrics['file_ok']=(metrics.n_rows.eq(metrics.n_coords)&metrics.n_coords.eq(metrics.n_lat*metrics.n_lon)&
          metrics.n_lat.ge(2)&metrics.n_lon.ge(2)&metrics.n_finite.eq(metrics.n_rows)&metrics.n_invalid_coords.eq(0)&
          metrics.lat_min.le(a)&metrics.lat_max.ge(b)&metrics.lon_min.le(c)&metrics.lon_max.ge(d))
        for (ts,ch),group in metrics.groupby(['timestamp_utc','channel'],sort=False):
            state='multiple_files_review' if len(group)>1 else ('finite_rectangular_cover' if group.file_ok.iloc[0] else 'incomplete_or_invalid_grid')
            details.append({'timestamp_utc':ts,'channel':ch,'state':state,'source_files':len(group),
              'rows':int(group.n_rows.sum()),'finite_rows':int(group.n_finite.sum())})
    diag=pd.DataFrame(details,columns=['timestamp_utc','channel','state','source_files','rows','finite_rows'])
    diag.to_parquet(out/'era5_channel_diagnostics.parquet',index=False)
    if len(diag):
        pivot=diag.pivot(index='timestamp_utc',columns='channel',values='state').reindex(columns=CHANNELS)
        result=result.merge(pivot,on='timestamp_utc',how='left',validate='one_to_one')
    else:
        for ch in CHANNELS: result[ch]=None
    result[CHANNELS]=result[CHANNELS].fillna('missing_channel_or_time')
    result['era5_ready_channels']=result[CHANNELS].eq('finite_rectangular_cover').sum(axis=1)
    result['era5_complete_12']=result.era5_ready_channels.eq(12)
    # A corrupt/unreadable file could conceal an overlapping version. Do not certify any pairing globally.
    result['era5_inventory_has_errors']=bool(nerrors)
    result['era5_technical_candidate']=result.era5_complete_12 & (nerrors==0)
    nominal,final,reasons=split_labels(times,args.purge_hours)
    result['nominal_split']=nominal;result['split']=final;result['purge_boundary']=reasons
    result['locked']=result.split.isin(['test_primary','test_stress_ood','excluded_purge'])
    result['radar_palette_on_grid']=result.status.eq('palette_sampled_observation_validity_unconfirmed')
    result['paired_palette_candidate']=result.era5_technical_candidate & result.radar_palette_on_grid & ~result.local_filename_ambiguous
    result['train_smoke_candidate']=result.paired_palette_candidate & result.split.eq('train')
    result['radar_target_validity']='unconfirmed'
    result['training_ready']=False
    result.to_parquet(out/'paired_manifest.parquet',index=False)
    result.to_csv(out/'paired_manifest.csv',index=False)
    counts=result.groupby(['split','status'],dropna=False).size().rename('hours').reset_index()
    denominators=result.groupby('split').size()
    counts['denominator_hours']=counts.split.map(denominators)
    counts['percent']=100*counts.hours/counts.denominator_hours
    counts.to_csv(out/'split_radar_status.csv',index=False)
    result['year_utc']=result.timestamp_utc.dt.year
    result.groupby(['year_utc','status']).size().rename('hours').reset_index().to_csv(out/'annual_radar_status.csv',index=False)
    coverage=result.groupby('split').agg(hours=('timestamp_utc','size'),era5_complete_12=('era5_complete_12','sum'),
      era5_technical_candidates=('era5_technical_candidate','sum'),palette_pair_candidates=('paired_palette_candidate','sum'),
      ambiguous_local_hours=('local_filename_ambiguous','sum')).reset_index()
    coverage.to_csv(out/'split_pairing_coverage.csv',index=False)
    for split in ('train','validation','test_primary','test_stress_ood','excluded_purge'):
        result[result.split.eq(split)][['timestamp_utc','status','era5_technical_candidate','paired_palette_candidate','locked']].to_parquet(out/f'{split}_timestamps.parquet',index=False)
    # Conditional scenarios quantify the unresolved transparency policy without applying it.
    scenarios=[]
    for sp,group in result.groupby('split'):
        transparent=group.status.eq('transparent_ambiguous')
        eligible=group.era5_technical_candidate & ~group.local_filename_ambiguous
        scenarios.append({'split':sp,'all_hours':len(group),'transparent_hours':int(transparent.sum()),
          'transparent_percent_of_split':100*float(transparent.mean()),
          'transparent_with_era5_unambiguous':int((transparent & eligible).sum()),
          'palette_pair_candidates':int(group.paired_palette_candidate.sum()),
          'hypothetical_palette_plus_transparent':int((group.paired_palette_candidate | (transparent & eligible)).sum()),
          'hypothesis_applied':False})
    pd.DataFrame(scenarios).to_csv(out/'transparency_scenarios.csv',index=False)
    repeat=result.dropna(subset=['source_sha256']).groupby('source_sha256').agg(
      hours=('timestamp_utc','size'),n_splits=('split','nunique'),splits=('split',lambda s:';'.join(sorted(set(s)))),
      statuses=('status',lambda s:';'.join(sorted(set(s)))))
    repeat[repeat.n_splits.gt(1)].reset_index().to_csv(out/'source_hash_cross_split.csv',index=False)
    channel_rows=[]
    for ch in CHANNELS:
        for state,n in result[ch].value_counts().items(): channel_rows.append({'channel':ch,'state':state,'hours':int(n),'denominator_hours':len(result)})
    pd.DataFrame(channel_rows).to_csv(out/'channel_coverage.csv',index=False)
    missing=int(result.status.eq('not_processed_in_snapshot').sum())
    complete=bool(rsummary.get('complete')) and missing==0
    summary={'version':VERSION,'complete':True,'radar_snapshot_complete':complete,'provisional':not complete,
      'expected_hours':len(result),'radar_not_processed':missing,'era5_error_files':nerrors,
      'era5_complete_12_hours':int(result.era5_complete_12.sum()),
      'paired_palette_candidate_hours':int(result.paired_palette_candidate.sum()),
      'train_smoke_candidate_hours':int(result.train_smoke_candidate.sum()),
      'split_counts':{k:int(v) for k,v in result.split.value_counts().items()},
      'production_ready':False,'dry_rule_documented':False,'temporal_aggregation':'none',
      'note':'Candidates are technical availability only, not valid full radar targets. No patches or normalizations created.'}
    dump(out/'summary.json',summary)
    dump(out/'run_config.json',{'version':VERSION,'created_at_utc':datetime.now(timezone.utc).isoformat(),
      'args':vars(args),'code_sha256':digest(Path(__file__)),'inventory_config':iconfig,
      'radar_config_sha256':digest(cache/'config.json'),'geometry_sha256':digest(cache/'geometry.npz'),
      'radar_snapshot_sha256':digest(out/'radar_manifest_snapshot.sqlite'),
      'era5_snapshot_sha256':digest(out/'era5_inventory_snapshot.sqlite'),'target_bounds':bounds,
      'purge_interval':'[boundary-48h, boundary+48h) by default; matches phase15',
      'units_policy':'recorded from source metadata; no conversion or physical range validation'})
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    inv=sub.add_parser('inventory',help='Can run while 14j is building')
    inv.add_argument('--single-dir',required=True);inv.add_argument('--pressure-dir',required=True)
    inv.add_argument('--pressure-unit',choices=['hPa','Pa'],default='hPa')
    inv.add_argument('--rescan',action='store_true',help='Ignore size/mtime reuse and reread all values')
    pairp=sub.add_parser('pair',help='Read-only snapshot of 14j and ERA5 inventories')
    pairp.add_argument('--radar-cache-dir',required=True);pairp.add_argument('--inventory-dir',required=True)
    pairp.add_argument('--allow-partial',action='store_true')
    pairp.add_argument('--purge-hours',type=int,default=48)
    for x in (inv,pairp):
        x.add_argument('--output-dir',required=True)
        x.add_argument('--start-utc',default='2011-01-01T00:00:00Z')
        x.add_argument('--end-utc',default='2024-12-31T23:00:00Z')
    args=p.parse_args()
    start,end=utc(args.start_utc),utc(args.end_utc)
    if end<start or start!=start.floor('h') or end!=end.floor('h'): p.error('Use ordered exact-hour UTC boundaries')
    if start.year<2011 or end.year>2024: p.error('Split protocol is limited to 2011–2024')
    if args.command=='pair' and args.purge_hours<0: p.error('purge-hours must be >=0')
    out=Path(args.output_dir).resolve()
    inputs=[Path(getattr(args,k)).resolve() for k in ('single_dir','pressure_dir','radar_cache_dir','inventory_dir') if hasattr(args,k)]
    if any(out==r or out in r.parents or r in out.parents for r in inputs): p.error('Output and input trees must be separate')
    if args.command=='pair' and out.exists(): p.error('Pair output must be a new directory; snapshots are immutable')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=index_era5(args,out,start,end) if args.command=='inventory' else pair(args,out,start,end)
    print(json.dumps(result,indent=2),flush=True)
    return 2 if result.get('error_files',result.get('era5_error_files',0)) else 0

if __name__=='__main__':
    try: sys.exit(main())
    except KeyboardInterrupt: print('Interrupted. Rerun inventory; use a new directory for pair.',file=sys.stderr);sys.exit(130)
    except Exception as e: print(f'{type(e).__name__}: {e}',file=sys.stderr);sys.exit(1)
