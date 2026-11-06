"""Reference Zarr builder: frozen temporal manifest, producer policy, train-only statistics."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import duckdb
import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc
from .contract_common import *
from .contract_era5 import Inventory,Month


def cache_index(c,semantic):
    root=Path(c['radar_cache_dir']);summary=read(root/'summary.json')
    expected={k:v for k,v in semantic.items() if k!='channels'}
    if not summary.get('complete') or read(root/'contract.json')!=expected:raise ValueError('Complete matching radar cache required')
    if sha(root/'index.sqlite')!=summary['index_sha256']:raise ValueError('Radar index changed')
    with ro(root/'index.sqlite') as db:return {r[0]:r[1:] for r in db.execute('SELECT sha,path,digest,policy FROM objects')}


def month_stages(c,df,g,out,index,inventory):
    engine=duckdb.connect();engine.execute("SET TimeZone='UTC'")
    engine.execute(f"SET threads={int(c['duckdb_threads'])}")
    engine.execute('SET memory_limit=?',[c['duckdb_memory']])
    engine.execute('SET temp_directory=?',[str(out/'duckdb_tmp')])
    nc=len(c['channels']);stages=[]
    try:
        for name,rows in df.groupby(df.timestamp_utc.dt.strftime('%Y%m'),sort=True):
            folder=out/'staging'/name;folder.mkdir(parents=True,exist_ok=True);done=folder/'complete.json'
            files=[folder/(n+'.npy') for n in ('input','target_code','mask')]
            if done.exists():
                record=read(done)
                if record['hours']!=len(rows) or any(sha(f)!=record['sha256'][f.name] for f in files):raise ValueError('Staging corruption '+name)
            else:
                x=np.lib.format.open_memmap(files[0],mode='w+',dtype='float32',shape=(len(rows),nc,70,84))
                y=np.lib.format.open_memmap(files[1],mode='w+',dtype='float32',shape=(len(rows),1,70,84))
                m=np.lib.format.open_memmap(files[2],mode='w+',dtype='bool',shape=(len(rows),1,70,84))
                reader=Month(engine,inventory,rows.timestamp_utc.tolist(),c['channels'])
                try:
                    for i,r in enumerate(rows.itertuples(index=False)):
                        for j,ch in enumerate(c['channels']):x[i,j]=reader.field(r.timestamp_utc,ch,g['lat'],g['lon'])
                        rel,digest,policy=index[r.sha]
                        if policy!=r.policy_status:raise ValueError('Radar policy differs from manifest')
                        p=inside(c['radar_cache_dir'],rel)
                        if sha(p)!=digest:raise ValueError('Radar object changed '+rel)
                        with np.load(p,allow_pickle=False) as z:
                            code=z['target_code'];mask=z['mask']
                            if str(z['source_sha256'])!=r.sha:raise ValueError('Wrong radar source')
                            validate_target(code,mask,g['support']);y[i,0]=code;m[i,0]=mask
                        if (i+1)%100==0:print(json.dumps({'month':name,'hour':i+1,'total':len(rows)}),flush=True)
                finally:reader.close()
                for arr in (x,y,m):arr.flush()
                del x,y,m,arr
                dump(done,{'hours':len(rows),'sha256':{f.name:sha(f) for f in files}})
            stages.append((name,rows,folder))
            print(json.dumps({'staged_month':name,'hours':len(rows)}),flush=True)
    finally:engine.close()
    return stages


def normalization(stages,channels):
    count=0;mean=np.zeros(len(channels),np.float64);m2=mean.copy();hours=0
    for name,rows,folder in stages:
        x=np.load(folder/'input.npy',mmap_mode='r')
        for i in np.flatnonzero(rows.split.to_numpy()=='train'):
            values=x[i].astype(np.float64).reshape(len(channels),-1)
            n=values.shape[1];mu=values.mean(1);v=((values-mu[:,None])**2).sum(1)
            delta=mu-mean;total=count+n;m2+=v+delta**2*count*n/total;mean+=delta*n/total;count=total;hours+=1
    if not count:raise ValueError('Need training hours to fit normalization; include train in configured period')
    std=np.sqrt(m2/count);effective=np.maximum(std,1e-8)
    return dict(channels=channels,input_mean=mean.tolist(),input_std=effective.tolist(),input_std_unfloored=std.tolist(),
                train_hours=hours,points_per_channel=count,scope='unique train hours, all 70x84 grid points once before patches',
                target_transform='target_code / 50',target_mean=[0.0],target_std=[1.0],normalization_floor=1e-8)


def patches(mask):
    return [(r,c) for r in starts(70) for c in starts(84) if mask[...,r:r+32,c:c+32].any()]


def checksum_arrays(group):
    result={}
    for key in sorted(group.array_keys()):
        a=group[key];h=hashlib.sha256()
        h.update(str(a.shape).encode());h.update(str(a.dtype).encode())
        # Small fixed coordinates and hour manifests can have distinct ranks.
        for i in range(0,a.shape[0],48):h.update(np.ascontiguousarray(a[i:i+48]).tobytes())
        result[key]=h.hexdigest()
    return result


def write_split(c,split,stages,g,out,semantic):
    dest=out/(split+'.zarr');done=out/(split+'.complete.json')
    if done.exists():
        z=zarr.open_group(str(dest),mode='r');record=read(done)
        if checksum_arrays(z)!=record['array_sha256']:raise ValueError('Completed Zarr corruption: '+split)
        return record
    # Only this builder's disposable incomplete directory may be removed.
    tmp=out/('.'+split+'.zarr.build')
    if tmp.exists():shutil.rmtree(tmp)
    if dest.exists():raise ValueError('Zarr exists without completion record; move aside before resume: '+str(dest))
    plans=[];total=0;used_hours=0
    for name,rows,folder in stages:
        m=np.load(folder/'mask.npy',mmap_mode='r')
        for i in np.flatnonzero(rows.split.to_numpy()==split):
            positions=patches(m[i]);r=rows.iloc[i]
            plans.append((folder,int(i),int(r.timestamp_utc.timestamp()),int(r.source_utc.timestamp()),int(r.offset_seconds),r.sha,len(positions)))
            total+=len(positions);used_hours+=bool(positions)
    z=zarr.open_group(str(tmp),mode='w');compress=Blosc(cname='zstd',clevel=3,shuffle=Blosc.BITSHUFFLE)
    nc=len(c['channels'])
    spec={'input':((total,nc,32,32),'f4'),'target':((total,1,32,32),'f4'),'mask':((total,1,32,32),'bool'),
          'timestamps':((total,),'i8'),'source_timestamps':((total,),'i8'),'offset_seconds':((total,),'i2'),
          'patch_row':((total,),'i2'),'patch_col':((total,),'i2'),'source_sha256':((total,),'S64')}
    for key,(shape,dtype) in spec.items():
        a=z.create_dataset(key,shape=shape,chunks=(48,)+shape[1:],dtype=dtype,compressor=compress,fill_value=None)
        a.attrs['_ARRAY_DIMENSIONS']=['sample']+([('input_channel' if key=='input' else 'target_channel'),'y','x'] if len(shape)==4 else [])
    for key in ('lat','lon','support'):
        a=z.create_dataset('grid_'+key,data=g[key],compressor=compress,fill_value=None);a.attrs['_ARRAY_DIMENSIONS']=['grid_y','grid_x']
    z.attrs.update(dict(version=VERSION,split=split,channels=c['channels'],input_units='physical; see ERA5 provenance',
        input_normalized=False,target_transform='target_code / 50',no_echo_code=0,unknown='mask=false; tensor placeholder=0',
        timestamp_units='seconds since 1970-01-01T00:00:00Z',contract_sha256=digest_obj(semantic),
        coverage_state=semantic['coverage_state'],spatial_validity_confirmed=semantic['spatial_validity_confirmed'],
        production_ready=False,complete=False))
    cursor=0;loaded=None;support_count=0;zero_count=0
    for hour,(folder,i,stamp,source_stamp,offset,source_hash,npositions) in enumerate(plans,1):
        if folder!=loaded:
            x=np.load(folder/'input.npy',mmap_mode='r');y=np.load(folder/'target_code.npy',mmap_mode='r');m=np.load(folder/'mask.npy',mmap_mode='r');loaded=folder
        positions=patches(m[i])
        if len(positions)!=npositions:raise ValueError('Staging mask changed')
        if not positions:continue
        xx=np.stack([x[i,:,r:r+32,c0:c0+32] for r,c0 in positions])
        mm=np.stack([m[i,:,r:r+32,c0:c0+32] for r,c0 in positions])
        yy=np.stack([y[i,:,r:r+32,c0:c0+32] for r,c0 in positions]);yy=np.where(mm,yy/50,0).astype('f4')
        n=len(positions);sl=slice(cursor,cursor+n)
        z['input'][sl]=xx;z['target'][sl]=yy;z['mask'][sl]=mm
        z['timestamps'][sl]=stamp;z['source_timestamps'][sl]=source_stamp
        z['offset_seconds'][sl]=offset;z['patch_row'][sl]=[p[0] for p in positions];z['patch_col'][sl]=[p[1] for p in positions]
        z['source_sha256'][sl]=source_hash.encode();cursor+=n;support_count+=int(mm.sum());zero_count+=int((mm&(yy==0)).sum())
        if hour%500==0:print(json.dumps({'split':split,'hours_written':hour,'hours':len(plans),'patches':cursor}),flush=True)
    if cursor!=total:raise ValueError('Patch count mismatch')
    # Read every tensor chunk back before publishing; checksums bind all arrays.
    for start in range(0,total,48):
        xx=z['input'][start:start+48];yy=z['target'][start:start+48];mm=z['mask'][start:start+48]
        if not np.isfinite(xx).all() or not np.isfinite(yy).all() or (yy[~mm]!=0).any():raise ValueError('Tensor validation failed')
        if not np.isin(np.rint(yy[mm]*50),[0,20,25,30,35,40,45,50]).all():raise ValueError('Unexpected target code')
    record=dict(split=split,hours_selected=len(plans),hours_with_patches=used_hours,patches=total,
                supported_pixels_with_overlap=support_count,no_echo_pixels_with_overlap=zero_count,array_sha256=checksum_arrays(z))
    z.attrs['complete']=True;zarr.consolidate_metadata(str(tmp))
    # Write a recoverable pending record before rename, then atomically publish directory.
    dump(out/(split+'.pending.json'),record);tmp.rename(dest)
    (out/(split+'.pending.json')).replace(done)
    return record


def run(c,resume=False,check_only=False):
    started=time.monotonic();df,g,semantic=inputs(c);index=cache_index(c,semantic)
    if set(df.sha)-set(index):raise ValueError('Required radar sources absent from compact cache')
    dbpath=Path(c['pairing_dir'])/'era5_inventory_snapshot.sqlite';semantic['era5_inventory_sha256']=sha(dbpath)
    semantic['inventory_config']=read(Path(c['pairing_dir'])/'run_config.json')['inventory_config']
    semantic['radar_cache_index_sha256']=sha(Path(c['radar_cache_dir'])/'index.sqlite')
    print(json.dumps({'selected_hours':len(df),'split_hours':df.split.value_counts().to_dict(),
        'maximum_patches':len(df)*48,'uncompressed_patch_GiB':len(df)*48*(len(c['channels'])*4+5)*1024/2**30},indent=2),flush=True)
    if check_only:return
    with output_lock(c['output_dir'],resume,semantic) as out:
        for split in SPLITS:
            pending=out/(split+'.pending.json');dest=out/(split+'.zarr')
            if pending.exists() and dest.exists():
                if checksum_arrays(zarr.open_group(str(dest),mode='r'))!=read(pending)['array_sha256']:raise ValueError('Interrupted publication corrupt')
                pending.replace(out/(split+'.complete.json'))
        dump(out/'summary.json',{'version':VERSION,'complete':False,'production_ready':False})
        inventory=Inventory(c,out)
        try:stages=month_stages(c,df,g,out,index,inventory)
        finally:inventory.close()
        norm=normalization(stages,c['channels']);dump(out/'normalization.json',norm)
        np.savez(out/'normalization.npz',input_mean=np.asarray(norm['input_mean'],np.float32),input_std=np.asarray(norm['input_std'],np.float32),
                 target_mean=np.array([0],np.float32),target_std=np.array([1],np.float32))
        records=[write_split(c,s,stages,g,out,semantic) for s in SPLITS]
        df.to_parquet(out/'selected_manifest.parquet',index=False)
        summary=dict(version=VERSION,complete=True,production_ready=False,zarr_created=True,selected_hours=len(df),
            splits={r['split']:{k:v for k,v in r.items() if k not in ('split','array_sha256')} for r in records},
            channels=c['channels'],target_transform='target_code / 50',no_echo_rule_documented=True,
            temporal_strategy=semantic['temporal_contract']['strategy'],temporal_aggregation='none',
            coverage_state=semantic['coverage_state'],spatial_validity_confirmed=semantic['spatial_validity_confirmed'],
            corrdiff_native_tested=False,diagnostics_0_18_rerun=False,elapsed_seconds=round(time.monotonic()-started,2),
            artifact_sha256={n:sha(out/n) for n in ('contract.json','normalization.json','normalization.npz','selected_manifest.parquet','era5_identity.json')},
            note='Reference tensors built and read back. Scientific diagnostics and native CorrDiff integration are separate gates.')
        dump(out/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--contract-config',required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--check-only',action='store_true')
    a=p.parse_args(argv);run(load_config(a.contract_config),a.resume,a.check_only)
