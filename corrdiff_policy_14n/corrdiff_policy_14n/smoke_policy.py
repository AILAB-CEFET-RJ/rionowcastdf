#!/usr/bin/env python3
"""Train-only technical test of 14n no-echo + echo targets; not native CorrDiff."""
import argparse
import json
import sqlite3
from pathlib import Path
import sys
import time

import duckdb
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from pair_contract import CHANNELS, digest, dump, qi, literal, choose

VERSION='14n-policy-smoke-v1'

def strict_bool(series):
    if series.isna().any() or not series.isin([True,False]).all():
        raise ValueError('Expected non-null boolean column: '+series.name)
    return series.astype(bool)

def select_train(manifest, maximum):
    df=manifest.copy()
    df['timestamp_utc']=pd.to_datetime(df.timestamp_utc,utc=True)
    if df.timestamp_utc.duplicated().any(): raise ValueError('Duplicate timestamps')
    eligible=(df.split.eq('train') & ~strict_bool(df.locked) &
      strict_bool(df.policy_train_candidate) & strict_bool(df.policy_pair_candidate) &
      strict_bool(df.era5_technical_candidate) & strict_bool(df.in_archive_period) &
      ~strict_bool(df.local_filename_ambiguous) & df.observation_pixels.gt(0))
    df=df[eligible].sort_values('timestamp_utc')
    if len(df) and ((df.timestamp_utc<pd.Timestamp('2011-01-17T02:00Z'))|
                   (df.timestamp_utc>=pd.Timestamp('2021-01-01',tz='UTC'))).any():
        raise ValueError('Candidate outside train years/archive')
    if not len(df): raise ValueError('No policy train candidates')
    # Explicit diagnostic stratification, NOT the final training sampler.
    chosen=[]
    for group in (df[df.echo_pixels.eq(0)],df[df.echo_pixels.gt(0)]):
        if len(group):
            take=np.unique(np.linspace(0,len(group)-1,min((maximum+1)//2,len(group))).astype(int))
            chosen.extend(group.iloc[take].index.tolist())
    chosen=list(dict.fromkeys(chosen))[:maximum]
    remaining=df.drop(index=chosen)
    if len(chosen)<maximum and len(remaining):
        take=np.unique(np.linspace(0,len(remaining)-1,min(maximum-len(chosen),len(remaining))).astype(int))
        chosen.extend(remaining.iloc[take].index.tolist())
    return df.loc[chosen].sort_values('timestamp_utc').copy()

def regrid(rows, lat, lon):
    if rows.empty: raise ValueError('No ERA5 rows at exact UTC timestamp')
    if rows[['lat','lon']].duplicated().any(): raise ValueError('Duplicate ERA5 coordinates')
    if not np.isfinite(rows[['lat','lon','value']].to_numpy(dtype=float)).all():
        raise ValueError('ERA5 contains nonfinite values/coordinates')
    ys=np.sort(rows.lat.unique());xs=np.sort(rows.lon.unique())
    if len(ys)<2 or len(xs)<2 or len(rows)!=len(ys)*len(xs): raise ValueError('Incomplete ERA5 rectangular grid')
    values=rows.pivot(index='lat',columns='lon',values='value').reindex(index=ys,columns=xs).to_numpy()
    interp=RegularGridInterpolator((ys,xs),values,method='linear',bounds_error=True)
    target=interp(np.column_stack([lat.ravel(),lon.ravel()])).reshape(lat.shape).astype(np.float32)
    if not np.isfinite(target).all(): raise ValueError('Nonfinite ERA5 interpolation')
    return target

def read_channel(engine, db, stamp, ch, config, lat, lon, checked, provenance):
    sources=db.execute('SELECT DISTINCT path FROM metrics WHERE timestamp_utc=? AND channel=?',(stamp.isoformat(),ch)).fetchall()
    if len(sources)!=1: raise ValueError(f'{stamp} {ch}: expected exactly one source file, got {len(sources)}')
    path=Path(sources[0][0])
    stored=db.execute('SELECT fingerprint,status,details FROM files WHERE path=?',(str(path),)).fetchone()
    if not stored or stored[1]!='ok': raise ValueError('Source was not successfully inventoried')
    stat=path.stat(); fp=f'{stat.st_size}:{stat.st_mtime_ns}'
    if fp!=stored[0]: raise ValueError('ERA5 changed since inventory: '+str(path))
    details=json.loads(stored[2])
    if str(path) not in checked:
        checked.add(str(path));provenance.append({'path':str(path),'fingerprint':fp,'metadata':details.get('metadata',{})})
    kind='pressure' if '_' in ch else 'single'
    var,level=ch.rsplit('_',1) if kind=='pressure' else (ch,None)
    col=details['variable_columns'][var];t=details['time_column'];names=details['columns']
    la=choose(names,['latitude','lat']);lo=choose(names,['longitude','lon'])
    where=f'try_cast({qi(t)} AS TIMESTAMPTZ)={literal(stamp.isoformat())}::TIMESTAMPTZ'
    if level:
        lv=details['pressure_column'];div='/100.0' if config['pressure_unit']=='Pa' else ''
        where+=f' AND try_cast({qi(lv)} AS DOUBLE){div}={int(level)}'
    sql=f'''SELECT try_cast({qi(la)} AS DOUBLE) AS lat,
      ((try_cast({qi(lo)} AS DOUBLE)+180)%360+360)%360-180 AS lon,
      try_cast({qi(col)} AS DOUBLE) AS value
      FROM read_parquet({literal(path)},hive_partitioning=false) WHERE {where}'''
    rows=engine.execute(sql).fetchdf()
    after=path.stat()
    if fp!=f'{after.st_size}:{after.st_mtime_ns}': raise ValueError('ERA5 changed during read')
    return regrid(rows,lat,lon)

def inside(root,relative):
    path=(root/relative).resolve()
    if root not in path.parents: raise ValueError('Cache path escapes root')
    return path

def read_radar(row,policy,shape,geometry_mask,palette):
    path=inside(policy,row.target_path)
    if digest(path)!=row.target_sha256: raise ValueError('Policy target checksum mismatch')
    with np.load(path,allow_pickle=False) as z:
        raw=z['target_code'].copy();mask=z['observation_mask'].copy();classes=z['observation_class']
        if raw.shape!=shape or mask.shape!=shape or mask.dtype!=bool:raise ValueError('Target shape/dtype mismatch')
        if (mask & ~geometry_mask).any():raise ValueError('Target exceeds support region')
        if not np.array_equal(mask,classes!=0):raise ValueError('Target classes disagree with support')
        if not np.isfinite(raw[mask]).all() or not np.isnan(raw[~mask]).all():raise ValueError('Target finite/unknown contract broken')
        if not np.all(raw[classes==1]==0) or not np.isin(raw[classes==2],palette).all():raise ValueError('Target encoding broken')
    return raw,mask

def patches(x,raw,support,size=32,stride=16):
    items=[];records=[]
    for y in range(0,x.shape[-2]-size+1,stride):
        for col in range(0,x.shape[-1]-size+1,stride):
            sl=np.s_[y:y+size,col:col+size]
            m=support[sl];n=int(m.sum())
            records.append({'row_start':y,'col_start':col,'support_pixels':n,'included':n>0})
            if n:
                t=raw[sl]
                items.append((x[:,y:y+size,col:col+size],t[None],m[None],y,col))
    return items,records

def masked_mse(pred,target,mask):
    if pred.shape!=target.shape or mask.shape!=target.shape: raise ValueError('Loss shapes differ')
    if mask.dtype!=torch.bool: raise ValueError('Mask must be bool')
    if not mask.any(): raise ValueError('Batch has no supported target pixels')
    # Index BEFORE arithmetic: NaN * zero would still be NaN.
    p=pred[mask];t=target[mask]
    if not torch.isfinite(p).all() or not torch.isfinite(t).all(): raise ValueError('Nonfinite values inside loss support')
    return ((p-t)**2).mean()

class TinyRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(12,16,3,padding=1),nn.SiLU(),nn.Conv2d(16,16,3,padding=1),nn.SiLU(),nn.Conv2d(16,1,1))
    def forward(self,x): return self.net(x)

def train_smoke(x,y,mask,out,steps,batch_size,device,seed):
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    dataset=TensorDataset(torch.from_numpy(x),torch.from_numpy(y),torch.from_numpy(mask))
    gen=torch.Generator().manual_seed(seed)
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=True,generator=gen,num_workers=0)
    model=TinyRegressor().to(device);optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    before=[p.detach().clone() for p in model.parameters()]
    records=[];iterator=iter(loader)
    for step in range(steps):
        try:bx,by,bm=next(iterator)
        except StopIteration:iterator=iter(loader);bx,by,bm=next(iterator)
        bx,by,bm=bx.to(device),by.to(device),bm.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred=model(bx)
        if not torch.isfinite(pred).all():raise ValueError('Nonfinite forward output')
        loss=masked_mse(pred,by,bm);loss.backward()
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):raise ValueError('Missing/nonfinite gradients')
        grad_norm=float(torch.sqrt(sum((g*g).sum() for g in grads)).detach().cpu())
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in model.parameters()): raise ValueError('Nonfinite parameters after update')
        records.append({'step':step+1,'loss_masked_mse':float(loss.detach().cpu()),'support_pixels':int(bm.sum()),'gradient_norm':grad_norm})
        print(f'Step {step+1}/{steps}: loss={records[-1]["loss_masked_mse"]:.6f}',flush=True)
    changed=any(not torch.equal(a,p.detach()) for a,p in zip(before,model.parameters()))
    if not changed: raise ValueError('No parameter changed')
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},out/'diagnostic_model_state.pt')
    pd.DataFrame(records).to_csv(out/'training_steps.csv',index=False)
    return {'steps':steps,'parameters_changed':changed,'all_steps_finite':True,'model':'TinyRegressor; not CorrDiff',
      'initial_batch_loss':records[0]['loss_masked_mse'],'final_batch_loss':records[-1]['loss_masked_mse']}

def run(args):
    policy=Path(args.policy_dir).resolve();out=Path(args.output_dir).resolve()
    policyconfig=json.loads((policy/'policy_config.json').read_text())
    policysummary=json.loads((policy/'summary.json').read_text())
    if policysummary.get('version')!='14n-producer-policy-v1' or not policysummary.get('complete') or policysummary.get('object_error_hours',1):raise ValueError('Need completed error-free 14n policy')
    for name,expected in policysummary['artifact_sha256'].items():
        if digest(inside(policy,name))!=expected:raise ValueError('Policy artifact changed: '+name)
    pairdir=Path(policyconfig['pairing_dir']);cache=Path(policyconfig['cache_dir'])
    if any(out==r or out in r.parents or r in out.parents for r in (pairdir,cache,policy)):raise ValueError('Output and inputs must be separate')
    if out.exists():raise ValueError('Use a new output directory')
    if digest(pairdir/'run_config.json')!=policyconfig['pairing_run_config_sha256']:raise ValueError('Base pairing config changed')
    runconfig=json.loads((pairdir/'run_config.json').read_text())
    summary=json.loads((pairdir/'summary.json').read_text())
    if summary.get('version')!='14k-pairing-v1' or not summary.get('complete'):raise ValueError('Expected completed14k pairing export')
    if summary.get('provisional'):raise ValueError('Partial cache snapshot not allowed')
    if summary.get('era5_error_files',1):raise ValueError('ERA5 inventory contains errors')
    for path,expected in [(cache/'config.json',runconfig['radar_config_sha256']),
        (cache/'geometry.npz',runconfig['geometry_sha256']),
        (pairdir/'era5_inventory_snapshot.sqlite',runconfig['era5_snapshot_sha256'])]:
        if digest(path)!=expected:raise ValueError('Input contract checksum mismatch: '+str(path))
    if digest(pairdir/'paired_manifest.parquet')!=policyconfig['paired_manifest_sha256']:raise ValueError('Base pairing changed')
    manifest=pd.read_parquet(policy/'policy_manifest.parquet')
    selected=select_train(manifest,args.max_timestamps)
    purge=int(runconfig['args']['purge_hours'])
    if (selected.timestamp_utc>=pd.Timestamp('2021-01-01',tz='UTC')-pd.Timedelta(hours=purge)).any():
        raise ValueError('Candidate inside temporal purge')
    with np.load(policy/'coverage.npz',allow_pickle=False) as g:
        lat=g['lat'].copy();lon=g['lon'].copy();geom=g['observed_region_mask'].copy()
    config=json.loads((cache/'config.json').read_text())
    palette=config['reference_contract']['palette_dbz']
    if lat.shape!=(70,84) or lon.shape!=lat.shape or geom.shape!=lat.shape:raise ValueError('Expected70x84 geometry')
    out.mkdir(parents=True)
    dump(out/'summary.json',{'version':VERSION,'complete':False,'production_ready':False})
    selected.to_csv(out/'selected_train_timestamps.csv',index=False)
    db=sqlite3.connect((pairdir/'era5_inventory_snapshot.sqlite').as_uri()+'?mode=ro',uri=True)
    engine=duckdb.connect();engine.execute("SET TimeZone='UTC'");engine.execute('SET threads=2');engine.execute("SET memory_limit='1GB'")
    engine.execute('SET temp_directory='+literal(out/'duckdb_tmp'))
    provenance=[];checked=set();fields=[];targets=[];supports=[]
    try:
        for i,row in enumerate(selected.itertuples(index=False),1):
            stamp=row.timestamp_utc
            channels=[read_channel(engine,db,stamp,ch,runconfig['inventory_config'],lat,lon,checked,provenance) for ch in CHANNELS]
            fields.append(np.stack(channels))
            raw,support=read_radar(row,policy,lat.shape,geom,palette)
            targets.append(raw);supports.append(support)
            print(f'Loaded {i}/{len(selected)} {stamp.isoformat()} support={int(support.sum())}',flush=True)
    finally:engine.close();db.close()
    fields=np.stack(fields)
    # Fit on unique selected TRAIN frames before patch overlap; diagnostic subset only.
    mean=fields.mean(axis=(0,2,3),dtype=np.float64)
    std=fields.std(axis=(0,2,3),dtype=np.float64)
    scale=np.where(std<1e-8,1.,std)
    normalized=((fields-mean[None,:,None,None])/scale[None,:,None,None]).astype(np.float32)
    if not np.isfinite(normalized).all():raise ValueError('Nonfinite normalized conditioning')
    dump(out/'diagnostic_normalization.json',{'channels':CHANNELS,'mean':mean.tolist(),'std':std.tolist(),'scale':scale.tolist(),
      'fit_scope':'selected unique train frames only; NOT production normalization','constant_channels':[ch for ch,v in zip(CHANNELS,std) if v<1e-8]})
    allitems=[];audit=[];patchmeta=[]
    for row,x,raw,support in zip(selected.itertuples(index=False),normalized,targets,supports):
        items,records=patches(x,raw,support)
        for record in records:audit.append({'timestamp_utc':row.timestamp_utc.isoformat(),**record})
        for item in items:
            xx,yy,mm,iy,ix=item;allitems.append((xx,yy,mm))
            patchmeta.append({'timestamp_utc':row.timestamp_utc.isoformat(),'split':'train','row_start':iy,'col_start':ix,'support_pixels':int(mm.sum())})
    pd.DataFrame(audit).to_csv(out/'patch_audit.csv',index=False)
    if not allitems:raise ValueError('No supported pixels in any32x32 patch; no training performed')
    x=np.stack([i[0] for i in allitems]);raw=np.stack([i[1] for i in allitems]);mask=np.stack([i[2] for i in allitems])
    # Zero outside support is only a storage placeholder; excluded BEFORE loss arithmetic.
    y=np.where(mask,raw/50.,0.).astype(np.float32)
    np.savez_compressed(out/'smoke_tensors.npz',conditioning=x,target_code=raw,target_tensor=y,observation_support=mask,lat=lat,lon=lon)
    pd.DataFrame(patchmeta).to_csv(out/'patch_manifest.csv',index=False)
    result=train_smoke(x,y,mask,out,args.steps,args.batch_size,args.device,args.seed)
    dump(out/'era5_source_provenance.json',provenance)
    dump(out/'run_config.json',{'version':VERSION,'args':vars(args),'channels':CHANNELS,'code_sha256':digest(Path(__file__)),
      'contract_code_sha256':digest(Path(__file__).with_name('pair_contract.py')),
      'policy_manifest_sha256':digest(policy/'policy_manifest.parquet'),'policy_config_sha256':digest(policy/'policy_config.json'),'coverage_state':policysummary['coverage_state'],'geometry_sha256':digest(cache/'geometry.npz'),
      'input_interpolation':'bilinear on lat/lon rectilinear axes; no extrapolation',
      'target_scaling':'diagnostic target_code/50; zero denotes no displayed echo, not measured 0 dBZ; NOT final transform',
      'mask_semantics':'producer no-echo plus palette inside declared coverage; unknown excluded; coverage status explicit',
      'cache_provisional':summary.get('provisional',True)})
    final={'version':VERSION,'complete':True,'technical_smoke_passed':True,'production_ready':False,
      'no_echo_rule_documented':True,'coverage_state':policysummary['coverage_state'],'corrdiff_native_tested':False,'selected_train_hours':len(selected),
      'patches_attempted':len(audit),'patches_used':len(allitems),'empty_patches_skipped':len(audit)-len(allitems),
      'conditioning_shape':list(x.shape),'target_shape':list(y.shape),'support_pixels_with_patch_overlap':int(mask.sum()),
      'support_fraction_with_patch_overlap':float(mask.mean()),'cache_provisional':summary.get('provisional',True),
      'selected_no_echo_only_hours':int(selected.echo_pixels.eq(0).sum()),'selected_echo_hours':int(selected.echo_pixels.gt(0).sum()),
      'training':result,'note':'Technical test only. No validation/test used. Zero is a category code. Coverage and final target remain explicit methodological decisions.'}
    dump(out/'summary.json',final)
    return final

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy-dir',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--max-timestamps',type=int,default=8);p.add_argument('--steps',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=4);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    if not 1<=args.max_timestamps<=64:p.error('max-timestamps must be1..64')
    if not 1<=args.steps<=1000 or args.batch_size<1:p.error('steps must be1..1000 and batch-size positive')
    if args.device=='cuda' and not torch.cuda.is_available():p.error('CUDA unavailable; use CPU')
    print(json.dumps(run(args),indent=2))

if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:print('Interrupted; use a new output directory for the next run.',file=sys.stderr);sys.exit(130)
    except Exception as e:print(f'{type(e).__name__}: {e}',file=sys.stderr);sys.exit(1)
