#!/usr/bin/env python3
"""Regenerate mask-aware target diagnostics and frozen chronological split indices."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import zarr
from radar_qc import event_mask,sha256
from qc_dataset import SPLITS,BOUNDARIES,timestamps,split_codes,index_fingerprint

THRESHOLDS=[20,30,40,45]


def audit(dataset,batch_size):
    root=zarr.open_group(str(dataset/'train.zarr'),mode='r');ts,pos=timestamps(root)
    y=root['target'];m=root['mask'];codes=split_codes(ts)
    valid_counts=np.zeros(len(y),np.int64);dry_counts=np.zeros(len(y),np.int64)
    events=np.zeros((len(y),len(THRESHOLDS)),np.int64);target_sum=np.zeros(len(y),np.float64)
    for start in range(0,len(y),batch_size):
        end=min(start+batch_size,len(y));x=np.asarray(y[start:end],np.float32);mask=np.asarray(m[start:end])
        if not np.isin(mask,[0,1]).all():raise ValueError('Mask must be binary')
        valid=mask.astype(bool)
        if np.any(valid & (~np.isfinite(x)|(x<0))):raise ValueError('Nonfinite/negative valid target')
        axes=tuple(range(1,x.ndim))
        valid_counts[start:end]=valid.sum(axis=axes);dry_counts[start:end]=(valid & (x==0)).sum(axis=axes)
        target_sum[start:end]=np.where(valid,np.expm1(x.astype(np.float64)),0).sum(axis=axes)
        for j,th in enumerate(THRESHOLDS):events[start:end,j]=event_mask(x,valid,th).sum(axis=axes)
        if start%max(batch_size,20000//batch_size*batch_size)==0:print('audit',dataset,start,len(y),flush=True)
    patch=pd.DataFrame({'timestamp':ts,'year':ts.year,'split_code':codes,'patch_row':pos[:,0],'patch_col':pos[:,1],
        'valid_pixels':valid_counts,'dry_pixels':dry_counts,'total_pixels':int(np.prod(y.shape[1:])),
        'dbz_sum':target_sum})
    for j,th in enumerate(THRESHOLDS):patch[f'n_ge_{th}']=events[:,j]
    cols=['valid_pixels','dry_pixels','total_pixels','dbz_sum',*[f'n_ge_{th}' for th in THRESHOLDS]]
    time=patch.groupby(['timestamp','year','split_code'],sort=True)[cols].sum().reset_index()
    n_patches=patch.groupby('timestamp').size();time['n_patches']=time.timestamp.map(n_patches)
    time['split']=time.split_code.map(dict(enumerate(SPLITS)))
    records=[]
    for year,g in time.groupby('year'):
        denom=int(g.valid_pixels.sum());observed=g[g.valid_pixels>0]
        hours=(pd.Timestamp(year=year+1,month=1,day=1,tz='UTC')-pd.Timestamp(year=year,month=1,day=1,tz='UTC')).total_seconds()/3600
        row=dict(year=int(year),retained_hours=len(g),hours_with_valid_pixels=len(observed),expected_year_hours=int(hours),
            availability_ratio=len(g)/hours,n_patches=int(g.n_patches.sum()),valid_pixels=denom,
            valid_pixel_fraction=denom/int(g.total_pixels.sum()),dry_pixels=int(g.dry_pixels.sum()),
            mean_dbz_valid=float(g.dbz_sum.sum()/denom) if denom else np.nan)
        for th in THRESHOLDS:
            row[f'pixel_fraction_ge_{th}']=float(g[f'n_ge_{th}'].sum()/denom) if denom else np.nan
            row[f'timestamp_event_rate_ge_{th}']=float((observed[f'n_ge_{th}']>0).mean()) if len(observed) else np.nan
        records.append(row)
    return root,ts,pos,codes,patch,time,pd.DataFrame(records)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir',type=Path,required=True)
    p.add_argument('--reference-dataset-dir',type=Path)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=512)
    a=p.parse_args()
    if a.batch_size<1:p.error('Batch size must be positive')
    a.output_dir.mkdir(parents=True,exist_ok=False)
    root,ts,pos,codes,patch,time,yearly=audit(a.dataset_dir,a.batch_size)
    patch.to_parquet(a.output_dir/'patch_quality.parquet',index=False)
    time.to_parquet(a.output_dir/'timestamp_split_manifest.parquet',index=False)
    yearly.to_csv(a.output_dir/'yearly_masked_metrics.csv',index=False)
    for code,name in enumerate(SPLITS):np.save(a.output_dir/(name+'_patch_indices.npy'),np.flatnonzero(codes==code))
    summaries=[]
    for code,name in enumerate(SPLITS):
        g=time[time.split_code==code]
        summaries.append(dict(split=name,split_code=code,n_timestamps=len(g),n_patches=int(g.n_patches.sum()),
            first_timestamp_utc=str(g.timestamp.min()) if len(g) else None,last_timestamp_utc=str(g.timestamp.max()) if len(g) else None,
            valid_pixels=int(g.valid_pixels.sum()),empty_target_patches=int(((codes==code)&(patch.valid_pixels.to_numpy()==0)).sum()),
            locked=code in [0,3,4],model_selection_access=code in [1,2]))
    pd.DataFrame(summaries).to_csv(a.output_dir/'split_summary.csv',index=False)
    boundaries=[]
    for i,b in enumerate(BOUNDARIES,1):
        left=ts[codes==i];right=ts[codes==i+1]
        gap=(right.min()-left.max()).total_seconds()/3600 if len(left) and len(right) else None
        boundaries.append(dict(boundary=b,left=SPLITS[i],right=SPLITS[i+1],separation_hours=gap,
            passes=gap>=96 if gap is not None else None))
    pd.DataFrame(boundaries).to_csv(a.output_dir/'boundary_audit.csv',index=False)
    if any(b['passes'] is False for b in boundaries):raise ValueError('Temporal purge failed')
    splitconfig=dict(version='phase15-qc-mask-aware-v2',index_sha256=index_fingerprint(ts,pos),
        boundaries=BOUNDARIES,purge_each_side_hours=48,train_years=[2011,2020],validation_years=[2021,2022],
        primary_test_year=2023,stress_test_year=2024,split_names=SPLITS,
        phase17_channel_matrix_unchanged=True,locked_test_model_selection=False)
    (a.output_dir/'split_config.json').write_text(json.dumps(splitconfig,indent=2))
    if a.reference_dataset_dir:
        _,oldts,oldpos,_,_,oldtime,oldyear=audit(a.reference_dataset_dir,a.batch_size)
        oldyear.to_csv(a.output_dir/'reference_yearly_masked_metrics.csv',index=False)
        oldyear.merge(yearly,on='year',how='outer',suffixes=('_before','_after')).to_csv(a.output_dir/'before_after_yearly.csv',index=False)
        pd.DataFrame({'timestamp':oldtime.timestamp}).merge(pd.DataFrame({'timestamp':time.timestamp}),on='timestamp',how='outer',indicator=True).to_csv(a.output_dir/'timestamp_retention.csv',index=False)
    summary=dict(version='phase14-15-qc-mask-aware-v2',patches=len(ts),timestamps=len(time),
        index_sha256=splitconfig['index_sha256'],complete=True,production_ready=False,
        valid_pixels=int(patch.valid_pixels.sum()),dry_pixels=int(patch.dry_pixels.sum()),
        empty_target_patches=int((patch.valid_pixels==0).sum()),
        patches_per_timestamp_min=int(time.n_patches.min()),patches_per_timestamp_max=int(time.n_patches.max()),
        spatial_scope='patch-weighted; overlaps count repeatedly; all fractions use valid denominators',
        thresholds='float32 log1p comparisons',test_model_selection=False,
        prerequisite_warning='New missing-data policy changes estimand. Old regime/ERA5 diagnostics need regeneration if timestamp set changes.')
    (a.output_dir/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
