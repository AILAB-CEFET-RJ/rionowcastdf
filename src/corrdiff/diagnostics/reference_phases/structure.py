import itertools
import json
import numpy as np
import pandas as pd
from .common import SPLITS,CODES,sha,csv,dump,split_for,derived


def phase15(ctx,out):
    """Full metadata scan, not dependent on numerical diagnostics."""
    root=ctx.reader('train').root
    summary=ctx.reader('train').summary
    if sha(root/'selected_manifest.parquet')!=summary['artifact_sha256']['selected_manifest.parquet']:
        raise ValueError('Selected manifest hash mismatch')
    manifest=pd.read_parquet(root/'selected_manifest.parquet')
    required={'timestamp_utc','source_utc','split','path','sha','offset_seconds'}
    if not required.issubset(manifest):raise ValueError('Manifest columns missing: '+str(required-set(manifest)))
    for k in ('timestamp_utc','source_utc'):
        manifest[k]=pd.to_datetime(manifest[k],utc=True)
    manifest['timestamp_seconds']=manifest.timestamp_utc.astype('int64')//10**9
    manifest['source_seconds']=manifest.source_utc.astype('int64')//10**9
    if manifest.timestamp_utc.duplicated().any():raise ValueError('Repeated reference hours in manifest')
    if not manifest.split.isin(SPLITS).all():raise ValueError('Unknown split in manifest')
    if manifest.groupby('path').split.nunique().gt(1).any():raise ValueError('Same PNG path in different splits')
    if manifest.groupby('source_seconds').split.nunique().gt(1).any():raise ValueError('Same observation time in different splits')
    if not np.array_equal(split_for(manifest.timestamp_seconds),manifest.split):raise ValueError('Reference violates frozen split/purge')
    if not np.array_equal(split_for(manifest.source_seconds),manifest.split):raise ValueError('Source violates frozen split/purge')
    offsets=manifest.source_seconds-manifest.timestamp_seconds
    if (offsets.abs()>120).any() or not np.array_equal(offsets,manifest.offset_seconds):raise ValueError('Invalid nearest2 offsets')
    if (manifest.timestamp_seconds%3600!=0).any():raise ValueError('Non-hourly references')
    boundary=int(pd.Timestamp(ctx.c['archive_start_utc']).timestamp())
    end=int(pd.Timestamp(ctx.c['archive_end_exclusive_utc']).timestamp())
    if ((manifest.timestamp_seconds<boundary)|(manifest.timestamp_seconds>=end)).any():raise ValueError('References outside archive bounds')
    reports=[];hour_parts=[]
    all_sources={}
    for split in SPLITS:
        r=ctx.reader(split);z=r.z
        times=z['timestamps'][:]
        if np.any(np.diff(times)<0):raise ValueError('Unordered timestamps')
        hours,start,count=np.unique(times,return_index=True,return_counts=True)
        rows=manifest[manifest.split==split].sort_values('timestamp_seconds')
        if not np.array_equal(rows.timestamp_seconds,hours):raise ValueError('Manifest/Zarr hours mismatch: '+split)
        source=z['source_timestamps'][:];offset=z['offset_seconds'][:];hashes=z['source_sha256'][:]
        expected_source=rows.source_seconds.to_numpy()
        expected_hash=rows.sha.to_numpy(dtype='S64')
        for vals,want,label in ((source,expected_source,'source time'),(offset,rows.offset_seconds.to_numpy(),'offset'),(hashes,expected_hash,'hash')):
            if not np.array_equal(vals,np.repeat(want,count)):raise ValueError('Patch metadata mismatch: '+label)
        pr=z['patch_row'][:];pc=z['patch_col'][:]
        allowed={(rr,cc) for rr in list(range(0,39,8))+[38] for cc in list(range(0,53,8))+[52]}
        for st,n in zip(start,count):
            positions=list(zip(pr[st:st+n].tolist(),pc[st:st+n].tolist()))
            if len(set(positions))!=n or not set(positions)<=allowed:raise ValueError('Invalid/duplicated patch position')
        recorded=summary['splits'][split]
        if len(hours)!=recorded['hours_selected'] or len(times)!=recorded['patches']:raise ValueError('Summary counts mismatch')
        all_sources[split]=np.unique(expected_source)
        part=rows[['timestamp_seconds','source_seconds','split','path','sha','offset_seconds']].copy()
        part['patches']=count;hour_parts.append(part)
        reports.append(dict(split=split,hours=len(hours),patches=len(times),min_patches=int(count.min()) if len(count) else 0,
            max_patches=int(count.max()) if len(count) else 0,first_utc=str(rows.timestamp_utc.min()),last_utc=str(rows.timestamp_utc.max())))
        print(f'phase15 {split}: {len(hours)} hours, {len(times)} patches',flush=True)
    distances=[]
    for a,b in itertools.combinations(SPLITS,2):
        aa,bb=all_sources[a],all_sources[b]
        if not len(aa) or not len(bb):continue
        ix=np.searchsorted(bb,aa);near=np.minimum(abs(aa-bb[np.clip(ix,0,len(bb)-1)]),abs(aa-bb[np.clip(ix-1,0,len(bb)-1)]))
        distances.append(dict(split_a=a,split_b=b,min_source_gap_seconds=int(near.min())))
    identity=pd.concat(hour_parts,ignore_index=True)
    identity.to_parquet(out/'hour_identity.parquet',index=False)
    csv(out/'split_counts.csv',reports);csv(out/'cross_split_source_distance.csv',distances)
    repeated=identity.groupby('sha').agg(hours=('sha','size'),splits=('split','nunique')).reset_index()
    repeated[repeated.splits>1].to_csv(out/'shared_content_hashes.csv',index=False)
    return dict(passed=True,scope='all manifest rows and all patch metadata',hours=len(manifest),
        same_content_across_splits_is_not_proof_of_leakage=True,split_counts=reports,
        purge_hours_each_side=48,test_targets_read=False)


def phase0(ctx,out):
    all_rows=[];stats=[];coverage=[]
    for split in SPLITS:
        r=ctx.reader(split);hours=r.hours
        limit=ctx.c['phase0_max_hours_per_split']
        if limit:hours=ctx.selected(split,limit)
        nc=len(r.channels);n=np.zeros(nc,dtype='i8');mean=np.zeros(nc);m2=np.zeros(nc)
        low=np.full(nc,np.inf);high=np.full(nc,-np.inf)
        valid_grid=np.zeros((70,84),dtype='i8');echo_grid=np.zeros_like(valid_grid)
        for index,t in enumerate(hours):
            h=r.hour(int(t));mask=h['mask'][0];x=h['input'];y=h['target'][0]
            if np.any(mask & ~r.support.astype(bool)):raise ValueError('Valid target outside contracted support')
            if not (h['patch_count']>0).all():raise ValueError('Incomplete grid reconstruction')
            row=dict(timestamp_utc=int(t),split=split,valid_pixels=int(mask.sum()),
                no_echo_pixels=int(h['no_echo'].sum()),covered_grid_points=int((h['patch_count']>0).sum()),
                max_code=float(h['target_code'][h['mask']].max()) if mask.any() else None,
                mean_target_code=float(h['target_code'][h['mask']].mean()) if mask.any() else None,
                patches=len(h['metadata']['timestamps']),source_utc=int(h['metadata']['source_timestamps'][0]))
            for code in CODES:
                row[f'code_{int(code)}']=int((mask & (y==np.float32(code/50))).sum())
            for code in CODES[1:]:row[f'ge_{int(code)}']=int((mask & (y>=np.float32(code/50))).sum())
            for j,name in enumerate(r.channels):
                values=x[j][np.isfinite(x[j])].astype('f8');size=len(values)
                mu=values.mean();delta=mu-mean[j];total=n[j]+size
                m2[j]+=np.square(values-mu).sum()+delta*delta*n[j]*size/total
                mean[j]+=delta*size/total;n[j]=total
                low[j]=min(low[j],values.min());high[j]=max(high[j],values.max())
                row[name]=float(mu)
                if name in ('r_850','r_500'):row[name+'_outside_0_100']=int(((values<0)|(values>100)).sum())
            for name,value in derived(x,r.channels).items():row[name]=float(np.nanmean(value))
            all_rows.append(row);valid_grid+=mask;echo_grid+=mask&(y>=np.float32(.4))
            if (index+1)%500==0:print(f'phase0 {split}: {index+1}/{len(hours)} hours',flush=True)
        for j,name in enumerate(r.channels):stats.append(dict(split=split,channel=name,n=int(n[j]),mean=mean[j],std=np.sqrt(m2[j]/n[j]) if n[j] else None,min=low[j],max=high[j]))
        np.savez_compressed(out/f'{split}_spatial_counts.npz',valid=valid_grid,echo=echo_grid,lat=r.lat,lon=r.lon,support=r.support)
        coverage.append(dict(split=split,hours_available=len(r.hours),hours_scanned=len(hours)))
    table=pd.DataFrame(all_rows);table.to_parquet(out/'hour_metrics.parquet',index=False)
    csv(out/'input_statistics.csv',stats);csv(out/'scan_scope.csv',coverage)
    normalization=json.loads((r.root/'normalization.json').read_text())
    train_stats=[row for row in stats if row['split']=='train']
    comparison=[]
    for j,row in enumerate(train_stats):
        comparison.append(dict(channel=row['channel'],recomputed_mean=row['mean'],
            stored_mean=normalization['input_mean'][j],mean_difference=row['mean']-normalization['input_mean'][j],
            recomputed_std=row['std'],stored_std=normalization['input_std'][j],
            std_difference=row['std']-normalization['input_std'][j],points=row['n']))
    csv(out/'train_normalization_comparison.csv',comparison)
    expected=pd.date_range(ctx.c['archive_start_utc'],ctx.c['archive_end_exclusive_utc'],freq='h',inclusive='left')
    seconds=(expected.astype('int64')//10**9).to_numpy()
    calendar=pd.DataFrame(dict(timestamp_utc=seconds,split=split_for(seconds)))
    calendar['selected']=calendar.timestamp_utc.isin(pd.concat([pd.Series(ctx.reader(s).hours) for s in SPLITS]))
    calendar['year']=expected.year;calendar['month']=expected.month;calendar['hour_utc']=expected.hour
    calendar.groupby(['split','year','month'],as_index=False).agg(expected_hours=('selected','size'),selected_hours=('selected','sum')).to_csv(out/'monthly_coverage.csv',index=False)
    table.groupby('split',as_index=False)[['valid_pixels','no_echo_pixels']+[f'ge_{int(c)}' for c in CODES[1:]]].sum().to_csv(out/'target_counts.csv',index=False)
    return dict(passed=True,full_tensor_scan=ctx.c['phase0_max_hours_per_split']==0,
        scope='unique grid points per hour; predictor stats over all covered grid points',
        hours_scanned=len(table),coverage_state=r.contract.get('coverage_state'),spatial_validity_confirmed=False,
        raw_sources_rechecked=False,scan_scope=coverage)
