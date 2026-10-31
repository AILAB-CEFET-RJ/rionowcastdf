#!/usr/bin/env python3
"""Train-only climatologies, validation-only evaluation with explicit masks."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import zarr
from scipy.ndimage import uniform_filter
from radar_qc import event_mask
from qc_dataset import timestamps, split_codes, index_fingerprint

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir',type=Path,required=True)
    p.add_argument('--splits-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=256)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=False)
    root=zarr.open_group(str(a.dataset_dir/'train.zarr'),mode='r')
    y=root['target'];mask=root['mask']
    ts,pos=timestamps(root);local=ts.tz_convert('America/Sao_Paulo')
    positions=[tuple(v) for v in pos.tolist()]
    split_config=json.loads((a.splits_dir/'split_config.json').read_text())
    if split_config.get('index_sha256')!=index_fingerprint(ts,pos):raise ValueError('Splits belong to a different dataset')
    slots={pos:i for i,pos in enumerate(sorted(set(positions)))}
    shape=tuple(y.shape[1:]); nslots=len(slots)
    sums=np.zeros((12,24,nslots,*shape),np.float64);counts=np.zeros_like(sums,dtype=np.uint32)
    train=np.load(a.splits_dir/'train_patch_indices.npy');val=np.load(a.splits_dir/'validation_patch_indices.npy')
    if a.batch_size<1:raise ValueError('Batch size must be positive')
    codes=split_codes(ts)
    for idx,code in [(train,1),(val,2)]:
        if not np.array_equal(idx,np.flatnonzero(codes==code)):raise ValueError('Split indices do not match frozen chronology and purge')
    if not len(train) or not len(val):raise ValueError('Empty train/validation')
    if (ts[train].year>2020).any() or not np.isin(ts[val].year,[2021,2022]).all():raise ValueError('Invalid splits')
    input_sum=np.zeros(root['input'].shape[1],np.float64)
    input_sq=np.zeros_like(input_sum);input_count=np.zeros_like(input_sum,dtype=np.int64)
    for start in range(0,len(train),a.batch_size):
        indices=train[start:start+a.batch_size].astype(int)
        xb=np.asarray(root['input'].oindex[indices],dtype=np.float64)
        finite=np.isfinite(xb)
        input_sum+=np.where(finite,xb,0).sum(axis=(0,2,3))
        input_sq+=np.where(finite,xb*xb,0).sum(axis=(0,2,3))
        input_count+=finite.sum(axis=(0,2,3))
        yb=np.asarray(y.oindex[indices]);mb=np.asarray(mask.oindex[indices])>0
        for j,i in enumerate(indices):
            x=yb[j];m=mb[j]
            if np.any(m & ~np.isfinite(x)):raise ValueError('Nonfinite valid target')
            key=(local[i].month-1,local[i].hour,slots[positions[i]])
            sums[key]+=np.where(m,x,0);counts[key]+=m
        if start%max(a.batch_size,20000//a.batch_size*a.batch_size)==0:print('fit',start,len(train),flush=True)
    if np.any(input_count==0):raise ValueError('Input channel without finite train values')
    input_mean=input_sum/input_count
    input_std=np.sqrt(np.maximum(input_sq/input_count-input_mean**2,0))
    if np.any(input_std==0):raise ValueError('Constant input channel; review normalisation')
    np.savez(a.output_dir/'train_input_normalization.npz',mean=input_mean,std=input_std,count=input_count)
    def mean(s,c):return np.divide(s,c,out=np.full(s.shape,np.nan),where=c>0).astype(np.float32)
    clim=mean(sums,counts);slot=mean(sums.sum(axis=(0,1)),counts.sum(axis=(0,1)))
    glob=mean(sums.sum(axis=(0,1,2)),counts.sum(axis=(0,1,2)))
    np.savez_compressed(a.output_dir/'train_climatologies.npz',climatology=clim,slot=slot,global_field=glob,counts=counts)
    lookup={(t.value,pos):i for i,(t,pos) in enumerate(zip(ts,positions))}
    records={}
    validation_valid_pixels=0
    common_valid_pixels=0
    persistence_valid_pixels=0
    def accumulate(name,cohort,pred,x,valid):
        # Nonzero baseline with no training support is not silently replaced with dry.
        v=valid & np.isfinite(pred)
        r=records.setdefault((name,cohort),dict(n=0,sse=0.,sae=0.,bias=0.,tp=np.zeros(4),fp=np.zeros(4),fn=np.zeros(4),fss_num=np.zeros((4,4)),fss_den=np.zeros((4,4))))
        d=np.expm1(pred[v].astype(np.float64))-np.expm1(x[v].astype(np.float64))
        r['n']+=len(d);r['sse']+=float(d@d);r['sae']+=float(abs(d).sum());r['bias']+=float(d.sum())
        for j,th in enumerate([20,30,40,45]):
            obs=event_mask(x,v,th);forecast=event_mask(pred,v,th)
            r['tp'][j]+=np.sum(obs&forecast);r['fp'][j]+=np.sum(~obs&forecast&v);r['fn'][j]+=np.sum(obs&~forecast)
            for l,size in enumerate([1,2,4,8]):
                # 2km grid. Identical valid support in both fractions, no padding as dry.
                support=uniform_filter(v[0].astype(float),size=size,mode='constant',cval=0)
                of=uniform_filter(obs[0].astype(float),size=size,mode='constant',cval=0)
                pf=uniform_filter(forecast[0].astype(float),size=size,mode='constant',cval=0)
                good=support>0;of=of[good]/support[good];pf=pf[good]/support[good]
                r['fss_num'][j,l]+=np.sum((of-pf)**2);r['fss_den'][j,l]+=np.sum(of**2+pf**2)
    validation_set=set(val.tolist())
    for k,i in enumerate(val):
        if k%a.batch_size==0:
            batch=val[k:k+a.batch_size].astype(int)
            ybatch=np.asarray(y.oindex[batch]);mbatch=np.asarray(mask.oindex[batch])>0
            prevs=[lookup.get((ts[int(n)].value-3600*10**9,positions[int(n)])) for n in batch]
            prior=np.array(sorted({n for n in prevs if n in validation_set}),dtype=int)
            prior_data={}
            if len(prior):
                yp=np.asarray(y.oindex[prior]);mp=np.asarray(mask.oindex[prior])>0
                prior_data={int(n):(yp[j],mp[j]) for j,n in enumerate(prior)}
        i=int(i);x=ybatch[k%a.batch_size];v=mbatch[k%a.batch_size];s=slots[positions[i]]
        if np.any(v & ~np.isfinite(x)):raise ValueError('Nonfinite validation target')
        c=clim[local[i].month-1,local[i].hour,s]
        c=np.where(np.isfinite(c),c,slot[s]);c=np.where(np.isfinite(c),c,glob)
        preds={'zero':np.zeros_like(x),'train_global_mean_field':glob,
               'train_slot_mean':np.where(np.isfinite(slot[s]),slot[s],glob),'climatology_month_hour_slot':c}
        common=v & np.isfinite(glob)
        validation_valid_pixels+=int(v.sum());common_valid_pixels+=int(common.sum())
        for name,pred in preds.items():accumulate(name,'common_training_support',pred,x,common)
        previous=lookup.get((ts[i].value-3600*10**9,positions[i]))
        if previous in prior_data:
            previous_y,pv=prior_data[previous];cv=common&pv
            persistence_valid_pixels+=int(cv.sum())
            for name,pred in preds.items():accumulate(name,'persistence_common',pred,x,cv)
            accumulate('persistence_1h_radar_reference','persistence_common',previous_y,x,cv)
        if k%10000==0:print('evaluate',k,len(val),flush=True)
    rows=[]
    def ratio(n,d):return float(n/d) if d else None
    for (name,cohort),r in records.items():
        for j,th in enumerate([20,30,40,45]):
            tp,fp,fn=r['tp'][j],r['fp'][j],r['fn'][j]
            row=dict(baseline=name,cohort=cohort,threshold_dbz=th,n_pixels=r['n'],
                rmse_dbz=np.sqrt(r['sse']/r['n']) if r['n'] else None,
                mae_dbz=ratio(r['sae'],r['n']),bias_dbz=ratio(r['bias'],r['n']),
                csi=ratio(tp,tp+fp+fn),pod=ratio(tp,tp+fn),far=ratio(fp,tp+fp),tp=int(tp),fp=int(fp),fn=int(fn))
            for l,km in enumerate([2,4,8,16]):
                den=r['fss_den'][j,l];row[f'fss_{km}km']=1-r['fss_num'][j,l]/den if den else None
            rows.append(row)
    pd.DataFrame(rows).to_csv(a.output_dir/'validation_metrics.csv',index=False)
    (a.output_dir/'protocol.json').write_text(json.dumps(dict(fit='train',evaluation='validation',
        test_targets_read=False,persistence_previous_split='validation only',scope='valid pixels with training support, patch-weighted',
        validation_valid_pixels=validation_valid_pixels,common_valid_pixels=common_valid_pixels,
        persistence_valid_pixels=persistence_valid_pixels,index_sha256=index_fingerprint(ts,pos),
        missing_target='excluded',fss='fractions normalised by valid support, constant empty borders',
        zero_zero_fss='undefined',persistence='radar t-1h, not input-equivalent to ERA5-only',
        production_ready=False,warning='Exact-palette policy does not establish dry observations; metrics conditional on retained valid pixels.'),indent=2))
if __name__=='__main__':main()
