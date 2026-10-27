#!/usr/bin/env python3
"""Read-only source provenance audit. No automatic quality exclusions."""
from __future__ import annotations
import argparse, ast, hashlib, json, logging
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import zarr
from audit14b_helpers import reconstruct

def digest(a):
    a=np.asarray(a)
    return hashlib.sha256(str((a.shape,str(a.dtype))).encode()+a.tobytes()).hexdigest()

def load_converter(path):
    # Extract only the reviewed class: avoid importing builder CLI/dependencies.
    tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RadarCacheSourceDataset')
    code='from __future__ import annotations\n'+ast.unparse(node)
    ns=dict(np=np,pd=pd,Path=Path,logging=logging,LOGGER=logging.getLogger('audit'),
            _normalize_timestamp=lambda t: pd.Timestamp(t).tz_convert('UTC').tz_localize(None))
    exec(compile(code,str(path),'exec'),ns)
    return ns['RadarCacheSourceDataset']

def compare(a,b):
    if a.shape!=b.shape: return dict(shape_match=False)
    valid=np.isfinite(a)&np.isfinite(b)
    return dict(shape_match=True,finite_mask_mismatch=int(np.sum(np.isfinite(a)!=np.isfinite(b))),
                unequal_finite=int(np.sum(a[valid]!=b[valid])),
                max_abs_difference=float(np.max(np.abs(a[valid]-b[valid]))) if valid.any() else None)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['dataset-dir','audit14b-dir','radar-dir','radar-cache-dir','builder-file','output-dir']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--candidate-fraction',type=float,default=.10)
    p.add_argument('--controls-per-year',type=int,default=2)
    p.add_argument('--max-panels',type=int,default=40)
    p.add_argument('--lat-range',type=float,nargs=2,default=[-23.5,-22.25])
    p.add_argument('--lon-range',type=float,nargs=2,default=[-44,-42.5])
    p.add_argument('--resolution-km',type=float,default=2)
    p.add_argument('--image-height',type=int,default=654)
    p.add_argument('--image-width',type=int,default=656)
    a=p.parse_args()
    if not 0<a.candidate_fraction<=1: p.error('candidate-fraction must be in (0,1]')
    a.output_dir.mkdir(parents=True,exist_ok=False)
    out=a.output_dir
    df=pd.read_csv(a.audit14b_dir/'timestamp_metrics.csv',parse_dates=['timestamp'])
    df['timestamp']=pd.to_datetime(df.timestamp,utc=True)
    if df.timestamp.duplicated().any(): raise ValueError('Duplicate timestamps in audit14b')
    ep=pd.read_csv(a.audit14b_dir/'episodes_ranked.csv')
    root=zarr.open_group(str(a.dataset_dir/'train.zarr' if (a.dataset_dir/'train.zarr').exists() else a.dataset_dir),mode='r')
    candidates=df.unique_fraction_45.ge(a.candidate_fraction)
    peaks=set(pd.to_datetime(ep.head(10).peak_timestamp,utc=True))
    peaks.update(pd.to_datetime(ep[ep.peak_year.eq(2024)].head(10).peak_timestamp,utc=True))
    selected=set(df.loc[candidates,'timestamp'])|peaks
    for _,g in df[~candidates & df.unique_count_45.gt(0)].groupby(df.timestamp.dt.year):
        selected.update(g.sort_values('unique_count_45',ascending=False).head(a.controls_per_year).timestamp)
    # Include neighboring available frames, never synthesize missing hours.
    selected |= {t+pd.Timedelta(hours=h) for t in list(selected) for h in [-1,1]}
    work=df[df.timestamp.isin(selected)].copy()
    work['_peak_priority']=~work.timestamp.isin(peaks)
    work=work.sort_values(['_peak_priority','timestamp'])
    repeats=df[df.field_hash.duplicated(keep=False)&df.unique_count_45.gt(0)].copy()
    repeats.to_csv(out/'repeated_fields_all_years.csv',index=False)
    C=load_converter(a.builder_file)
    converter=C(a.radar_dir,out/'unused_cache_directory',a.resolution_km,a.lat_range,a.lon_range,
                df.timestamp.min(),df.timestamp.max(),image_height=a.image_height,image_width=a.image_width)
    converter.build_grid(); converter.precompute_pixel_map()
    records=[]; colors=[]; reviews=[]; panels=0
    for i,row in work.iterrows():
        t=row.timestamp; key=t.strftime('%Y%m%d_%H_%M'); rec=dict(timestamp=str(t),candidate=bool(candidates.loc[i]))
        png=a.radar_dir/t.strftime('%Y/%m/%d/%Y_%m_%d_%H_%M.png'); cache=a.radar_cache_dir/(key+'.npy')
        rec.update(source_path=str(png),cache_path=str(cache))
        try:
            lo,hi=int(row.first_row),int(row.end_row)
            times=pd.to_datetime(root['timestamps'][lo:hi],unit='s',utc=True)
            if not len(times) or not (times==t).all(): raise ValueError('Stale audit14b row indices')
            y=np.asarray(root['target'][lo:hi,0],np.float32); m=np.asarray(root['mask'][lo:hi,0])
            if not np.isin(m,[0,1]).all(): raise ValueError('Nonbinary masks')
            rr=np.asarray(root['patch_row'][lo:hi],int); cc=np.asarray(root['patch_col'][lo:hi],int)
            from audit14b_helpers import summarize_frame
            if summarize_frame(t,y,m.astype(bool),rr,cc)['field_hash']!=row.field_hash:
                raise ValueError('Dataset differs from audit14b hash')
            stored,valid,seen=reconstruct(y,m.astype(bool),rr,cc)
            extreme=valid&(stored>=np.float32(np.log1p(45)))
            rec.update(extreme_mask_hash=digest(extreme),extreme_pixels=int(extreme.sum()))
            cached=np.load(cache,allow_pickle=False) if cache.exists() else None
            rec['cache_status']='available' if cached is not None else 'missing'
            if cached is not None:
                cp=np.stack([cached[r:r+y.shape[1],c:c+y.shape[2]] for r,c in zip(rr,cc)])
                rec['cache_target']=compare(y,np.log1p(np.maximum(np.nan_to_num(cp,nan=0,posinf=0,neginf=0),0)).astype(np.float32))
                rec['cache_mask_mismatch']=int(np.sum(m.astype(bool)!=np.isfinite(cp)))
            if not png.exists():
                rec['source_status']='missing'; records.append(rec); continue
            with Image.open(png) as im:
                rgb=np.asarray(im.convert('RGB')); rgba=np.asarray(im.convert('RGBA'))
                rec.update(source_mode=im.mode,source_size=list(im.size))
            rec.update(source_status='available',rgb_hash=digest(rgb),alpha_hash=digest(rgba[:,:,3]),
                       dimensions_match=rgb.shape[:2]==(a.image_height,a.image_width))
            px=np.clip(converter.px,0,rgb.shape[1]-1); py=np.clip(converter.py,0,rgb.shape[0]-1)
            sampled=rgb[py,px]; alpha=rgba[py,px,3]
            # Pointwise conversion equals full-image conversion then sampling.
            rebuilt=converter.rgb_to_reflectivity(sampled).astype(np.float32)
            if cached is not None: rec['png_cache']=compare(rebuilt,cached)
            h,w=stored.shape
            if rebuilt.shape[0]<h or rebuilt.shape[1]<w: raise ValueError('Source grid smaller than patch footprint')
            rgbfoot=sampled[:h,:w]; af=alpha[:h,:w]
            distance=np.linalg.norm(rgbfoot[:,:,None,:].astype(np.float32)-converter.legend_colors,axis=-1).min(-1)
            rec.update(extreme_transparent_pixels=int(np.sum(extreme&(af<255))),
                       extreme_rgb_distance_mean=float(distance[extreme].mean()) if extreme.any() else None,
                       sampled_rgb_hash=digest(rgbfoot[seen]))
            vals,cnt=np.unique(np.column_stack([rgbfoot[extreme],af[extreme]]),axis=0,return_counts=True)
            for v,n in zip(vals,cnt):
                colors.append(dict(timestamp=str(t),r=int(v[0]),g=int(v[1]),b=int(v[2]),alpha=int(v[3]),count=int(n),
                                   nearest_legend_distance=float(np.linalg.norm(converter.legend_colors-v[:3],axis=1).min())))
            if panels<a.max_panels and (t in peaks or rec['candidate']):
                fig,ax=plt.subplots(2,3,figsize=(14,8))
                ax[0,0].imshow(rgba); ax[0,0].set_title('PNG original (transparencia preservada)')
                ax[0,1].imshow(sampled,origin='lower');ax[0,1].set_title('RGB amostrado pela geometria atual')
                for axis,arr,title in [(ax[0,2],rebuilt,'PNG convertido (dBZ)'),(ax[1,0],np.where(valid,np.expm1(stored),np.nan),'Dataset reconstruido (dBZ)')]:
                    im=axis.imshow(arr,origin='lower',vmin=0,vmax=55,cmap='turbo');axis.set_title(title);fig.colorbar(im,ax=axis)
                ax[1,1].imshow(extreme,origin='lower',vmin=0,vmax=1);ax[1,1].set_title('Mascara >=45 dBZ no dataset')
                im=ax[1,2].imshow(np.where(seen,distance,np.nan),origin='lower');ax[1,2].set_title('Distancia RGB a legenda (nao e QC)');fig.colorbar(im,ax=ax[1,2])
                fig.suptitle(str(t));fig.tight_layout();fig.savefig(out/(key+'.png'),dpi=130);plt.close(fig);panels+=1
            rec['status']='checked'
        except Exception as e: rec.update(status='error',detail=str(e))
        records.append(rec)
        print(f'{len(records)}/{len(work)} {key} {rec.get("status",rec.get("source_status"))}',flush=True)
    flat=[{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in records]
    pd.DataFrame(flat).to_csv(out/'source_checks.csv',index=False)
    pd.DataFrame(colors,columns=['timestamp','r','g','b','alpha','count','nearest_legend_distance']).to_csv(out/'extreme_source_colors.csv',index=False)
    for r in records: reviews.append(dict(timestamp=r['timestamp'],decision='pending',reason='',evidence='',reviewer=''))
    pd.DataFrame(reviews).to_csv(out/'manual_review.csv',index=False)
    checks=pd.DataFrame(flat)
    for col in ['extreme_mask_hash','rgb_hash','sampled_rgb_hash']:
        if col in checks:
            g=checks[checks[col].notna() & checks[col].duplicated(keep=False)]
            g.to_csv(out/(col+'_repeats.csv'),index=False)
    summary=dict(version='14c-source-audit-v1',arguments={k:str(v) for k,v in vars(a).items()},
        builder_sha256=hashlib.sha256(a.builder_file.read_bytes()).hexdigest(),
        selected_frames=len(work),candidate_frames=int(candidates.sum()),errors=sum(r.get('status')=='error' for r in records),
        missing_png=sum(r.get('source_status')=='missing' for r in records),panels=panels,
        conclusion='Pending source review; no timestamps excluded; repeated masks and RGB distances are diagnostics, not proof of artifacts.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    if summary['errors']: raise SystemExit('Audit completed with errors: inspect source_checks.csv')
if __name__=='__main__': main()
