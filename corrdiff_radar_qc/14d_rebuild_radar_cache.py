#!/usr/bin/env python3
"""Scan all expected hourly sources and create a separate, traceable QC cache."""
from __future__ import annotations
import argparse,csv,json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from radar_qc import classify,fingerprint,load_policy,sha256


def read_reviews(path):
    if path is None:return {}
    df=pd.read_csv(path,keep_default_na=False,dtype=str)
    required={'timestamp','source_sha256','action','reason','reviewer'}
    if not required.issubset(df.columns):raise ValueError(f'Review CSV needs {sorted(required)}')
    result={}
    for r in df.to_dict('records'):
        t=pd.Timestamp(r['timestamp'])
        if t.tzinfo is None:raise ValueError('Review timestamps must include timezone')
        key=t.tz_convert('UTC').isoformat()
        if key in result:raise ValueError(f'Duplicate review: {key}')
        if r['action'] not in ('accept','exclude') or not r['reason'].strip() or not r['reviewer'].strip():
            raise ValueError(f'Incomplete source review: {key}')
        if len(r['source_sha256'])!=64:raise ValueError('Review must bind source SHA256')
        result[key]=r
    return result


def panel(path,rgba,grid,old,title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(12,4))
    ax[0].imshow(rgba);ax[0].set_title('RGB(A) amostrado — origem')
    for a,data,label in [(ax[1],old,'Cache antigo'),(ax[2],grid,'QC: cinza = inválido')]:
        cmap=plt.get_cmap('turbo').copy();cmap.set_bad('#bcbcbc')
        im=a.imshow(np.ma.masked_invalid(data),origin='upper',vmin=0,vmax=50,cmap=cmap)
        a.set_title(label);fig.colorbar(im,ax=a,shrink=.7,label='dBZ')
    for a in ax:a.set_axis_off()
    fig.suptitle(title,fontsize=9);fig.tight_layout();fig.savefig(path,dpi=120);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--radar-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--old-cache-dir',type=Path)
    p.add_argument('--policy',type=Path)
    p.add_argument('--reviews',type=Path)
    p.add_argument('--begin',default='2011-01-01')
    p.add_argument('--end',default='2024-12-31 23:00:00')
    p.add_argument('--lat-range',type=float,nargs=2,default=[-23.5,-22.25])
    p.add_argument('--lon-range',type=float,nargs=2,default=[-44,-42.5])
    p.add_argument('--resolution-km',type=float,default=2)
    p.add_argument('--image-height',type=int,default=654)
    p.add_argument('--image-width',type=int,default=656)
    p.add_argument('--max-panels',type=int,default=40)
    a=p.parse_args();policy=load_policy(a.policy);reviews=read_reviews(a.reviews)
    if not a.radar_dir.is_dir():p.error('Radar source directory does not exist')
    if a.old_cache_dir and not a.old_cache_dir.is_dir():p.error('Old cache directory does not exist')
    if a.resolution_km<=0 or a.lat_range[0]>=a.lat_range[1] or a.lon_range[0]>=a.lon_range[1]:p.error('Invalid geometry')
    if min(a.image_width,a.image_height)<2:p.error('Invalid image dimensions')
    start=pd.to_datetime(a.begin,utc=True);end=pd.to_datetime(a.end,utc=True)
    if start!=start.floor('h') or end!=end.floor('h'):p.error('Use exact hourly boundaries')
    times=pd.date_range(start,end,freq='1h')
    if not len(times):p.error('Empty date range')
    a.output_dir.mkdir(parents=True,exist_ok=False)
    (a.output_dir/'panels').mkdir()
    config=dict(policy=policy,lat_range=a.lat_range,lon_range=a.lon_range,
        resolution_km=a.resolution_km,image_height=a.image_height,image_width=a.image_width,
        begin=str(times[0]),end=str(times[-1]),radar_dir=str(a.radar_dir.resolve()),
        review_sha256=sha256(a.reviews) if a.reviews else None,
        implementation_sha256=sha256(Path(__file__).with_name('radar_qc.py')),runner_sha256=sha256(__file__))
    config['config_sha256']=fingerprint(config)
    (a.output_dir/'qc_config.json').write_text(json.dumps(config,indent=2))
    lat=np.arange(*a.lat_range,a.resolution_km/111.);lon=np.arange(*a.lon_range,a.resolution_km/111.)
    xx,yy=np.meshgrid(lon,lat)
    px=((xx-a.lon_range[0])/(a.lon_range[1]-a.lon_range[0])*(a.image_width-1)).astype(int)
    py=((a.lat_range[1]-yy)/(a.lat_range[1]-a.lat_range[0])*(a.image_height-1)).astype(int)
    fields=['timestamp','year','source','source_sha256','status','map_fraction','decoded_fraction_before_quarantine',
        'dry_pixels_before_quarantine','unknown_pixels_before_quarantine','extreme_pixels_before_quarantine',
        'valid_fraction','extreme_pixels','dry_pixels','old_extreme_pixels','old_valid_fraction','old_cache_status',
        'cache_sha256','review_action','review_reason','error']
    totals=Counter();yearly={};panel_groups=Counter();panels=0;used_reviews=set()
    with (a.output_dir/'qc_manifest.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for i,t in enumerate(times):
            source=a.radar_dir/t.strftime('%Y/%m/%d/%Y_%m_%d_%H_%M.png');name=t.strftime('%Y%m%d_%H_%M.npy')
            row=dict(timestamp=t.isoformat(),year=t.year,source=str(source),status='missing_png')
            old=np.full(px.shape,np.nan,np.float32)
            if a.old_cache_dir:
                oldfile=a.old_cache_dir/name;row['old_cache_status']='missing'
                if oldfile.exists():
                    try:
                        old=np.load(oldfile,allow_pickle=False)
                        if old.shape!=px.shape:raise ValueError('Old cache shape mismatch')
                        row.update(old_extreme_pixels=int(np.sum(old>=45)),old_valid_fraction=float(np.isfinite(old).mean()),old_cache_status='available')
                    except Exception as e:
                        old=np.full(px.shape,np.nan,np.float32);row['old_cache_status']=f'error:{type(e).__name__}'
            try:
                if source.exists():
                    row['source_sha256']=sha256(source);review=reviews.get(t.isoformat())
                    if review:
                        used_reviews.add(t.isoformat())
                        if review['source_sha256']!=row['source_sha256']:raise ValueError('Review source hash mismatch')
                        row.update(review_action=review['action'],review_reason=review['reason'])
                    with Image.open(source) as im:
                        if im.size!=(a.image_width,a.image_height):raise ValueError(f'Unexpected dimensions {im.size}')
                        rgba=np.asarray(im.convert('RGBA'))[py,px]
                    grid,info=classify(rgba,policy,row.get('review_action',''));row.update(info)
                    if info['status'].startswith('accepted_'):
                        dest=a.output_dir/name
                        with dest.with_suffix('.tmp').open('wb') as out:np.save(out,grid)
                        dest.with_suffix('.tmp').replace(dest);row['cache_sha256']=sha256(dest)
                    group=(t.year,info['status'])
                    if panels<a.max_panels and panel_groups[group]<1:
                        panel(a.output_dir/'panels'/f'{t:%Y%m%d_%H_%M}.png',rgba,grid,old,f'{t.isoformat()} | {info["status"]}')
                        panels+=1;panel_groups[group]+=1
            except Exception as e:
                row.update(status='error',error=f'{type(e).__name__}: {e}')
            totals[row['status']]+=1;yearly.setdefault(t.year,Counter())[row['status']]+=1
            writer.writerow(row)
            if i%1000==0:f.flush();print(f'{i+1}/{len(times)} {dict(totals)}',flush=True)
    df=pd.read_csv(a.output_dir/'qc_manifest.csv')
    df[df.status.str.startswith('quarantine')][['timestamp','source_sha256','status']].assign(action='',reason='',reviewer='').to_csv(a.output_dir/'source_review_queue.csv',index=False)
    coverage=df.groupby(['year','status']).size().rename('hours').reset_index()
    coverage['expected_hours']=coverage.year.map(df.groupby('year').size())
    coverage['fraction_of_expected']=coverage.hours/coverage.expected_hours
    coverage.to_csv(a.output_dir/'coverage_by_year.csv',index=False)
    summary=dict(config_sha256=config['config_sha256'],manifest_sha256=sha256(a.output_dir/'qc_manifest.csv'),
        counts=dict(totals),by_year=yearly,grid_shape=list(px.shape),complete=True,panels=panels,
        unused_review_timestamps=sorted(set(reviews)-used_reviews),production_ready=False,
        dry_rule_documented=bool(policy['dry_rgba'] and policy['dry_evidence']),
        note='Source semantics and coverage must be reviewed. Quarantine is reversible; it is not proof of an invalid image.')
    (a.output_dir/'qc_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    if totals['error']:raise SystemExit(2)
if __name__=='__main__':main()
