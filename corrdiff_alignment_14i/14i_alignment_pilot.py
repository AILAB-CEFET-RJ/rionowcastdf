#!/usr/bin/env python3
"""Inspect short radar sequences with corrected local->UTC time and AEQD geometry."""
import argparse
import csv
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from alignment import (CONTRACT, LOCAL, VERSION, analyze_array, digest, geometry,
                       local_to_utc, parse_utc, png_path, utc_candidates)

DEFAULT_CASES=['2011_08_27_11_00','2016_10_02_14_00','2014_10_02_04_00',
               '2012_02_25_03_00','2012_02_25_04_00','2012_12_11_17_00']

def csv_write(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def inspect(root,u,out,g,layout):
    local=u.astimezone(LOCAL);name=u.strftime('%Y%m%dT%H%MZ')
    path=png_path(root,u)
    if layout=='flat':path=root/path.name
    row=dict(timestamp_utc=u.isoformat(),timestamp_local=local.isoformat(),
        png_name=path.name,source=str(path),utc_offset_hours=local.utcoffset().total_seconds()/3600,
        local_filename_ambiguous=len(utc_candidates(local.replace(tzinfo=None)))!=1,
        status='missing_png',cache_file='',error='')
    if not path.exists():return row,None,None
    try:
        raw=path.read_bytes();row['source_sha256']=hashlib.sha256(raw).hexdigest()
        with Image.open(path) as im:
            row['source_mode']=im.mode
            if im.size!=(656,654):raise ValueError(f'Unexpected dimensions {im.size}, expected (656,654)')
            a=np.array(im.convert('RGBA'))
        native,corrected,legacy,review,counts=analyze_array(a,g);row.update(counts)
        if row['local_filename_ambiguous']:
            row['status_before_time_quarantine']=row['status']
            row['status']='ambiguous_local_filename_review';review[:]=np.nan
        cache=out/'cache_utc'/f'{name}.npz'
        np.savez_compressed(cache,
            dbz_palette_corrected_raw=corrected,dbz_review=review,
            palette_mask=np.isfinite(corrected),geometric_mask=g['geometric_mask'],
            dbz_palette_legacy_geometry=legacy,
            sampled_alpha=a[g['py'].clip(0,653),g['px'].clip(0,655),3],
            timestamp_utc=np.array(u.isoformat()),timestamp_local=np.array(local.isoformat()),
            config_sha256=np.array(digest(CONTRACT)),source_sha256=np.array(row['source_sha256']))
        row['cache_file']=str(cache.relative_to(out))
        row['cache_sha256']=hashlib.sha256(cache.read_bytes()).hexdigest()
        return row,a,native
    except Exception as e:
        row.update(status='error',error=f'{type(e).__name__}: {e}')
        return row,None,None

def display_rgba(a):
    y,x=np.indices(a.shape[:2]);bg=np.where(((x//12+y//12)%2)[...,None],.85,.98)
    alpha=a[:,:,3:4]/255.
    return a[:,:,:3]/255.*alpha+bg*(1-alpha)

def sequence_panel(path,items,g):
    cols=min(4,len(items));rows=(len(items)+cols-1)//cols
    fig,axes=plt.subplots(rows,cols,figsize=(max(9,4.4*cols),3.6*rows),squeeze=False)
    yy,xx=np.nonzero(g['native_roi']);xmin,xmax=xx.min(),xx.max();ymin,ymax=yy.min(),yy.max()
    for ax,(r,a,_) in zip(axes.flat,items):
        if a is not None:
            ax.imshow(display_rgba(a))
            ax.contour(g['native_roi'].astype(int),levels=[.5],colors=['cyan'],linewidths=.6)
            ax.set_xlim(xmin-5,xmax+5);ax.set_ylim(ymax+5,ymin-5)
        else:ax.text(.5,.5,r['status'],ha='center',va='center',transform=ax.transAxes)
        ax.set_title(r['timestamp_local'][11:16]+' Brasília | '+r['timestamp_utc'][11:16]+' UTC\n'+
            r['status'].replace('_',' '),fontsize=7)
        ax.set_xlabel('coluna do PNG');ax.set_ylabel('linha do PNG')
    for ax in list(axes.flat)[len(items):]:ax.axis('off')
    fig.suptitle('Domínio corrigido: xadrez = transparência; ciano = ROI geométrica\nSem classificação automática de tempo seco ou validade',fontsize=10)
    fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)

def geometry_panel(path,g):
    fig,ax=plt.subplots(figsize=(8,8))
    ax.scatter(g['legacy_px'],g['legacy_py'],s=1,c='gray',label='Amostragem legada')
    m=g['geometric_mask'];ax.scatter(g['px'][m],g['py'][m],s=2,c='red',label='AEQD de referência')
    ax.contour(g['native_roi'].astype(int),levels=[.5],colors=['blue'],linewidths=1)
    ax.scatter([327.5],[326.5],c='black',marker='+',label='Centro do radar')
    ax.set_xlim(-.5,655.5);ax.set_ylim(653.5,-.5);ax.set_aspect('equal');ax.legend()
    ax.set_xlabel('coluna do PNG');ax.set_ylabel('linha do PNG')
    ax.set_title('Mesmo domínio lat/lon, diferentes pixels de origem\nContorno azul: domínio projetado; não é máscara de observação')
    fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)

def anchor_panel(path,a,g,case):
    _,corrected,legacy,_,_=analyze_array(a,g)
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    axes[0].imshow(display_rgba(a));axes[0].contour(g['native_roi'].astype(int),levels=[.5],colors=['cyan'])
    axes[0].set_title('PNG original e domínio projetado')
    axes[0].set_xlabel('coluna PNG');axes[0].set_ylabel('linha PNG')
    for ax,z,title in zip(axes[1:],[legacy,corrected],['Grade: geometria legada','Grade: AEQD de referência']):
        plot=ax.pcolormesh(g['lon'],g['lat'],np.ma.masked_invalid(z),shading='nearest',vmin=20,vmax=50,cmap='turbo')
        ax.set_title(title);ax.set_xlabel('longitude');ax.set_ylabel('latitude')
        fig.colorbar(plot,ax=ax,label='dBZ da paleta adotada')
    fig.suptitle(case+' — sem média temporal; branco nas grades = desconhecido, não tempo seco')
    fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--radar-dir',required=True,type=Path)
    p.add_argument('--output-dir',required=True,type=Path)
    group=p.add_mutually_exclusive_group()
    group.add_argument('--local-cases',nargs='+',help='Nomes locais YYYY_MM_DD_HH_MM; padrão: seis casos de treino')
    group.add_argument('--utc-cases',nargs='+',help='Instantes com Z ou +00:00, para associação ERA5 explícita')
    p.add_argument('--radius-minutes',type=int,default=10)
    p.add_argument('--step-minutes',type=int,default=2)
    p.add_argument('--source-layout',choices=['tree','flat'],default='tree')
    a=p.parse_args()
    if a.radius_minutes<0 or a.step_minutes<=0 or a.radius_minutes%a.step_minutes:
        p.error('radius >= 0 e múltiplo de step > 0')
    if 2*a.radius_minutes//a.step_minutes+1>61:p.error('Piloto limitado a 61 frames por caso')
    try:
        anchors=[parse_utc(t) for t in a.utc_cases] if a.utc_cases else [local_to_utc(t) for t in (a.local_cases or DEFAULT_CASES)]
    except ValueError as e:p.error(str(e))
    if len(anchors)!=len(set(anchors)):p.error('Casos duplicados')
    if len(anchors)>20:p.error('Piloto limitado a 20 casos')
    if not a.radar_dir.is_dir():p.error('radar-dir não existe')
    if a.output_dir.exists():p.error('output-dir já existe; escolha uma versão nova')
    a.output_dir.mkdir(parents=True)
    (a.output_dir/'cache_utc').mkdir();(a.output_dir/'panels').mkdir()
    g=geometry();np.savez_compressed(a.output_dir/'geometry.npz',**g)
    config=dict(CONTRACT,config_sha256=digest(CONTRACT))
    (a.output_dir/'config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False))
    geometry_panel(a.output_dir/'panels'/'geometry_comparison.png',g)
    records={};case_rows=[];transitions=[]
    for anchor in anchors:
        case=anchor.strftime('%Y%m%dT%H%MZ');items=[]
        offsets=range(-a.radius_minutes,a.radius_minutes+1,a.step_minutes)
        for minutes in offsets:
            u=anchor+timedelta(minutes=minutes)
            if u not in records:records[u]=inspect(a.radar_dir,u,a.output_dir,g,a.source_layout)
            items.append(records[u])
        sequence_panel(a.output_dir/'panels'/f'{case}_sequence.png',items,g)
        center=records[anchor][0]
        if records[anchor][1] is not None:
            anchor_panel(a.output_dir/'panels'/f'{case}_comparison.png',records[anchor][1],g,case)
        for prev,now in zip(items,items[1:]):
            pr,_,pn=prev;nr,_,nn=now
            tr=dict(case=case,previous_utc=pr['timestamp_utc'],current_utc=nr['timestamp_utc'],
                previous_status=pr['status'],current_status=nr['status'],same_file_hash=None,
                palette_roi_jaccard=None,palette_roi_union_pixels=None)
            if 'source_sha256' in pr and 'source_sha256' in nr:
                tr['same_file_hash']=pr['source_sha256']==nr['source_sha256']
            if pn is not None and nn is not None:
                x=np.isfinite(pn)&g['native_roi'];y=np.isfinite(nn)&g['native_roi'];union=int((x|y).sum())
                tr['palette_roi_union_pixels']=union
                if union:tr['palette_roi_jaccard']=float((x&y).sum()/union)
            transitions.append(tr)
        local=anchor.astimezone(LOCAL)
        # This path is what a naive UTC-as-local filename lookup would request.
        oldpath=a.radar_dir/anchor.strftime('%Y/%m/%d/%Y_%m_%d_%H_%M.png')
        case_rows.append(dict(case=case,era5_reference_utc=anchor.isoformat(),
            corrected_local_filename=local.strftime('%Y_%m_%d_%H_%M.png'),
            corrected_source=center['source'],
            legacy_same_utc_label_source=str(a.radar_dir/oldpath.name if a.source_layout=='flat' else oldpath),center_status=center['status'],
            expected_frames=len(items),readable_frames=sum(x[1] is not None for x in items),
            transparent_frames=sum(x[0]['status']=='transparent_ambiguous' for x in items),
            frames_with_palette_in_roi=sum(x[0].get('native_roi_palette_pixels',0)>0 for x in items),
            frames_with_palette_on_corrected_grid=sum(x[0].get('corrected_grid_palette_points',0)>0 for x in items),
            missing_frames=sum(x[0]['status']=='missing_png' for x in items),
            error_frames=sum(x[0]['status']=='error' for x in items)))
        print(json.dumps(case_rows[-1],ensure_ascii=False),flush=True)
    rows=[records[u][0] for u in sorted(records)]
    csv_write(a.output_dir/'frames.csv',rows);csv_write(a.output_dir/'cases.csv',case_rows)
    csv_write(a.output_dir/'transitions.csv',transitions)
    counts={s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})}
    summary=dict(version=VERSION,config_sha256=digest(CONTRACT),cases=len(anchors),
        unique_requested_frames=len(rows),counts=counts,complete=True,
        source_read_errors=counts.get('error',0),all_requested_files_present=not counts.get('missing_png',0),
        grid_shape=list(g['lat'].shape),geometric_grid_points=int(g['geometric_mask'].sum()),
        production_ready=False,dry_rule_documented=False,temporal_aggregation='none',
        era5_values_read=False,old_cache_used=False,old_dataset_modified=False,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        module_sha256=hashlib.sha256(Path(__file__).with_name('alignment.py').read_bytes()).hexdigest(),
        note='Complete means all requested frames were attempted. Palette_mask is not observational validity. Missing/transparent/map frames are never filled as dry. UTC references do not prove ERA5 availability.')
    (a.output_dir/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)
    if counts.get('error',0):return 1
    return 0

if __name__=='__main__':sys.exit(main())
