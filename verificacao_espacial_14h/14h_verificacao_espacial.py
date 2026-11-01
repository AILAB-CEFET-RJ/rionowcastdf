#!/usr/bin/env python3
"""Diagnóstico somente leitura: paleta RGBA versus amostragem pontual legada.
Não infere cobertura, validade da varredura ou tempo seco. Não altera cache.
Dependências: numpy pillow matplotlib scipy.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import label, distance_transform_edt

PALETTE = [(197,0,197,50),(227,6,5,45),(255,112,0,40),
           (195,230,0,35),(4,85,4,30),(19,122,19,25),(0,167,12,20)]
DEFAULTS = ['2011_08_27_11_00','2016_10_02_14_00','2014_10_02_04_00']

def grid_map():
    lat = np.arange(-23.5, -22.25, 2.0/111.0)
    lon = np.arange(-44.0, -42.5, 2.0/111.0)
    xx, yy = np.meshgrid(lon, lat)
    px = ((xx+44.0)/1.5*655).astype(int)
    py = ((-22.25-yy)/1.25*653).astype(int)
    return px, py

def masks(a):
    colors = np.full(a.shape[:2], np.nan)
    for r,g,b,z in PALETTE:
        colors[np.all(a[:,:,:3] == (r,g,b), axis=2) & (a[:,:,3] == 255)] = z
    return colors, np.isfinite(colors)

def write_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def audit(path, output, name):
    raw = path.read_bytes()
    with Image.open(path) as im:
        mode = im.mode
        if im.size != (656,654):
            raise ValueError(f'Dimensões {im.size}; esperado (656,654). Não redimensionado.')
        a = np.array(im.convert('RGBA'))
    z, echo = masks(a)
    px,py = grid_map()
    sampled = np.zeros(echo.shape, bool)
    sampled[py,px] = True
    distance, nearest = distance_transform_edt(~sampled, return_indices=True)
    ids,n = label(echo, structure=np.ones((3,3), int))
    y,x = np.nonzero(echo)
    pixels = []
    for yi,xi in zip(y,x):
        pixels.append(dict(x=int(xi),y=int(yi),dbz_palette=float(z[yi,xi]),
            component=int(ids[yi,xi]),sampled=bool(sampled[yi,xi]),
            nearest_sample_x=int(nearest[1,yi,xi]),nearest_sample_y=int(nearest[0,yi,xi]),
            distance_source_pixels=float(distance[yi,xi])))
    write_csv(output/f'{name}_pixels.csv', pixels,
        ['x','y','dbz_palette','component','sampled','nearest_sample_x','nearest_sample_y','distance_source_pixels'])
    components=[]
    for k in range(1,n+1):
        cy,cx=np.nonzero(ids==k)
        components.append(dict(component=k,pixels=len(cx),x_min=int(cx.min()),x_max=int(cx.max()),
            y_min=int(cy.min()),y_max=int(cy.max()),sampled_pixels=int(sampled[cy,cx].sum())))
    write_csv(output/f'{name}_components.csv',components,
        ['component','pixels','x_min','x_max','y_min','y_max','sampled_pixels'])
    # Checkerboard explicitly exposes transparency; no geographic coverage inferred.
    yy,xx=np.indices(echo.shape)
    bg=np.where(((xx//12+yy//12)%2)[...,None],0.85,0.98)
    alpha=a[:,:,3:4]/255.0
    shown=a[:,:,:3]/255.0*alpha+bg*(1-alpha)
    fig,axes=plt.subplots(1,3,figsize=(18,6))
    axes[0].imshow(shown); axes[0].set_title('PNG original sobre xadrez (transparência)')
    axes[1].imshow(shown)
    axes[1].scatter(px.ravel(),py.ravel(),s=2,c='dodgerblue',label='Pontos da grade')
    axes[1].scatter(x,y,s=14,facecolors='none',edgecolors='red',label='Cor exata opaca da paleta')
    axes[1].legend(loc='upper right',fontsize=7)
    axes[1].set_title('Localização no raster original')
    axes[2].imshow(shown)
    axes[2].scatter(px.ravel(),py.ravel(),s=18,c='dodgerblue')
    axes[2].scatter(x,y,s=45,facecolors='none',edgecolors='red')
    if len(x):
        largest=max(components,key=lambda c:c['pixels'])
        axes[2].set_xlim(max(-.5,largest['x_min']-15),min(655.5,largest['x_max']+15))
        axes[2].set_ylim(min(653.5,largest['y_max']+15),max(-.5,largest['y_min']-15))
    axes[2].set_title('Zoom do maior componente de cor')
    for ax in axes: ax.set_xlabel('x: coluna PNG'); ax.set_ylabel('y: linha PNG (origem superior)')
    fig.suptitle(name+' — cor da paleta não comprova eco meteorológico')
    fig.tight_layout(); fig.savefig(output/f'{name}_panel.png',dpi=160); plt.close(fig)
    hits=int((echo & sampled).sum())
    result=dict(timestamp_file=name,source=str(path),source_sha256=hashlib.sha256(raw).hexdigest(),
        source_mode=mode,status='checked',transparent_pixels=int((a[:,:,3]==0).sum()),
        palette_opaque_pixels=int(echo.sum()),palette_sampled_pixels=hits,
        palette_unsampled_pixels=int(echo.sum())-hits,grid_palette_points=int(echo[py,px].sum()),
        components=n,components_with_sample=sum(c['sampled_pixels']>0 for c in components),
        nearest_sample_distance_median_px=float(np.median(distance[echo])) if echo.any() else None,
        interpretation=('palette_present_but_not_sampled' if echo.any() and not hits else
                        'palette_sampled' if hits else 'no_opaque_exact_palette'))
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--radar-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True,help='Novo diretório; recusa sobrescrever')
    p.add_argument('--timestamps',nargs='+',default=DEFAULTS,help='YYYY_MM_DD_HH_MM, sem extensão')
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    px,py=grid_map()
    results=[]
    for name in args.timestamps:
        import datetime
        dt=datetime.datetime.strptime(name,'%Y_%m_%d_%H_%M')
        path=args.radar_dir/dt.strftime('%Y/%m/%d')/(name+'.png')
        if not path.exists(): path=args.radar_dir/(name+'.png')
        try: result=audit(path,args.output_dir,name)
        except Exception as e: result=dict(timestamp_file=name,source=str(path),status='error',error=f'{type(e).__name__}: {e}')
        results.append(result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
    fields=list(dict.fromkeys(k for r in results for k in r))
    write_csv(args.output_dir/'spatial_summary.csv',results,fields)
    metadata=dict(version='14h-spatial-v1',grid_shape=list(px.shape),
        source_size=[656,654],grid_resolution_km=2,lat_range=[-23.5,-22.25],lon_range=[-44,-42.5],
        pixel_map='legacy linear geographic mapping, integer truncation; no area aggregation',
        geographic_registration_verified=False,palette=PALETTE,alpha_required=255,
        production_ready=False,results=results,
        limits=['Não distingue cor de legenda de eco meteorológico.',
                'Não determina validade de transparência nem cobertura física.',
                'Reproduz geometria legada; não valida georreferenciamento do produtor.',
                'Nenhuma alteração em PNG, cache, dataset ou splits.'])
    (args.output_dir/'summary.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    if any(r['status']=='error' for r in results): raise SystemExit(1)

if __name__=='__main__': main()
