"""Content-addressed, resumable radar targets under the frozen producer policy."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
import sqlite3
from pathlib import Path
import numpy as np
from PIL import Image
from .contract_common import *
from .reference_geometry import analyze_array


def target(classes, alpha, g, policy, native_count):
    geom=g['geometric_mask'];support=g['support']
    if classes.shape!=geom.shape or alpha.shape!=geom.shape:raise ValueError('Invalid cached grid')
    if not np.isin(classes,[0,1,2,3,4,5,6,7,253,254,255]).all():raise ValueError('Unknown class')
    if ((classes==0)&geom&(alpha!=0)).any() or ((classes>=1)&(classes<=7)&geom&(alpha!=255)).any():
        raise ValueError('Class/alpha disagreement')
    echo=(classes>=1)&(classes<=7)&(alpha==255)&support
    dry=(classes==0)&(alpha==0)&support
    if policy=='reviewed_map_no_echo':
        if native_count:raise ValueError('Reviewed no-echo map contains palette pixels')
        dry=support.copy();echo=np.zeros_like(support)
    elif policy=='valid_transparent_no_echo':
        if native_count or (classes[geom]!=0).any():raise ValueError('Transparent policy disagreement')
    elif policy=='valid_no_echo_on_sampled_grid':
        if native_count==0 or echo.any():raise ValueError('Unsampled palette policy disagreement')
    elif policy=='valid_echo_and_no_echo':
        if native_count==0:raise ValueError('Echo policy without palette')
    else:raise ValueError('Unsupported policy')
    mask=echo|dry;code=np.full(mask.shape,np.nan,np.float32)
    code[dry]=0;code[echo]=np.asarray(REFERENCE['palette_dbz'],np.float32)[classes[echo]-1]
    validate_target(code,mask,support)
    return code,mask


def decode_job(job,c,g,out):
    digest,rel,policy,reuse=job
    origin='png'
    if reuse:
        p=inside(c['reuse_14j_dir'],reuse[0])
        if sha(p)!=reuse[1]:raise ValueError('14j decoded checksum mismatch '+str(p))
        with np.load(p,allow_pickle=False) as z:
            classes=z['grid_class'].copy();alpha=z['grid_alpha'].copy()
            meta=json.loads(str(z['metadata_json']))
            # Native count is computed from cached source classes, avoiding metadata-layout assumptions.
            nc=int(np.isin(z['native_class'],np.arange(1,8)).sum())
        if meta.get('status')=='map_screen_review' and policy!='reviewed_map_no_echo':raise ValueError('Unreviewed cached map selected')
        origin='14j'
    else:
        p=inside(c['radar_dir'],rel);b=p.read_bytes()
        import hashlib
        if hashlib.sha256(b).hexdigest()!=digest:raise ValueError('PNG changed since freeze: '+str(p))
        with Image.open(io.BytesIO(b)) as im:
            if im.format!='PNG' or im.size!=(656,654) or getattr(im,'n_frames',1)!=1:raise ValueError('Invalid PNG dimensions/type')
            im.verify()
        with Image.open(io.BytesIO(b)) as im:rgba=np.asarray(im.convert('RGBA'))
        native,_,_,_,meta=analyze_array(rgba,g);nc=int(np.isfinite(native).sum())
        if meta['status']=='map_screen_review' and policy!='reviewed_map_no_echo':raise ValueError('Unreviewed map selected')
        gm=g['geometric_mask'];classes=np.full(gm.shape,253,np.uint8);alpha=np.zeros(gm.shape,np.uint8)
        a=rgba[g['py'][gm],g['px'][gm],3];v=native[g['py'][gm],g['px'][gm]]
        local=np.full(a.shape,255,np.uint8);local[a==0]=0;local[(a>0)&(a<255)]=254
        for i,value in enumerate(REFERENCE['palette_dbz'],1):local[v==value]=i
        classes[gm]=local;alpha[gm]=a
    code,mask=target(classes,alpha,g,policy,nc)
    relout=f'objects/{digest[:2]}/{digest}.npz';dest=inside(out,relout);dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,target_code=code,mask=mask,source_sha256=np.array(digest))
    tmp.replace(dest)
    return (digest,relout,sha(dest),policy,origin)


def run(c,resume=False):
    df,g,semantic=inputs(c)
    # Radar cache is independent of ERA5 channel selection but bound to selected period and policy.
    semantic={k:v for k,v in semantic.items() if k!='channels'}
    with output_lock(c['radar_cache_dir'],resume,semantic) as out:
        previous=read(out/'summary.json') if (out/'summary.json').exists() else {}
        if previous.get('complete') and sha(out/'index.sqlite')!=previous['index_sha256']:raise ValueError('Cache index changed')
        db=sqlite3.connect(out/'index.sqlite')
        db.execute('CREATE TABLE IF NOT EXISTS objects (sha TEXT PRIMARY KEY,path TEXT,digest TEXT,policy TEXT,origin TEXT)')
        existing={r[0]:r for r in db.execute('SELECT * FROM objects')}
        for digest,rel,h,policy,origin in existing.values():
            if sha(inside(out,rel))!=h:raise ValueError('Compact cache corruption: '+rel)
        reuse={}
        if c.get('reuse_14j_dir'):
            old=Path(c['reuse_14j_dir'])
            if not read(old/'summary.json').get('complete'):raise ValueError('14j incomplete')
            if sha(old/'geometry.npz')!=semantic['geometry_sha256']:raise ValueError('14j geometry changed')
            with ro(old/'manifest.sqlite') as od:
                reuse={r[0]:(r[1],r[2]) for r in od.execute('SELECT sha256,decoded_path,decoded_sha256 FROM sources WHERE decoded_path IS NOT NULL')}
        unique=df.drop_duplicates('sha');jobs=[]
        for r in unique.itertuples(index=False):
            if r.sha in existing:
                if existing[r.sha][3]!=r.policy_status:raise ValueError('Changed cache policy')
            else:jobs.append((r.sha,r.path,r.policy_status,reuse.get(r.sha)))
        dump(out/'summary.json',dict(complete=False,selected_hours=len(df),unique_sources=len(unique)))
        with ThreadPoolExecutor(max_workers=c['workers']) as pool:
            for start in range(0,len(jobs),256):
                for record in pool.map(lambda j:decode_job(j,c,g,out),jobs[start:start+256]):
                    db.execute('INSERT OR REPLACE INTO objects VALUES (?,?,?,?,?)',record)
                db.commit()
                print(json.dumps({'cache_processed':len(existing)+min(start+256,len(jobs)),'unique_sources':len(unique)}),flush=True)
        count=db.execute('SELECT count(*) FROM objects').fetchone()[0]
        db.close()
        if count!=len(unique):raise ValueError('Cache object count differs')
        summary=dict(version=VERSION,complete=True,selected_hours=len(df),unique_sources=count,
                     index_sha256=sha(out/'index.sqlite'),production_ready=False,
                     note='Content-addressed targets; zero means no displayed echo, not measured 0 dBZ.')
        dump(out/'summary.json',summary);print(json.dumps(summary,indent=2))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--contract-config',required=True)
    p.add_argument('--resume',action='store_true');a=p.parse_args(argv);run(load_config(a.contract_config),a.resume)
