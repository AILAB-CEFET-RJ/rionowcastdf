#!/usr/bin/env python3
"""Read-only PNG census. Diagnostics do not establish observational validity."""
import argparse
import csv
import fcntl
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import sys
import time
import uuid
import warnings
import zipfile
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image, ImageDraw, __version__ as PIL_VERSION

VERSION = '14m-full-png-audit-v1'
PALETTE = [(197,0,197),(227,6,5),(255,112,0),(195,230,0),
           (4,85,4),(19,122,19),(0,167,12)]
DBZ = [50,45,40,35,30,25,20]
PATTERN = re.compile(r'^(\d{4}_\d{2}_\d{2}_\d{2}_\d{2})\.png$', re.I)
TZ = ZoneInfo('America/Sao_Paulo')
TRANSIENT = {'read_error','source_changed_during_read'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(tmp, path)


def timestamp_info(path):
    m = PATTERN.match(path.name)
    if not m:
        return None, 'unparsed', 'unknown', None
    try:
        t = datetime.strptime(m[1], '%Y_%m_%d_%H_%M')
    except ValueError:
        return None, 'invalid', 'unknown', None
    candidates = set()
    for fold in (0, 1):
        u = t.replace(tzinfo=TZ, fold=fold).astimezone(timezone.utc)
        if u.astimezone(TZ).replace(tzinfo=None) == t:
            candidates.add(u.isoformat())
    state = 'unique' if len(candidates) == 1 else ('ambiguous' if candidates else 'nonexistent')
    return t.isoformat(), state, str(t.year), next(iter(candidates)) if len(candidates) == 1 else None


def reference_grid():
    # Same explicit reference as 14j; not an inferred coverage/validity mask.
    lat, lon = np.meshgrid(np.arange(-23.5,-22.25,2/111),
                          np.arange(-44,-42.5,2/111), indexing='ij')
    a, b = np.deg2rad(lat), np.deg2rad(lon)
    a0, b0 = np.deg2rad([-22.955139,-43.248278])
    d = b - b0
    c = np.arccos(np.clip(np.sin(a0)*np.sin(a)+np.cos(a0)*np.cos(a)*np.cos(d),-1,1))
    k = np.ones_like(c)
    nz = c > 1e-12
    k[nz] = c[nz]/np.sin(c[nz])
    east = 6371008.8*k*np.cos(a)*np.sin(d)
    north = 6371008.8*k*(np.cos(a0)*np.sin(a)-np.sin(a0)*np.cos(a)*np.cos(d))
    rows = np.rint(326.5-north/(138900/326.5)).astype(int)
    cols = np.rint(327.5+east/(138900/327.5)).astype(int)
    inside = (rows>=0)&(rows<654)&(cols>=0)&(cols<656)
    return rows[inside], cols[inside]

GRID_ROWS, GRID_COLS = reference_grid()


def decode(data):
    result = {'status':'decode_error'}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            # Verify PNG structure/CRC, then force actual decoding.
            with Image.open(io.BytesIO(data)) as check:
                result.update(format=check.format, mode=check.mode,
                              width=check.width, height=check.height,
                              n_frames=getattr(check, 'n_frames', 1))
                check.verify()
            with Image.open(io.BytesIO(data)) as im:
                im.load()
                rgba = np.array(im.convert('RGBA'))
        rgb, alpha = rgba[:,:,:3], rgba[:,:,3]
        n = int(alpha.size)
        transparent = int(np.count_nonzero(alpha==0))
        partial = int(np.count_nonzero((alpha>0)&(alpha<255)))
        opaque = int(np.count_nonzero(alpha==255))
        palette = np.zeros(alpha.shape, bool)
        counts = []
        for color in PALETTE:
            hit = (alpha==255)&np.all(rgb==color,axis=2)
            palette |= hit
            counts.append(int(hit.sum()))
        p = int(palette.sum())
        # Historical beige diagnostic from 14i/14j; NOT a general map detector.
        beige = (alpha==255)&np.all(rgb==(242,239,233),axis=2)
        map_fraction = float(beige.mean())
        rows, cols = np.nonzero(palette)
        normal = result['width']==656 and result['height']==654
        grid_n = int(palette[GRID_ROWS,GRID_COLS].sum()) if normal else None
        result.update(
            pixels=n, transparent_pixels=transparent, partial_alpha_pixels=partial,
            opaque_pixels=opaque, opaque_nonpalette_pixels=opaque-p,
            palette_pixels=p, palette_counts=dict(zip(map(str,DBZ),counts)),
            transparent_fraction=transparent/n, map_fraction=map_fraction,
            map_beige_heuristic=map_fraction>=0.01,
            alpha_hash=sha(alpha.tobytes()), rgba_hash=sha(rgba.tobytes()),
            palette_mask_hash=sha(palette.tobytes()),
            corrected_grid_palette_points=grid_n,
            palette_bbox_xyxy=[int(cols.min()),int(rows.min()),int(cols.max()),int(rows.max())] if p else None,
            palette_centroid_xy=[float(cols.mean()),float(rows.mean())] if p else None,
            palette_quadrants=[int(palette[:alpha.shape[0]//2,:alpha.shape[1]//2].sum()),
                               int(palette[:alpha.shape[0]//2,alpha.shape[1]//2:].sum()),
                               int(palette[alpha.shape[0]//2:,:alpha.shape[1]//2].sum()),
                               int(palette[alpha.shape[0]//2:,alpha.shape[1]//2:].sum())],
            sparse_palette_diagnostic=0<p<=20,
        )
        if result['format'] != 'PNG':
            state = 'not_png_format_review'
        elif result['n_frames'] != 1:
            state = 'multiframe_review'
        elif not normal:
            state = 'unexpected_dimensions_review'
        elif map_fraction >= .01:
            state = 'map_screen_review'
        elif transparent == n:
            state = 'transparent_ambiguous'
        elif p == 0:
            state = 'no_exact_palette_review'
        elif grid_n == 0:
            state = 'palette_in_png_not_sampled_in_corrected_grid'
        else:
            state = 'palette_sampled_observation_validity_unconfirmed'
        result['status'] = state
    except Exception as e:
        result['error'] = f'{type(e).__name__}: {e}'
    return result


def init_db(db):
    db.executescript('''
    PRAGMA journal_mode=WAL;
    PRAGMA synchronous=FULL;
    CREATE TABLE IF NOT EXISTS objects (
      sha TEXT PRIMARY KEY, status TEXT NOT NULL, metrics TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS files (
      path TEXT PRIMARY KEY, size INTEGER, mtime_ns INTEGER, ctime_ns INTEGER,
      sha TEXT, status TEXT NOT NULL, error TEXT, local_time TEXT,
      time_state TEXT, year TEXT, utc_time TEXT, seen_run TEXT);
    CREATE TABLE IF NOT EXISTS changes (
      run TEXT,path TEXT,old_sha TEXT,new_sha TEXT,changed_at TEXT);
    CREATE INDEX IF NOT EXISTS files_sha ON files(sha);
    CREATE INDEX IF NOT EXISTS files_time ON files(local_time);
    CREATE INDEX IF NOT EXISTS files_run ON files(seen_run);
    ''')


def stat_key(s):
    return s.st_size, s.st_mtime_ns, s.st_ctime_ns


def scan(root, out, db, args, run):
    counters = {'encountered_pngs':0,'reused_by_stat':0,'decoded_unique_this_run':0,
                'reused_by_hash':0,'source_changes_this_run':0,'skipped_symlinks':0,
                'traversal_errors':0}
    start = time.monotonic()
    traversal = (out/'traversal_errors.jsonl').open('w')
    def walk_error(e):
        counters['traversal_errors'] += 1
        traversal.write(json.dumps({'path':e.filename,'error':str(e)})+'\n')
        traversal.flush()
    def progress():
        db.commit()
        elapsed = time.monotonic()-start
        obj = dict(counters, complete=False, version=VERSION, run=run,
                   elapsed_seconds=round(elapsed,2),
                   files_per_second=round(counters['encountered_pngs']/max(elapsed,.001),2))
        atomic_json(out/'progress.json',obj)
        print(json.dumps(obj),flush=True)
    try:
        for directory, dirs, names in os.walk(root, followlinks=False, onerror=walk_error):
            dirs.sort()
            for name in list(dirs):
                if (Path(directory)/name).is_symlink():
                    counters['skipped_symlinks'] += 1
                    traversal.write(json.dumps({'path':str(Path(directory)/name),'error':'symlink_directory_not_followed'})+'\n')
                    dirs.remove(name)
            for name in sorted(names):
                if not name.lower().endswith('.png'):
                    continue
                path = Path(directory)/name
                rel = str(path.relative_to(root))
                counters['encountered_pngs'] += 1
                old = db.execute('SELECT * FROM files WHERE path=?',(rel,)).fetchone()
                local, time_state, year, utc = timestamp_info(path)
                digest, error = None, None
                size = mtime = ctime = None
                try:
                    if path.is_symlink():
                        counters['skipped_symlinks'] += 1
                        raise ValueError('symlink_file_not_followed')
                    before = path.stat()
                    if not stat.S_ISREG(before.st_mode):
                        raise ValueError('Not a regular file')
                    size, mtime, ctime = stat_key(before)
                    if (old and not args.rehash and old['status'] not in TRANSIENT
                        and (old['size'],old['mtime_ns'],old['ctime_ns'])==(size,mtime,ctime)):
                        db.execute('UPDATE files SET seen_run=? WHERE path=?',(run,rel))
                        counters['reused_by_stat'] += 1
                        if counters['encountered_pngs'] % args.progress_every == 0:
                            progress()
                        continue
                    if size > args.max_file_mib*1024**2:
                        status = 'oversized_file_review'
                        error = 'Above configured per-file byte limit; bytes not decoded or hashed.'
                    else:
                        data = path.read_bytes()
                        if stat_key(path.stat()) != stat_key(before) or len(data)!=size:
                            status = 'source_changed_during_read'
                            error = 'Source changed during read; retry with --resume.'
                        else:
                            digest = sha(data)
                            cached = db.execute('SELECT status FROM objects WHERE sha=?',(digest,)).fetchone()
                            if cached:
                                status = cached['status']
                                counters['reused_by_hash'] += 1
                            else:
                                metrics = decode(data)
                                status = metrics['status']
                                db.execute('INSERT INTO objects VALUES (?,?,?)',
                                           (digest,status,json.dumps(metrics,allow_nan=False)))
                                counters['decoded_unique_this_run'] += 1
                    if old and old['sha'] and digest and old['sha']!=digest:
                        counters['source_changes_this_run'] += 1
                        db.execute('INSERT INTO changes VALUES (?,?,?,?,?)',
                                   (run,rel,old['sha'],digest,datetime.now(timezone.utc).isoformat()))
                except Exception as e:
                    status = 'symlink_file_skipped' if path.is_symlink() else 'read_error'
                    error = f'{type(e).__name__}: {e}'
                db.execute('INSERT OR REPLACE INTO files VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                           (rel,size,mtime,ctime,digest,status,error,local,time_state,year,utc,run))
                if counters['encountered_pngs'] % args.progress_every == 0:
                    progress()
    finally:
        traversal.close()
        progress()
    return counters


def write_query(db, path, query, params=()):
    cur = db.execute(query,params)
    with path.open('w',newline='') as f:
        w=csv.writer(f); w.writerow([d[0] for d in cur.description]); w.writerows(cur)


def checker_thumbnail(path):
    with Image.open(path) as im:
        im.load()
        rgba=im.convert('RGBA')
        rgba.thumbnail((300,260))
    y,x=np.indices((rgba.height,rgba.width))
    v=np.where(((x//12+y//12)%2)==0,230,190).astype('uint8')
    bg=Image.fromarray(np.repeat(v[:,:,None],3,axis=2)).convert('RGBA')
    return Image.alpha_composite(bg,rgba).convert('RGB')


def reports(root,out,db,args,run,counters):
    # Reports include only files seen in this traversal, never stale removed records.
    q='WHERE seen_run=?'
    write_query(db,out/'counts_by_year_status.csv',
                'SELECT year,status,count(*) AS files FROM files '+q+' GROUP BY year,status ORDER BY year,status',(run,))
    write_query(db,out/'timestamp_quality.csv',
                'SELECT year,time_state,count(*) AS files FROM files '+q+' GROUP BY year,time_state ORDER BY year,time_state',(run,))
    write_query(db,out/'duplicate_local_timestamps.csv',
                'SELECT local_time,count(*) AS files,count(DISTINCT sha) AS distinct_hashes FROM files '+q+
                ' AND local_time IS NOT NULL GROUP BY local_time HAVING count(*)>1 ORDER BY local_time',(run,))
    write_query(db,out/'repeated_file_hashes.csv',
                'SELECT sha,status,count(*) AS files,min(local_time) AS first_local,max(local_time) AS last_local FROM files '+q+
                ' AND sha IS NOT NULL GROUP BY sha,status HAVING count(*)>1 ORDER BY files DESC',(run,))
    write_query(db,out/'files_absent_on_rescan.csv',
                'SELECT path,sha,status FROM files WHERE seen_run<>? ORDER BY path',(run,))
    write_query(db,out/'dimensions.csv',
                '''SELECT json_extract(o.metrics,'$.width') AS width,json_extract(o.metrics,'$.height') AS height,
                json_extract(o.metrics,'$.mode') AS mode,count(*) AS files
                FROM files f LEFT JOIN objects o ON f.sha=o.sha WHERE f.seen_run=?
                GROUP BY width,height,mode ORDER BY files DESC''',(run,))
    write_query(db,out/'spatial_palette_by_year.csv',
                """SELECT f.year,count(*) AS files,
                sum(json_extract(o.metrics,'$.palette_pixels')) AS palette_pixels,
                sum(json_extract(o.metrics,'$.corrected_grid_palette_points')) AS sampled_grid_points,
                sum(json_extract(o.metrics,'$.palette_quadrants[0]')) AS northwest_pixels,
                sum(json_extract(o.metrics,'$.palette_quadrants[1]')) AS northeast_pixels,
                sum(json_extract(o.metrics,'$.palette_quadrants[2]')) AS southwest_pixels,
                sum(json_extract(o.metrics,'$.palette_quadrants[3]')) AS southeast_pixels
                FROM files f JOIN objects o ON f.sha=o.sha WHERE f.seen_run=?
                AND json_extract(o.metrics,'$.width')=656 AND json_extract(o.metrics,'$.height')=654
                AND f.status NOT IN ('decode_error','not_png_format_review','multiframe_review')
                GROUP BY f.year ORDER BY f.year""",(run,))
    with gzip.open(out/'files.csv.gz','wt',newline='') as f:
        cur=db.execute('''SELECT f.*,o.metrics FROM files f LEFT JOIN objects o ON f.sha=o.sha
                          WHERE f.seen_run=? ORDER BY f.path''',(run,))
        w=csv.writer(f); w.writerow([d[0] for d in cur.description]); w.writerows(cur)
    # Temporal diagnostics: unique local labels only. DST ambiguity is retained, not resolved.
    with (out/'temporal_runs.csv').open('w',newline='') as f, (out/'observed_gaps.csv').open('w',newline='') as g:
        w=csv.writer(f); w.writerow(['start_local','end_local','status','files','same_file_hash','cadence_minutes'])
        gw=csv.writer(g); gw.writerow(['previous_local','current_local','gap_minutes','note'])
        previous=None; streak=None; run_counts=0; gap_counts=0
        def emit(s):
            if s:
                w.writerow([s['first'],s['last'],s['status'],s['n'],s['same'],args.cadence_minutes])
        for row in db.execute('''SELECT local_time,min(status) AS status,min(sha) AS sha,min(time_state) AS time_state
             FROM files WHERE seen_run=? AND local_time IS NOT NULL GROUP BY local_time
             HAVING count(*)=1 ORDER BY local_time''',(run,)):
            if row['time_state']!='unique':
                emit(streak); streak=None; previous=None; continue
            t=datetime.fromisoformat(row['local_time'])
            gap=(t-previous).total_seconds()/60 if previous else None
            if gap is not None and gap>args.cadence_minutes:
                gap_counts+=1
                gw.writerow([previous.isoformat(),t.isoformat(),gap,'Observed local-label gap; not proof of outage or missing expected file.'])
            if streak and gap==args.cadence_minutes and row['status']==streak['status']:
                streak['last']=row['local_time']; streak['n']+=1
                streak['same'] = streak['same'] and bool(row['sha']) and row['sha']==streak['sha']
            else:
                emit(streak); run_counts+=1
                streak=dict(first=row['local_time'],last=row['local_time'],status=row['status'],
                            n=1,same=bool(row['sha']),sha=row['sha'])
            previous=t
        emit(streak)
    review=out/'review'/run
    (review/'samples').mkdir(parents=True)
    selected=[]; bytes_copied=0
    groups=db.execute('SELECT year,status FROM files WHERE seen_run=? GROUP BY year,status ORDER BY year,status',(run,)).fetchall()
    for group in [dict(g, sample_rank=rank) for rank in range(args.samples_per_group) for g in groups]:
        # Earliest examples with distinct file hashes within each year/category.
        # Selection is diagnostic, not a representative random meteorological sample.
        for row in db.execute('''SELECT min(f.path) AS path,f.sha,f.status,o.metrics
             FROM files f LEFT JOIN objects o ON f.sha=o.sha WHERE f.seen_run=? AND f.year=? AND f.status=?
             GROUP BY coalesce(f.sha,f.path) ORDER BY path LIMIT 1 OFFSET ?''',
             (run,group['year'],group['status'],group['sample_rank'])):
            if len(selected)>=args.max_samples: break
            entry=dict(row); entry['year']=group['year']; entry['copy_status']='not_copied'; entry['sample_file']=''
            original=root/row['path']
            try:
                size=original.stat().st_size
                if original.is_symlink() or size>10*1024**2 or bytes_copied+size>args.sample_budget_mib*1024**2:
                    raise ValueError('Sample size budget or symlink restriction')
                data=original.read_bytes()
                if row['sha'] is None or sha(data)!=row['sha']:
                    raise ValueError('No stable source hash or source changed since scan')
                target=review/'samples'/f'{len(selected):03d}_{original.name}'
                target.write_bytes(data); bytes_copied+=len(data)
                entry.update(copy_status='copied',sample_file=str(target.relative_to(review)))
            except Exception as e:
                entry['copy_status']=f'{type(e).__name__}: {e}'
            selected.append(entry)
    with (review/'samples.csv').open('w',newline='') as f:
        fields=['year','status','path','sha','copy_status','sample_file','metrics']
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(selected)
    for start in range(0,len(selected),12):
        page_rows=(min(12,len(selected)-start)+2)//3
        page=Image.new('RGB',(1000,310*page_rows),'white'); draw=ImageDraw.Draw(page)
        for i,entry in enumerate(selected[start:start+12]):
            x=(i%3)*330+5; y=(i//3)*310+5
            if entry['sample_file']:
                try: page.paste(checker_thumbnail(review/entry['sample_file']),(x,y))
                except Exception: draw.text((x,y+40),'Image cannot be decoded',fill='red')
            else: draw.text((x,y+40),'No sample copy; see samples.csv',fill='red')
            draw.text((x,y+262),f'{start+i:03d} {Path(entry["path"]).name}',fill='black')
            s=entry['status']
            draw.text((x,y+277),s[:43],fill='black')
            draw.text((x,y+291),s[43:],fill='black')
        page.save(review/f'panel_{start//12+1:03d}.jpg',quality=88)
    counts=dict(db.execute('SELECT status,count(*) FROM files WHERE seen_run=? GROUP BY status',(run,)).fetchall())
    summary=dict(version=VERSION,complete=counters['traversal_errors']==0 and counters['skipped_symlinks']==0 and counters['encountered_pngs']>0,
                 scan_finished=True,production_ready=False,dry_rule_documented=False,
                 observational_validity_inferred=False,root=str(root),run=run,counts=counts,
                 processed_files=sum(counts.values()),**counters,
                 unique_file_hashes=db.execute('SELECT count(DISTINCT sha) FROM files WHERE seen_run=?',(run,)).fetchone()[0],
                 temporal_runs=run_counts,observed_gaps=gap_counts,samples=len(selected),
                 sample_bytes=bytes_copied,review_directory=str(review.relative_to(out)),
                 note='Complete means traversal finished without inaccessible directories or skipped symlinks. Per-file errors remain in counts. No hourly subsampling. All PNG suffixes are considered. The root must be quiescent for a consistent census.')
    atomic_json(out/'summary.json',summary)
    small=['summary.json','config.json','counts_by_year_status.csv','dimensions.csv','timestamp_quality.csv','spatial_palette_by_year.csv','traversal_errors.jsonl']
    with zipfile.ZipFile(out/'review_bundle.zip.tmp','w',zipfile.ZIP_DEFLATED) as z:
        for name in small: z.write(out/name,name)
        for path in sorted(review.rglob('*')):
            if path.is_file(): z.write(path,str(Path('review')/path.relative_to(review)))
    os.replace(out/'review_bundle.zip.tmp',out/'review_bundle.zip')
    print(json.dumps(summary,indent=2),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--radar-dir',required=True,type=Path)
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--rehash',action='store_true',help='Reread every file even if stat fingerprint is unchanged.')
    p.add_argument('--progress-every',type=int,default=1000)
    p.add_argument('--cadence-minutes',type=int,default=2,help='Diagnostic only; never filters files.')
    p.add_argument('--samples-per-group',type=int,default=2)
    p.add_argument('--max-samples',type=int,default=160)
    p.add_argument('--sample-budget-mib',type=int,default=96)
    p.add_argument('--max-file-mib',type=int,default=64)
    args=p.parse_args()
    for name in ['progress_every','cadence_minutes','samples_per_group','max_samples','sample_budget_mib','max_file_mib']:
        if getattr(args,name)<=0: p.error(f'{name} must be positive')
    root=args.radar_dir.resolve(); out=args.output_dir.resolve()
    if not root.is_dir(): p.error('radar-dir must be an existing directory')
    if root==out or root in out.parents or out in root.parents: p.error('Input and output directory trees must be separate')
    if out.exists() and any(out.iterdir()) and not args.resume: p.error('Output exists; use --resume or a new directory')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.audit.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: p.error('Another auditor holds the output lock')
        config=dict(version=VERSION,radar_dir=str(root),script_sha256=sha(Path(__file__).read_bytes()),
                    numpy_version=np.__version__,pillow_version=PIL_VERSION,
                    palette_rgb_dbz=list(zip(PALETTE,DBZ)),expected_width=656,expected_height=654,
                    reference=dict(center_lat=-22.955139,center_lon=-43.248278,range_m=138900,
                                   earth_radius_m=6371008.8,projection='spherical AEQD',grid_shape=[70,84],
                                   validity='reference geometry only; not observational validity'),
                    options={k:getattr(args,k) for k in ['cadence_minutes','samples_per_group','max_samples','sample_budget_mib','max_file_mib']})
        config=json.loads(json.dumps(config))
        cp=out/'config.json'
        if cp.exists():
            if json.loads(cp.read_text())!=config: p.error('Configuration/code/dependencies changed; use a new output directory')
        else: atomic_json(cp,config)
        run=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
        db=sqlite3.connect(out/'audit.sqlite'); db.row_factory=sqlite3.Row
        init_db(db)
        atomic_json(out/'summary.json',dict(version=VERSION,complete=False,run=run,stage='scan'))
        try:
            c=scan(root,out,db,args,run)
            atomic_json(out/'summary.json',dict(version=VERSION,complete=False,run=run,stage='reports',**c))
            result=reports(root,out,db,args,run,c)
            db.commit()
            db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
            return 0 if result['complete'] else 2
        finally:
            db.close()

if __name__=='__main__':
    try: sys.exit(main())
    except KeyboardInterrupt:
        print('Interrupted. Restart the same command with --resume.',file=sys.stderr)
        sys.exit(130)
