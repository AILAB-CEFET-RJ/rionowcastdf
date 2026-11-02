#!/usr/bin/env python3
"""Lossless source archive + revisable radar cache. No training validity inference."""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile

import numpy as np
from PIL import Image
from alignment_reference import CONTRACT, geometry, analyze_array, png_path, utc_candidates, parse_utc

VERSION = '14j-intermediate-cache-v1'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.partial-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def json_bytes(obj):
    return (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)+'\n').encode()

def npz_bytes(**arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()

def code_hashes():
    return {name: sha(Path(__file__).with_name(name).read_bytes())
            for name in ('build_cache.py', 'alignment_reference.py')}

def decode(data, g):
    """Preserve all RGBA including hidden RGB. Never composite alpha or fill dry."""
    with Image.open(io.BytesIO(data)) as im:
        mode, size, nframes = im.mode, im.size, getattr(im, 'n_frames', 1)
        im.load()
        rgba = np.array(im.convert('RGBA'), dtype=np.uint8)
        info = {'source_mode': mode, 'source_size': list(size), 'source_frames': nframes,
                'source_info_repr': repr(im.info)}
    # Native categories do not assert observational validity.
    # 0 transparent, 1..7 exact opaque palette, 254 partial alpha, 255 other opaque.
    alpha = rgba[...,3]
    classes = np.full(alpha.shape, 255, dtype=np.uint8)
    classes[alpha == 0] = 0
    classes[(alpha > 0) & (alpha < 255)] = 254
    for index, rgb in enumerate(CONTRACT['palette_rgb'], 1):
        classes[(alpha == 255) & np.all(rgba[...,:3] == rgb, axis=2)] = index
    arrays = dict(native_rgba=rgba, native_class=classes)
    info.update(transparent_pixels=int((alpha == 0).sum()),
                partial_alpha_pixels=int(((alpha > 0)&(alpha < 255)).sum()),
                unrecognized_opaque_pixels=int((classes == 255).sum()),
                exact_palette_pixels=int(((classes >= 1)&(classes <= 7)).sum()))
    if size != (656,654):
        info['status'] = 'unexpected_dimensions_review'
    elif nframes != 1:
        info['status'] = 'multiframe_review'
    else:
        _, grid, _, review, counts = analyze_array(rgba, g)
        info.update(counts)
        mask = g['geometric_mask']; y = g['py'][mask]; x = g['px'][mask]
        grid_class = np.full(mask.shape, 253, np.uint8) # outside geometry
        grid_class[mask] = classes[y,x]
        grid_alpha = np.zeros(mask.shape, np.uint8)
        grid_alpha[mask] = alpha[y,x]
        arrays.update(grid_class=grid_class, grid_alpha=grid_alpha,
                      grid_dbz_palette=grid, grid_dbz_review=review,
                      geometric_mask=mask)
    arrays['metadata_json'] = np.array(json.dumps(info, ensure_ascii=False))
    return info, arrays

def open_db(path):
    db = sqlite3.connect(path)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=FULL')
    db.executescript('''
      CREATE TABLE IF NOT EXISTS sources (
        sha256 TEXT PRIMARY KEY, original_path TEXT NOT NULL,
        decoded_path TEXT, decoded_sha256 TEXT, status TEXT NOT NULL, metadata_json TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS frames (
        timestamp_utc TEXT PRIMARY KEY, timestamp_local TEXT NOT NULL,
        source_path TEXT NOT NULL, source_sha256 TEXT,
        utc_offset_hours REAL NOT NULL, local_filename_ambiguous INTEGER NOT NULL,
        status TEXT NOT NULL, error TEXT, checked_at_utc TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS changes (
        id INTEGER PRIMARY KEY, timestamp_utc TEXT, old_sha256 TEXT,
        new_sha256 TEXT, changed_at_utc TEXT);
    ''')
    return db

def process_source(db, root, data, g):
    digest = sha(data)
    row = db.execute('SELECT original_path,decoded_path,decoded_sha256,status,metadata_json FROM sources WHERE sha256=?',(digest,)).fetchone()
    if row:
        original, decoded, checksum, status, metadata = row
        if not (root/original).exists() or sha((root/original).read_bytes()) != digest:
            raise RuntimeError('Original cached object missing/corrupt: '+original)
        if decoded and (not (root/decoded).exists() or sha((root/decoded).read_bytes()) != checksum):
            raise RuntimeError('Decoded cached object missing/corrupt: '+decoded)
        return digest, status, json.loads(metadata)
    original = f'objects/{digest[:2]}/{digest}.png'
    atomic(root/original, data) # even unreadable PNGs retained for provenance
    try:
        info, arrays = decode(data, g)
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as e:
        info = {'status':'decode_error', 'error':f'{type(e).__name__}: {e}'}
        decoded = checksum = None
    else:
        decoded = f'decoded/{digest[:2]}/{digest}.npz'
        payload = npz_bytes(**arrays)
        checksum = sha(payload)
        atomic(root/decoded, payload)
    db.execute('INSERT INTO sources VALUES (?,?,?,?,?,?)',
               (digest,original,decoded,checksum,info['status'],json.dumps(info,ensure_ascii=False)))
    return digest, info['status'], info

def process_frame(db, root, radar, timestamp, g):
    source = png_path(radar, timestamp)
    local = timestamp.astimezone(__import__('zoneinfo').ZoneInfo('America/Sao_Paulo'))
    ambiguous = len(utc_candidates(local.replace(tzinfo=None))) != 1
    stamp = timestamp.isoformat()
    old = db.execute('SELECT source_sha256 FROM frames WHERE timestamp_utc=?',(stamp,)).fetchone()
    error = None; digest = None
    try:
        data = source.read_bytes()
    except FileNotFoundError:
        status = 'missing_png'
    except OSError as e:
        status = 'source_read_error'; error = f'{type(e).__name__}: {e}'
    else:
        digest, status, info = process_source(db, root, data, g)
        error = info.get('error')
    now = datetime.now(timezone.utc).isoformat()
    # Keep descriptive source status separate from temporal ambiguity flag.
    if old and old[0] != digest:
        db.execute('INSERT INTO changes(timestamp_utc,old_sha256,new_sha256,changed_at_utc) VALUES (?,?,?,?)',
                   (stamp,old[0],digest,now))
    db.execute('INSERT OR REPLACE INTO frames VALUES (?,?,?,?,?,?,?,?,?)',
               (stamp,local.isoformat(),str(source),digest,local.utcoffset().total_seconds()/3600,
                int(ambiguous),status,error,now))
    db.commit() # source objects and frame become visible together
    return status

def report(db, root, expected, complete, config_hash):
    counts = dict(db.execute('SELECT status,COUNT(*) FROM frames GROUP BY status'))
    n = sum(counts.values())
    result = dict(version=VERSION,config_sha256=config_hash,expected_timestamps=expected,
                  processed_timestamps=n,complete=complete and n==expected,counts=counts,
                  unique_source_objects=db.execute('SELECT COUNT(*) FROM sources').fetchone()[0],
                  ambiguous_local_timestamps=db.execute('SELECT COUNT(*) FROM frames WHERE local_filename_ambiguous=1').fetchone()[0],
                  source_changes=db.execute('SELECT COUNT(*) FROM changes').fetchone()[0],
                  production_ready=False,dry_rule_documented=False,era5_values_read=False,
                  temporal_aggregation='none',old_cache_used=False,
                  note='Complete is inventory completion, not meteorological approval. No observational validity mask has been inferred.')
    atomic(root/'summary.json',json_bytes(result))
    return result

def run(args):
    start, end = parse_utc(args.start_utc), parse_utc(args.end_utc)
    if end < start: raise ValueError('end must be >= start (inclusive)')
    step = timedelta(minutes=args.cadence_minutes)
    if (end-start).total_seconds() % step.total_seconds():
        raise ValueError('end must align with start and cadence')
    radar = Path(args.radar_dir).resolve()
    if not radar.is_dir(): raise ValueError('radar-dir does not exist')
    root = Path(args.output_dir).resolve()
    if root == radar or radar in root.parents or root in radar.parents:
        raise ValueError('Output and source trees must be separate')
    config = dict(version=VERSION,start_utc=start.isoformat(),end_utc=end.isoformat(),
                  cadence_minutes=args.cadence_minutes,radar_dir=str(radar),
                  source_layout='YYYY/MM/DD/YYYY_MM_DD_HH_MM.png',
                  reference_contract=CONTRACT,code_sha256=code_hashes(),
                  semantic_policy='unresolved; no dry fill; no observation validity inferred',
                  native_classes={'0':'transparent','1..7':'opaque exact palette in contract order',
                                  '253':'outside geometric grid','254':'partial alpha','255':'unrecognized opaque'})
    payload = json_bytes(config); fingerprint=sha(payload)
    existed = root.exists()
    if existed and not args.resume: raise FileExistsError('Output exists: use --resume with the identical contract')
    root.mkdir(parents=True,exist_ok=True)
    # POSIX lock is released by the OS on interruption; no stale lock deletion needed.
    import fcntl
    with (root/'.writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'config.json').exists():
            if (root/'config.json').read_bytes()!=payload:
                raise ValueError('Resume contract differs: use a new output directory')
        elif existed and any(p.name!='.writer.lock' for p in root.iterdir()):
            raise ValueError('Output has files but no config; use a new directory')
        else: atomic(root/'config.json',payload)
        g=geometry()
        geom=root/'geometry.npz'
        if not geom.exists(): atomic(geom,npz_bytes(**g))
        else:
            with np.load(geom,allow_pickle=False) as saved:
                if set(saved.files)!=set(g) or any(not np.array_equal(saved[k],v) for k,v in g.items()):
                    raise RuntimeError('Geometry file differs from contract')
        db=open_db(root/'manifest.sqlite')
        expected=int((end-start)/step)+1
        report(db,root,expected,False,fingerprint)
        done=False
        try:
            t=start; i=0
            while t<=end:
                status=process_frame(db,root,radar,t,g)
                i+=1
                if i==1 or i%args.progress_every==0:
                    print(f'{i}/{expected} {t.isoformat()} {status}',flush=True)
                    report(db,root,expected,False,fingerprint)
                t+=step
            done=True
        finally:
            result=report(db,root,expected,done,fingerprint)
            db.close()
        print(json.dumps(result,indent=2),flush=True)
        return 2 if any(result['counts'].get(k,0) for k in ('decode_error','source_read_error')) else 0

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--radar-dir',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--start-utc',required=True,help='Inclusive UTC, e.g. 2011-01-01T00:00:00Z')
    p.add_argument('--end-utc',required=True,help='Inclusive UTC')
    p.add_argument('--cadence-minutes',type=int,choices=(2,60),default=60)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--progress-every',type=int,default=250)
    args=p.parse_args()
    if args.progress_every<1: p.error('progress-every must be positive')
    try: return run(args)
    except KeyboardInterrupt:
        print('Interrupted; committed progress preserved. Resume with same arguments and --resume.',file=sys.stderr)
        return 130
    except Exception as e:
        print(f'FATAL {type(e).__name__}: {e}',file=sys.stderr)
        return 1

if __name__=='__main__': sys.exit(main())
