#!/usr/bin/env python3
"""14p: immutable census policy, temporal comparison, hash review and explicit freeze.
Never writes input databases, decodes the archive, or creates targets/Zarr.
"""
import argparse
import collections
import csv
from datetime import datetime, timezone
import gzip
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
from zoneinfo import ZoneInfo
import pandas as pd

VERSION = '14p-temporal-policy-v1'
BOUNDARY = int(datetime(2011, 1, 17, 2, tzinfo=timezone.utc).timestamp())
TZ = ZoneInfo('America/Sao_Paulo')
NORMAL = {
 'transparent_ambiguous': 'valid_transparent_no_echo',
 'palette_in_png_not_sampled_in_corrected_grid': 'valid_no_echo_on_sampled_grid',
 'palette_sampled_observation_validity_unconfirmed': 'valid_echo_and_no_echo'}
REVIEW = {'map_screen_review', 'no_exact_palette_review'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def dump(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temp.replace(path)


def read_json(path): return json.loads(Path(path).read_text())


def readonly(path):
    return sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True)


def new_output(path, inputs):
    path = Path(path).resolve()
    for p in inputs:
        p = Path(p).resolve()
        if path == p or path in p.parents or (p.is_dir() and p in path.parents):
            raise ValueError('Output and input trees must be separate')
    path.mkdir(parents=True, exist_ok=False)
    dump(path/'summary.json', {'version': VERSION, 'complete': False})
    return path


def utc_seconds(local):
    t = datetime.fromisoformat(local)
    if t.tzinfo is not None: raise ValueError('Expected naive local label')
    options = set()
    for fold in (0, 1):
        u = t.replace(tzinfo=TZ, fold=fold).astimezone(timezone.utc)
        if u.astimezone(TZ).replace(tzinfo=None) == t: options.add(int(u.timestamp()))
    return next(iter(options)) if len(options) == 1 else None


def split_at(sec):
    t = datetime.fromtimestamp(sec, timezone.utc)
    for y in (2021, 2023, 2024):
        b = datetime(y,1,1,tzinfo=timezone.utc).timestamp()
        if b-48*3600 <= sec < b+48*3600: return 'excluded_purge'
    return 'train' if t.year <= 2020 else ('validation' if t.year <= 2022 else ('test_primary' if t.year == 2023 else 'test_stress_ood'))


def reviews(path):
    if not path: return {}
    result = {}
    with Path(path).open(newline='') as f:
        for r in csv.DictReader(f):
            h = r['source_sha256']; d = r['decision']
            if len(h)!=64 or any(c not in '0123456789abcdef' for c in h): raise ValueError('Invalid review SHA')
            if h in result: raise ValueError('Duplicate review hash')
            if d not in ('pending','reject','accept_no_echo'): raise ValueError('Invalid decision')
            if d != 'pending' and not all(r.get(k,'').strip() for k in ('evidence','reviewer','reviewed_at')):
                raise ValueError('Review requires evidence, reviewer and reviewed_at')
            result[h] = r
    return result


def policy(status, sha, decisions, metrics):
    if status in NORMAL: return NORMAL[status], True
    decision = decisions.get(sha, {}).get('decision', 'pending')
    if decision == 'accept_no_echo':
        if status != 'map_screen_review' or metrics.get('palette_pixels') != 0:
            raise ValueError('accept_no_echo only allowed for map hash with zero palette pixels')
        return 'reviewed_map_no_echo', True
    return ('excluded_review_reject' if decision == 'reject' else 'excluded_'+status), False


def write_counter(path, counter, names):
    with Path(path).open('w', newline='') as f:
        w = csv.writer(f); w.writerow(names+['files'])
        for k,v in sorted(counter.items()): w.writerow(list(k)+[v])


def select(db, sec, tolerance, split):
    rows = db.execute('SELECT sec,path,sha,policy FROM candidates WHERE sec BETWEEN ? AND ? AND split=? ORDER BY abs(sec-?),sec,path',
                      (sec-tolerance, sec+tolerance, split, sec)).fetchall()
    return rows[0] if rows else None


def analyze(args):
    audit = args.audit_dir.resolve(); pairing = args.pairing_dir.resolve()
    summary = read_json(audit/'summary.json'); pair_summary = read_json(pairing/'summary.json')
    if summary.get('version') != '14m-full-png-audit-v1' or summary.get('complete') is not True:
        raise ValueError('Requires complete 14m census')
    if not pair_summary.get('complete') or pair_summary.get('provisional') or pair_summary.get('era5_error_files',0):
        raise ValueError('Requires complete, nonprovisional, error-free pairing')
    decisions = reviews(args.reviews)
    out = new_output(args.output_dir, [audit,pairing]+([args.radar_dir] if args.radar_dir else []))
    print('Snapshot SQLite consistente; nenhuma escrita na origem.', flush=True)
    with (audit/'.audit.lock').open('rb') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        if read_json(audit/'summary.json') != summary:
            raise ValueError('Audit changed before snapshot')
        with readonly(audit/'audit.sqlite') as src, sqlite3.connect(out/'audit_snapshot.sqlite') as dst:
            src.backup(dst)
        for name in ('summary.json','config.json'):
            shutil.copy2(audit/name, out/('source_'+name))
    shutil.copy2(pairing/'paired_manifest.parquet', out/'pairing_snapshot.parquet')
    shutil.copy2(pairing/'summary.json', out/'pairing_summary.json')
    if args.reviews: shutil.copy2(args.reviews,out/'review_decisions.csv')
    db = sqlite3.connect(out/'temporal_index.sqlite')
    db.executescript('CREATE TABLE candidates(sec INTEGER,path TEXT,sha TEXT,policy TEXT,split TEXT); CREATE TABLE review_groups(sha TEXT,year TEXT,status TEXT,path TEXT,files INTEGER,metrics TEXT);')
    counts = collections.Counter(); cadence = collections.Counter(); used_reviews = set(); n = 0
    with readonly(out/'audit_snapshot.sqlite') as src, gzip.open(out/'policy_files.csv.gz','wt',newline='') as f:
        src.row_factory = sqlite3.Row
        w = csv.writer(f); w.writerow(['path','sha','status','policy_status','time_state','timestamp_utc','eligible'])
        # Duplicated local labels are excluded even when content is identical.
        duplicates = {r[0] for r in src.execute('SELECT local_time FROM files WHERE seen_run=? AND local_time IS NOT NULL GROUP BY local_time HAVING count(*)>1',(summary['run'],))}
        q = 'SELECT f.*,o.metrics FROM files f LEFT JOIN objects o ON f.sha=o.sha WHERE f.seen_run=? ORDER BY f.path'
        for r in src.execute(q,(summary['run'],)):
            n += 1; metrics = json.loads(r['metrics'] or '{}')
            p, ok = policy(r['status'], r['sha'], decisions, metrics)
            if r['sha'] in decisions: used_reviews.add(r['sha'])
            sec = utc_seconds(r['local_time']) if r['time_state']=='unique' and r['local_time'] else None
            if sec is None: ok=False; reason='excluded_time_'+str(r['time_state'])
            elif r['local_time'] in duplicates: ok=False; reason='excluded_duplicate_local_label'
            elif sec < BOUNDARY: ok=False; reason='outside_archive_period'
            else: reason=p
            counts[(str(r['year']),reason)] += 1
            if r['local_time']:
                t = datetime.fromisoformat(r['local_time']); cadence[(t.strftime('%Y-%m'),f'{t.minute:02d}')] += 1
            iso = datetime.fromtimestamp(sec,timezone.utc).isoformat() if sec is not None else ''
            w.writerow([r['path'],r['sha'],r['status'],reason,r['time_state'],iso,ok])
            if ok: db.execute('INSERT INTO candidates VALUES(?,?,?,?,?)',(sec,r['path'],r['sha'],p,split_at(sec)))
            if n % 100000 == 0: db.commit(); print(f'Política {n}/{summary["processed_files"]}',flush=True)
        if n != summary['processed_files']: raise ValueError('Snapshot count differs from completed census')
        if set(decisions)-used_reviews: raise ValueError('Review hashes not found in census')
        for r in src.execute('''SELECT f.sha,f.year,f.status,min(f.path),count(*),o.metrics FROM files f
          LEFT JOIN objects o ON f.sha=o.sha WHERE f.seen_run=? AND f.status IN ('map_screen_review','no_exact_palette_review')
          GROUP BY f.sha,f.year,f.status ORDER BY count(*) DESC,f.sha,f.year''',(summary['run'],)):
            db.execute('INSERT INTO review_groups VALUES(?,?,?,?,?,?)',tuple(r))
    db.execute('CREATE INDEX candidate_time ON candidates(sec,split)'); db.commit()
    write_counter(out/'policy_counts_by_year.csv',counts,['year','policy_status'])
    write_counter(out/'minute_distribution.csv',cadence,['local_month','minute'])
    build_review(db, out, args.radar_dir, args.max_samples)
    compare(db, out)
    db.close()
    artifacts = {p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='summary.json'}
    dump(out/'summary.json',dict(version=VERSION,complete=True,processed_files=n,
      no_echo_rule_documented=True,production_ready=False,temporal_rule_finalized=False,
      archive_boundary_utc='2011-01-17T02:00:00+00:00',timezone='America/Sao_Paulo',
      tie_break='earlier UTC; exact has precedence',split_purge_hours=48,
      aggregation='none',raw_pngs_read='only bounded review samples if radar-dir supplied',
      artifact_sha256=artifacts,script_sha256=digest(__file__),
      note='Frame eligibility only. Pixel masks and numeric/spatial contracts remain those of 14n/14o.'))
    print(json.dumps(read_json(out/'summary.json'),indent=2),flush=True)


def build_review(db, out, root, limit):
    fields=['source_sha256','year','status','representative_path','files','metrics','decision','evidence','reviewer','reviewed_at']
    with (out/'review_groups.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(fields)
        for row in db.execute('SELECT * FROM review_groups ORDER BY files DESC,sha,year'):
            w.writerow(list(row)+['pending','','',''])
    # First representative for each year/status, then highest-frequency remaining hashes.
    ordered=list(db.execute('SELECT * FROM review_groups ORDER BY files DESC,sha,year'))
    selected=[]; groups=set(); hashes=set()
    for row in ordered:
        group=(row[1],row[2])
        if group not in groups and row[0] not in hashes:
            selected.append(row);groups.add(group);hashes.add(row[0])
    for row in ordered:
        if row[0] not in hashes: selected.append(row);hashes.add(row[0])
    selected=selected[:limit]
    with (out/'review_template.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['source_sha256','decision','evidence','reviewer','reviewed_at'])
        for r in selected: w.writerow([r[0],'pending','','',''])
    with (out/'review_samples.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['sha','year','status','source','files','copy_status'])
        for h,year,status,rel,n,_ in selected:
            state='not_requested'
            if root:
                source=(root/rel).resolve()
                if root.resolve() not in source.parents: raise ValueError('Source path outside radar-dir')
                try:
                    if source.stat().st_size>64*1024*1024: raise ValueError('Sample exceeds 64 MiB')
                    if digest(source)!=h: raise ValueError('Source hash changed')
                    folder=out/'review_pngs';folder.mkdir(exist_ok=True)
                    shutil.copyfile(source,folder/(h+'.png'))
                    if digest(folder/(h+'.png')) != h:
                        (folder/(h+'.png')).unlink()
                        raise ValueError('Source changed during copy')
                    state='copied'
                except (OSError,ValueError) as e: state=str(e)
            w.writerow([h,year,status,rel,n,state])


def compare(db, out):
    df=pd.read_parquet(out/'pairing_snapshot.parquet',columns=['timestamp_utc','split','era5_technical_candidate'])
    df['timestamp_utc']=pd.to_datetime(df.timestamp_utc,utc=True)
    if df.timestamp_utc.duplicated().any(): raise ValueError('Duplicate pairing timestamps')
    if df.era5_technical_candidate.isna().any() or not df.era5_technical_candidate.isin([True,False,0,1]).all(): raise ValueError('Invalid ERA5 flags')
    records=[]
    for ts,sp,era in df.itertuples(index=False,name=None):
        sec=int(ts.timestamp())
        if sec<BOUNDARY: continue
        if split_at(sec)!=sp: raise ValueError('Pairing splits differ from locked 14k 48h purge contract')
        if sec%3600: raise ValueError('Reference is not an exact UTC hour')
        for name,tol in [('exact',0),('nearest2',120),('nearest5',300)]:
            match=select(db,sec,tol,sp)
            found=match is not None
            records.append(dict(timestamp_utc=ts.isoformat(),year=ts.year,split=sp,strategy=name,
              era5_ready=bool(era),radar_found=found,paired=found and bool(era) and sp!='excluded_purge',
              source_utc=datetime.fromtimestamp(match[0],timezone.utc).isoformat() if found else '',
              offset_seconds=match[0]-sec if found else None,path=match[1] if found else '',
              sha=match[2] if found else '',policy_status=match[3] if found else 'missing_eligible_png'))
    result=pd.DataFrame(records)
    if result.empty: raise ValueError('Empty comparison')
    result.to_parquet(out/'temporal_comparison.parquet',index=False)
    grouped=result.groupby(['year','split','strategy']).agg(hours=('paired','size'),radar_found=('radar_found','sum'),
        era5_ready=('era5_ready','sum'),paired=('paired','sum'))
    grouped['paired_percent']=100*grouped.paired/grouped.hours
    grouped.to_csv(out/'temporal_coverage.csv')
    result[result.radar_found].groupby(['split','strategy','offset_seconds']).size().rename('hours').to_csv(out/'offset_distribution.csv')
    repeated=result[result.paired].groupby(['strategy','path']).agg(hours=('timestamp_utc','size'),splits=('split','nunique'))
    repeated[repeated.hours>1].to_csv(out/'reused_source_times.csv')
    train=result[result.split.eq('train')].groupby('strategy').paired.sum().to_dict()
    dump(out/'decision_support.json',{'train_paired_hours':{k:int(v) for k,v in train.items()},
      'selection_scope':'train only; other splits availability reporting only',
      'nearest2_additional_train_hours':int(train.get('nearest2',0)-train.get('exact',0)),
      'nearest5_additional_train_hours':int(train.get('nearest5',0)-train.get('exact',0)),
      'decision':'pending explicit freeze', 'no_target_averaging':True})


def freeze(args):
    src=args.analysis_dir.resolve(); s=read_json(src/'summary.json')
    if not s.get('complete') or s.get('version')!=VERSION: raise ValueError('Incomplete analysis')
    for name in ('temporal_comparison.parquet','decision_support.json'):
        if digest(src/name)!=s['artifact_sha256'][name]: raise ValueError('Analysis artifact hash mismatch')
    if not args.rationale.strip(): raise ValueError('Rationale required')
    out=new_output(args.output_dir,[src])
    df=pd.read_parquet(src/'temporal_comparison.parquet')
    df=df[df.strategy.eq(args.strategy)&df.paired].copy()
    if df.empty: raise ValueError('Selected strategy has no pairs')
    if df.timestamp_utc.duplicated().any(): raise ValueError('Duplicate target hour')
    if df.groupby('path').split.nunique().gt(1).any(): raise ValueError('Cross-split source reuse')
    df.to_parquet(out/'selected_manifest.parquet',index=False)
    df.groupby('split').size().rename('hours').to_csv(out/'selected_counts.csv')
    required=df[['path','sha']].drop_duplicates().copy()
    if args.radar_cache_dir:
        cache=args.radar_cache_dir.resolve()
        cs=read_json(cache/'summary.json')
        if not cs.get('complete'): raise ValueError('14j cache incomplete')
        with readonly(cache/'manifest.sqlite') as cdb:
            available={r[0]:r[1:] for r in cdb.execute('SELECT sha256,decoded_path,decoded_sha256 FROM sources WHERE decoded_path IS NOT NULL')}
        required['decoded_path']=required.sha.map(lambda h: available.get(h,('', ''))[0])
        required['decoded_sha256']=required.sha.map(lambda h: available.get(h,('', ''))[1])
        required['in_cache_index']=required.sha.isin(available)
        required[~required.in_cache_index].to_csv(out/'additional_sources_to_decode.csv',index=False)
        dump(out/'cache_reference.json',{'path':str(cache),'summary_sha256':digest(cache/'summary.json'),
             'note':'Index match by content hash only; loader must validate decoded object bytes.'})
    required.to_csv(out/'required_sources.csv',index=False)
    dump(out/'temporal_contract.json',dict(version=VERSION,strategy=args.strategy,rationale=args.rationale,
      tolerance_seconds={'exact':0,'nearest2':120,'nearest5':300}[args.strategy],
      tie_break='earlier UTC',timestamp_reference='ERA5 UTC exact hour',source_timezone='America/Sao_Paulo',
      aggregation='none',missing_policy='exclude; never fill dry',no_echo_code=0,
      no_echo_semantics='no displayed echo, not measured 0 dBZ',
      source_analysis_sha256=digest(src/'summary.json'),decision_scope='train only',
      spatial_numeric_contract='unchanged 14o; preserve unknown pixel masks',
      created_at=datetime.now(timezone.utc).isoformat()))
    dump(out/'summary.json',dict(version=VERSION,complete=True,temporal_rule_finalized=True,
      selected_hours=len(df),production_ready=False,zarr_created=False,cache_modified=False,
      artifact_sha256={p.name:digest(p) for p in out.iterdir() if p.name!='summary.json'}))
    print(json.dumps(read_json(out/'summary.json'),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('analyze');a.add_argument('--audit-dir',type=Path,required=True)
    a.add_argument('--pairing-dir',type=Path,required=True);a.add_argument('--output-dir',type=Path,required=True)
    a.add_argument('--radar-dir',type=Path);a.add_argument('--reviews',type=Path)
    a.add_argument('--max-samples',type=int,default=40)
    b=sub.add_parser('freeze');b.add_argument('--analysis-dir',type=Path,required=True)
    b.add_argument('--output-dir',type=Path,required=True);b.add_argument('--strategy',choices=['exact','nearest2','nearest5'],required=True)
    b.add_argument('--rationale',required=True);b.add_argument('--radar-cache-dir',type=Path)
    args=p.parse_args()
    if args.command=='analyze' and args.max_samples<0: p.error('max-samples must be nonnegative')
    try: analyze(args) if args.command=='analyze' else freeze(args)
    except Exception as e:
        p.exit(2,f'{type(e).__name__}: {e}\n')

if __name__=='__main__': main()
