#!/usr/bin/env python3
"""Versioned producer policy on immutable 14j objects + completed 14k pairing.
No raw PNG reads, ERA5 re-inventory, input mutation or temporal aggregation.
"""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
import pandas as pd
from pair_contract import CHANNELS, digest, dump

VERSION = '14n-producer-policy-v1'
PALETTE = [50,45,40,35,30,25,20]
NORMAL = {'transparent_ambiguous', 'palette_in_png_not_sampled_in_corrected_grid',
          'palette_sampled_observation_validity_unconfirmed'}
MAP_COLUMNS = ['source_sha256','decision','evidence','reviewer','reviewed_at']


def inside(root, rel):
    p=(root/str(rel)).resolve()
    if root not in p.parents: raise ValueError('Path escapes cache: '+str(rel))
    return p


def boolean(series):
    if series.isna().any() or not series.isin([True,False,0,1]).all():
        raise ValueError('Invalid boolean column: '+series.name)
    return series.astype(bool)


def load_reviews(path):
    if path is None: return {}
    df=pd.read_csv(path,dtype=str,keep_default_na=False)
    if list(df.columns)!=MAP_COLUMNS: raise ValueError('Map review columns must be '+','.join(MAP_COLUMNS))
    if df.source_sha256.duplicated().any(): raise ValueError('Duplicate map review SHA256')
    result={}
    for r in df.to_dict('records'):
        if len(r['source_sha256'])!=64 or any(c not in '0123456789abcdef' for c in r['source_sha256']):
            raise ValueError('Invalid source SHA256 in map review')
        if r['decision'] not in ('pending','reject','accept_no_echo'):
            raise ValueError('Unknown map review decision')
        if r['decision']=='accept_no_echo' and not all(r[k].strip() for k in ('evidence','reviewer','reviewed_at')):
            raise ValueError('Map acceptance needs evidence, reviewer and date')
        result[r['source_sha256']]=r
    return result


def support_contract(cache, args):
    with np.load(cache/'geometry.npz',allow_pickle=False) as z:
        lat=z['lat'].copy();lon=z['lon'].copy();geom=z['geometric_mask'].copy()
    if lat.shape!=(70,84) or lon.shape!=lat.shape or geom.shape!=lat.shape or geom.dtype!=bool:
        raise ValueError('Expected 14j boolean geometry 70x84')
    if bool(args.coverage_mask)!=bool(args.coverage_metadata):
        raise ValueError('Provide both --coverage-mask and --coverage-metadata')
    if args.coverage_mask:
        with np.load(args.coverage_mask,allow_pickle=False) as z:
            support=z['observed_region_mask'].copy()
            if support.dtype!=bool or support.shape!=geom.shape:
                raise ValueError('Coverage mask must be boolean 70x84')
            if not np.array_equal(z['lat'],lat) or not np.array_equal(z['lon'],lon):
                raise ValueError('Coverage mask coordinates differ from 14j')
        if (support & ~geom).any() or not support.any(): raise ValueError('Coverage must be a nonempty subset of geometry')
        meta=json.loads(Path(args.coverage_metadata).read_text())
        for k in ('evidence','reviewer','reviewed_at'):
            if not isinstance(meta.get(k),str) or not meta[k].strip(): raise ValueError('Missing coverage evidence: '+k)
        if meta.get('semantics')!='effective_observed_region':
            raise ValueError('Coverage semantics must be effective_observed_region')
        # A static mask is a user-reviewed domain, not a check of per-scan quality.
        state='reviewed_static_region'
    else:
        support=geom.copy();state='provisional_geometric_region'
        meta={'evidence':'14j geometric extent only; effective observational coverage not certified',
              'semantics':'geometric_region_only'}
    return lat,lon,geom,support,state,meta


def interpret(classes, alpha, geom, support, status, palette_pixels, map_approved=False):
    """0=unknown, 1=no displayed echo, 2=displayed echo. Never replace unknown with dry."""
    if any(a.shape!=geom.shape for a in (classes,alpha,support)):
        raise ValueError('Grid shape mismatch')
    if not np.isin(classes,[0,1,2,3,4,5,6,7,253,254,255]).all(): raise ValueError('Unknown 14j class')
    if ((classes==0)&geom&(alpha!=0)).any(): raise ValueError('Transparent class/alpha mismatch')
    if (((classes>=1)&(classes<=7))&geom&(alpha!=255)).any(): raise ValueError('Palette class/alpha mismatch')
    labels=np.zeros(geom.shape,np.uint8)
    if status=='map_screen_review':
        if not map_approved: return None
        if palette_pixels!=0: raise ValueError('Cannot accept map as no-echo: exact palette pixels exist')
        labels[support]=1
    elif status in NORMAL:
        if status=='transparent_ambiguous' and (palette_pixels!=0 or (classes[geom]!=0).any()):
            raise ValueError('Transparent status disagrees with cache classes')
        labels[support & (classes==0) & (alpha==0)]=1
        labels[support & (classes>=1)&(classes<=7)&(alpha==255)]=2
    else:
        return None
    valid=labels!=0;echo=labels==2
    # Only the displayed classes have a dBZ value. No-echo is NOT a measured 0 dBZ.
    displayed=np.full(geom.shape,np.nan,np.float32)
    if status!='map_screen_review': displayed[echo]=np.asarray(PALETTE,dtype=np.float32)[classes[echo]-1]
    # Explicit experimental code for regression: 0 denotes no displayed echo.
    code=np.full(geom.shape,np.nan,np.float32);code[labels==1]=0;code[echo]=displayed[echo]
    return dict(observation_class=labels,observation_mask=valid,echo_mask=echo,
                no_echo_mask=labels==1,displayed_dbz=displayed,target_code=code)


def process_object(row, cache, out, geom, support, reviews):
    status=row.status
    review=reviews.get(row.source_sha256,{})
    if status not in NORMAL and not (status=='map_screen_review' and review.get('decision')=='accept_no_echo'):
        return dict(policy_status=('map_rejected' if review.get('decision')=='reject' else 'map_review_pending') if status=='map_screen_review' else 'excluded_'+status,
                    target_path='',target_sha256='',observation_pixels=0,no_echo_pixels=0,echo_pixels=0,
                    unknown_pixels=int(support.sum()),policy_error='')
    path=inside(cache,row.decoded_path)
    if digest(path)!=row.decoded_sha256: raise ValueError('Decoded checksum mismatch: '+str(path))
    with np.load(path,allow_pickle=False) as z:
        if not np.array_equal(z['geometric_mask'],geom): raise ValueError('Decoded geometry mismatch')
        meta=json.loads(str(z['metadata_json']))
        if meta['status']!=status or meta['source_size']!=[656,654] or meta['source_frames']!=1:
            raise ValueError('Decoded metadata disagrees with manifest')
        native_count=int(meta['exact_palette_pixels'])
        arrays=interpret(z['grid_class'],z['grid_alpha'],geom,support,status,native_count,
                         review.get('decision')=='accept_no_echo')
        echo=arrays['echo_mask']
        if echo.any() and not np.array_equal(z['grid_dbz_palette'][echo],arrays['displayed_dbz'][echo]):
            raise ValueError('Decoded palette/value disagreement')
    n=int(arrays['observation_mask'].sum());e=int(echo.sum());d=int(arrays['no_echo_mask'].sum())
    if status=='map_screen_review': label='reviewed_map_no_echo'
    elif status=='transparent_ambiguous': label='valid_transparent_no_echo'
    elif e: label='valid_echo_and_no_echo'
    else: label='valid_no_echo_on_sampled_grid'
    if not n:label='no_supported_pixels'
    rel=f'targets/{row.source_sha256[:2]}/{row.source_sha256}.npz'
    p=out/rel;p.parent.mkdir(parents=True,exist_ok=True)
    arrays['source_sha256']=np.array(row.source_sha256)
    arrays['target_encoding']=np.array('experimental: 0=no displayed echo; 20..50=displayed dBZ class; NaN=unknown')
    tmp=p.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    tmp.replace(p)
    return dict(policy_status=label,target_path=rel,target_sha256=digest(p),observation_pixels=n,
                no_echo_pixels=d,echo_pixels=e,unknown_pixels=int(support.sum())-n,policy_error='')


def run(args):
    cache=Path(args.radar_cache_dir).resolve();pair=Path(args.pairing_dir).resolve();out=Path(args.output_dir).resolve()
    for p in (cache,pair):
        if out==p or out in p.parents or p in out.parents:raise ValueError('Inputs/output trees must be separate')
    base=json.loads((pair/'run_config.json').read_text());summary=json.loads((pair/'summary.json').read_text())
    if summary.get('version')!='14k-pairing-v1' or not summary.get('complete') or summary.get('provisional',True):
        raise ValueError('A completed, nonprovisional 14k pairing is required')
    if not summary.get('radar_snapshot_complete') or summary.get('era5_error_files',1):
        raise ValueError('Incomplete radar snapshot or ERA5 errors')
    checks={cache/'config.json':base['radar_config_sha256'],cache/'geometry.npz':base['geometry_sha256'],
            pair/'radar_manifest_snapshot.sqlite':base['radar_snapshot_sha256'],
            pair/'era5_inventory_snapshot.sqlite':base['era5_snapshot_sha256']}
    for p,expected in checks.items():
        if digest(p)!=expected:raise ValueError('Input fingerprint mismatch: '+str(p))
    cfg=json.loads((cache/'config.json').read_text())
    if cfg.get('version')!='14j-intermediate-cache-v1' or cfg.get('cadence_minutes')!=60 or cfg['reference_contract']['palette_dbz']!=PALETTE:
        raise ValueError('Unsupported cache contract')
    lat,lon,geom,support,coverage_state,coverage_meta=support_contract(cache,args)
    reviews=load_reviews(args.map_review)
    start_local=pd.Timestamp(args.archive_start_local)
    if start_local.tzinfo is not None or start_local!=start_local.normalize():raise ValueError('Use a local DATE, not a UTC timestamp')
    boundary=start_local.tz_localize('America/Sao_Paulo',ambiguous='raise',nonexistent='raise').tz_convert('UTC')
    df=pd.read_parquet(pair/'paired_manifest.parquet')
    df['timestamp_utc']=pd.to_datetime(df.timestamp_utc,utc=True)
    if df.timestamp_utc.duplicated().any():raise ValueError('Duplicate UTC timestamps')
    df=df.sort_values('timestamp_utc').reset_index(drop=True)
    # Verify radar lineage against the hashed immutable snapshot, not only CSV/Parquet labels.
    with sqlite3.connect((pair/'radar_manifest_snapshot.sqlite').as_uri()+'?mode=ro',uri=True) as db:
        snap=pd.read_sql_query('SELECT f.timestamp_utc,f.source_sha256,f.status,f.local_filename_ambiguous,s.decoded_path,s.decoded_sha256 FROM frames f LEFT JOIN sources s ON f.source_sha256=s.sha256',db)
    snap['timestamp_utc']=pd.to_datetime(snap.timestamp_utc,utc=True)
    checked=df.merge(snap,on='timestamp_utc',suffixes=('','_snapshot'),validate='one_to_one',how='left')
    for col in ('source_sha256','status','local_filename_ambiguous','decoded_path','decoded_sha256'):
        a=checked[col];b=checked[col+'_snapshot']
        if not (a.fillna('')==b.fillna('')).all():raise ValueError('Pairing differs from radar snapshot: '+col)
    for col in ('era5_technical_candidate','local_filename_ambiguous','locked','paired_palette_candidate'):
        df[col]=boolean(df[col])
    if not df['era5_technical_candidate'].equals(df[CHANNELS].eq('finite_rectangular_cover').all(axis=1)):
        raise ValueError('ERA5 candidate flag disagrees with channels')
    # Independently check existing split contract (no split redesign).
    times=pd.DatetimeIndex(df.timestamp_utc);nominal=np.select([times.year<=2020,times.year<=2022,times.year==2023],
        ['train','validation','test_primary'],default='test_stress_ood')
    splits=nominal.copy()
    for b in ('2021-01-01','2023-01-01','2024-01-01'):
        t=pd.Timestamp(b,tz='UTC');delta=pd.Timedelta(hours=int(base['args']['purge_hours']))
        splits[(times>=t-delta)&(times<t+delta)]='excluded_purge'
    if not np.array_equal(splits,df.split.to_numpy()):raise ValueError('Split differs from 14k contract')
    if not df.locked.equals(df.split.isin(['test_primary','test_stress_ood','excluded_purge'])):
        raise ValueError('Locked split flag mismatch')
    df['in_archive_period']=df.timestamp_utc.ge(boundary)
    if df.loc[~df.in_archive_period,'source_sha256'].notna().any():
        raise ValueError('PNG objects exist before supplied archive start. Review start date; no silent exclusion.')
    manifest_sha=digest(pair/'paired_manifest.parquet')
    config=dict(version=VERSION,cache_dir=str(cache),pairing_dir=str(pair),
        runtime_versions={'python':sys.version.split()[0],'numpy':np.__version__,'pandas':pd.__version__},
        pairing_run_config_sha256=digest(pair/'run_config.json'),paired_manifest_sha256=manifest_sha,
        input_checks={str(k):v for k,v in checks.items()},
        archive_start_local=start_local.date().isoformat(),archive_boundary_utc=boundary.isoformat(),
        archive_first_minute_known=False,coverage_state=coverage_state,coverage_metadata=coverage_meta,
        coverage_sha256=digest(Path(args.coverage_mask)) if args.coverage_mask else None,
        map_reviews=reviews,producer_evidence=json.loads(Path(__file__).with_name('producer_evidence.json').read_text()),
        code_sha256={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
        aggregation='none; exact hourly timestamp; unchanged nearest-neighbor grid',
        target_encoding='0 category no displayed echo; palette dBZ classes; NaN unknown; not measured zero dBZ')
    existed=out.exists()
    if existed and not args.resume:raise ValueError('Use a NEW output directory, or --resume for identical inputs')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'policy_config.json').exists():
            if json.loads((out/'policy_config.json').read_text())!=config:raise ValueError('Resume contract changed; use new output')
        elif existed and any(p.name!='.writer.lock' for p in out.iterdir()):raise ValueError('Output has no policy config')
        else:dump(out/'policy_config.json',config)
        dump(out/'summary.json',dict(version=VERSION,complete=False,production_ready=False))
        np.savez_compressed(out/'coverage.npz',lat=lat,lon=lon,geometric_mask=geom,observed_region_mask=support)
        db=sqlite3.connect(out/'policy_objects.sqlite')
        db.execute('CREATE TABLE IF NOT EXISTS objects (sha TEXT PRIMARY KEY, result TEXT NOT NULL)')
        unique=df[df.in_archive_period].dropna(subset=['source_sha256']).drop_duplicates('source_sha256')
        # Same source bytes must never be assigned contradictory decode/status paths.
        for col in ('status','decoded_path','decoded_sha256'):
            if (df.dropna(subset=['source_sha256']).groupby('source_sha256')[col].nunique()>1).any():raise ValueError('Inconsistent source metadata')
        results={}
        try:
            for i,row in enumerate(unique.itertuples(index=False),1):
                saved=db.execute('SELECT result FROM objects WHERE sha=?',(row.source_sha256,)).fetchone()
                result=json.loads(saved[0]) if saved else None
                if result and result['target_path'] and digest(inside(out,result['target_path']))!=result['target_sha256']:
                    raise ValueError('Derived target corrupted; use new output')
                if not result or result['policy_error']:
                    try:result=process_object(row,cache,out,geom,support,reviews)
                    except (OSError,ValueError,KeyError) as e:
                        result=dict(policy_status='policy_object_error',target_path='',target_sha256='',
                            observation_pixels=0,no_echo_pixels=0,echo_pixels=0,unknown_pixels=int(support.sum()),policy_error=f'{type(e).__name__}: {e}')
                    db.execute('INSERT OR REPLACE INTO objects VALUES (?,?)',(row.source_sha256,json.dumps(result)))
                results[row.source_sha256]=result
                if i%250==0 or i==len(unique):
                    db.commit();print(f'Policy objects {i}/{len(unique)}',flush=True)
                    dump(out/'progress.json',dict(objects_processed=i,unique_objects=len(unique),complete=False))
        finally:db.commit();db.close()
        rows=[]
        for row in df.itertuples(index=False):
            r=results.get(row.source_sha256)
            if not row.in_archive_period:
                r=dict(policy_status='outside_archive_period',target_path='',target_sha256='',observation_pixels=0,no_echo_pixels=0,echo_pixels=0,unknown_pixels=0,policy_error='')
            elif r is None:
                r=dict(policy_status='excluded_'+row.status,target_path='',target_sha256='',observation_pixels=0,no_echo_pixels=0,echo_pixels=0,unknown_pixels=int(support.sum()),policy_error='')
            rows.append(r)
        result=pd.concat([df,pd.DataFrame(rows)],axis=1)
        result['legacy_paired_palette_candidate']=result.paired_palette_candidate
        result['legacy_train_smoke_candidate']=result.train_smoke_candidate
        # Original candidate columns remain provenance; new policy uses distinct names.
        result['policy_pair_candidate']=(result.in_archive_period & result.era5_technical_candidate &
            ~result.local_filename_ambiguous & result.observation_pixels.gt(0) & result.policy_error.eq(''))
        result['policy_train_candidate']=result.policy_pair_candidate & result.split.eq('train') & ~result.locked
        result['coverage_state']=coverage_state;result['production_ready']=False
        result.to_parquet(out/'policy_manifest.parquet',index=False)
        result.to_csv(out/'policy_manifest.csv.gz',index=False,compression='gzip')
        current=result[result.in_archive_period]
        report=current.groupby('split').agg(hours=('timestamp_utc','size'),era5_complete_12=('era5_complete_12','sum'),
            legacy_palette_candidates=('legacy_paired_palette_candidate','sum'),policy_pair_candidates=('policy_pair_candidate','sum'),
            ambiguous_hours=('local_filename_ambiguous','sum')).reset_index()
        report['policy_candidate_percent']=100*report.policy_pair_candidates/report.hours
        report.to_csv(out/'split_coverage.csv',index=False)
        counts=current.groupby(['split','policy_status']).size().rename('hours').reset_index()
        counts['denominator_hours']=counts.split.map(current.groupby('split').size())
        counts['percent']=100*counts.hours/counts.denominator_hours
        counts.to_csv(out/'split_policy_status.csv',index=False)
        current.assign(year_utc=current.timestamp_utc.dt.year).groupby(['year_utc','policy_status']).size().rename('hours').reset_index().to_csv(out/'annual_policy_status.csv',index=False)
        result.groupby(['in_archive_period','status']).size().rename('hours').reset_index().to_csv(out/'period_reconciliation.csv',index=False)
        current[current.policy_status.eq('policy_object_error')].to_csv(out/'errors.csv',index=False)
        map_rows=[]
        for sha,g in current[current.status.eq('map_screen_review')].groupby('source_sha256'):
            review=reviews.get(sha,{})
            map_rows.append(dict(source_sha256=sha,hours=len(g),first_utc=g.timestamp_utc.min().isoformat(),
                example_source=g.source_path.iloc[0],decision=review.get('decision','pending'),evidence=review.get('evidence',''),
                reviewer=review.get('reviewer',''),reviewed_at=review.get('reviewed_at','')))
        pd.DataFrame(map_rows,columns=['source_sha256','hours','first_utc','example_source','decision','evidence','reviewer','reviewed_at']).to_csv(out/'map_review_queue.csv',index=False)
        pd.DataFrame([{k:r.get(k,'') for k in MAP_COLUMNS} for r in map_rows],columns=MAP_COLUMNS).to_csv(out/'map_decisions_template.csv',index=False)
        used=set(current.source_sha256.dropna())
        final=dict(version=VERSION,complete=True,production_ready=False,no_echo_rule_documented=True,
            coverage_state=coverage_state,spatial_validity_confirmed=coverage_state=='reviewed_static_region',
            target_encoding_finalized=False,corrdiff_native_tested=False,archive_boundary_utc=boundary.isoformat(),
            original_hours=len(result),outside_archive_period_hours=int((~result.in_archive_period).sum()),
            in_archive_hours=len(current),unique_objects_processed=len(results),
            policy_status_counts={k:int(v) for k,v in current.policy_status.value_counts().items()},
            policy_pair_candidate_hours=int(current.policy_pair_candidate.sum()),
            policy_train_candidate_hours=int(current.policy_train_candidate.sum()),
            paired_transparent_hours=int((current.policy_pair_candidate & current.status.eq('transparent_ambiguous')).sum()),
            object_error_hours=int(current.policy_status.eq('policy_object_error').sum()),
            unused_map_review_hashes=sorted(set(reviews)-used),
            artifact_sha256={name:digest(out/name) for name in ('policy_manifest.parquet','coverage.npz','policy_config.json')},
            temporal_aggregation='none',raw_pngs_read=False,era5_values_read=False,
            note='Producer semantics applied; geometric-only default is a provisional spatial assumption. Zero is a no-echo code, not measured dBZ. Original 14j/14k unchanged.')
        dump(out/'summary.json',final)
        return final


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--radar-cache-dir',required=True);p.add_argument('--pairing-dir',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--archive-start-local',default='2011-01-17')
    p.add_argument('--map-review',help='Reviewed SHA256 allowlist CSV; no automatic cartography release')
    p.add_argument('--coverage-mask',help='NPZ: observed_region_mask bool, lat, lon; same grid as 14j')
    p.add_argument('--coverage-metadata',help='JSON: semantics=effective_observed_region, evidence, reviewer, reviewed_at')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    try:
        result=run(args);print(json.dumps(result,indent=2));return 2 if result['object_error_hours'] else 0
    except KeyboardInterrupt:print('Interrupted. Resume with identical arguments and --resume.',file=sys.stderr);return 130
    except Exception as e:print(f'{type(e).__name__}: {e}',file=sys.stderr);return 1

if __name__=='__main__':sys.exit(main())
