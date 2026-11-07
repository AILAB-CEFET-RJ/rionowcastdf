#!/usr/bin/env python3
"""Reindex a completed 14o snapshot with stride 8, without reading ERA5/PNGs.

The new directory references the old immutable monthly shards by symlink.
Keep the original dataset directory; the new snapshot is not standalone.
"""
import argparse
from pathlib import Path
import shutil
import pandas as pd
from prepare import checked_json,finish,HERE
from pair_contract import digest,dump

def run(a):
    source=Path(a.prepared_dir).resolve();out=Path(a.output_dir).resolve()
    if out.exists() or source in out.parents or out in source.parents:raise ValueError('Use a new independent directory')
    summary=checked_json(source/'summary.json')
    if summary.get('version')!='14o-prepared-v1' or not summary.get('complete'):raise ValueError('Wait for the source build to complete')
    for n,h in summary['artifact_sha256'].items():
        if digest(source/n)!=h:raise ValueError('Source artifact changed: '+n)
    contract=checked_json(source/'contract.json');new=checked_json(HERE/'contract.json')
    for key in ('version','patch_stride','patch_edges'):contract[key]=new[key]
    contract['reindexed_from']=str(source);contract['source_summary_sha256']=digest(source/'summary.json')
    contract['code_sha256']={p.name:digest(p) for p in sorted(HERE.glob('*.py'))}
    hours=pd.read_parquet(source/'hours.parquet').rename(columns={'shard':'month_key'})
    hours['timestamp_utc']=pd.to_datetime(hours.timestamp_utc,utc=True)
    if hours.timestamp_utc.duplicated().any():raise ValueError('Duplicate timestamps')
    for _,group in hours.groupby('month_key',sort=True):
        if group.hour_index.tolist()!=list(range(len(group))):raise ValueError('Unexpected shard row order')
    out.mkdir(parents=True)
    dump(out/'contract.json',contract);dump(out/'summary.json',{'complete':False,'production_ready':False})
    shutil.copyfile(source/'coverage.npz',out/'coverage.npz')
    (out/'shards').symlink_to(source/'shards',target_is_directory=True)
    hours.to_parquet(out/'selected_hours.parquet',index=False)
    # Recompute the same train-only statistics from unchanged prepared fields.
    finish(out,hours,contract)
    if checked_json(out/'normalization.json')!=checked_json(source/'normalization.json'):
        dump(out/'summary.json',{'complete':False,'production_ready':False,'error':'Normalization changed during reindex'})
        raise ValueError('Normalization changed unexpectedly')
    print((out/'summary.json').read_text())

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared-dir',required=True);p.add_argument('--output-dir',required=True)
    run(p.parse_args())
