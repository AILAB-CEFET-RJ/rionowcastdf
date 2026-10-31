#!/usr/bin/env python3
"""Rebuild with original ERA5/channel/geometry configuration and QC-only cache."""
import argparse, csv, hashlib, json, os, shutil, subprocess, sys
import pandas as pd
from pathlib import Path
import numpy as np
from radar_qc import fingerprint, sha256

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-report',type=Path,required=True)
    p.add_argument('--qc-cache',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--builder-file',type=Path,default=Path('src/corrdiff/builder/dataset_builder.py'))
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if a.output_dir.exists(): p.error('Output already exists; choose a new version')
    report=json.loads(a.original_report.read_text()); c=report['configuration']
    q=json.loads((a.qc_cache/'qc_config.json').read_text())
    summary=json.loads((a.qc_cache/'qc_summary.json').read_text())
    digest=q.pop('config_sha256')
    if fingerprint(q)!=digest or summary['config_sha256']!=digest: p.error('QC configuration integrity failed')
    if not summary['complete'] or summary['counts'].get('error',0): p.error('QC incomplete or has errors')
    for field in ['lat_range','lon_range']:
        if list(c[field])!=q[field]: p.error(f'Geometry mismatch: {field}')
    if c['radar_resolution_km']!=q['resolution_km']: p.error('Resolution mismatch')
    if sha256(a.qc_cache/'qc_manifest.csv')!=summary['manifest_sha256']: p.error('Manifest integrity failed')
    if pd.to_datetime(c['begin'],utc=True).floor('h')<pd.to_datetime(q['begin'],utc=True) or pd.to_datetime(c['end'],utc=True).floor('h')>pd.to_datetime(q['end'],utc=True):
        p.error('QC cache interval does not cover original builder interval')
    if pd.Timedelta(c['time_frequency'])!=pd.Timedelta(hours=1):p.error('This QC run supports hourly datasets only')
    accepted=set()
    with (a.qc_cache/'qc_manifest.csv').open() as f:
        for row in csv.DictReader(f):
            if not row['status'].startswith('accepted_'): continue
            name=pd.Timestamp(row['timestamp']).strftime('%Y%m%d_%H_%M.npy')
            file=a.qc_cache/name; accepted.add(name)
            if hashlib.sha256(file.read_bytes()).hexdigest()!=row['cache_sha256']:
                p.error(f'Cache modified: {name}')
    if not accepted: p.error('No accepted frames')
    if {x.name for x in a.qc_cache.glob('*.npy')}!=accepted: p.error('Unexpected cache files')
    cmd=[sys.executable,str(a.builder_file)]
    mapping={'begin':'begin','end':'end','era5_root':'era5_root',
        'surface_variables':'surface_variables','pressure_variables':'pressure_variables',
        'pressure_levels_hpa':'pressure_levels','radar_resolution_km':'radar_res',
        'time_frequency':'time_frequency','patch_size':'patch_size','stride':'stride',
        'minimum_input_valid_ratio':'minimum_input_valid_ratio'}
    for key,flag in mapping.items():
        value=c[key]; value=','.join(map(str,value)) if isinstance(value,list) else str(value)
        cmd.extend(['--'+flag,value])
    for key in ['lat_range','lon_range']: cmd.extend(['--'+key,*map(str,c[key])])
    # Retain every spatial slot; missing targets remain masked. This is a diagnostic dataset.
    cmd.extend(['--radar_cache',str(a.qc_cache),'--radar_cache_policy','never',
                '--output_dir',str(a.output_dir),'--minimum_radar_valid_ratio','0'])
    import shlex
    print(shlex.join(cmd),flush=True)
    if not a.execute: return
    env=os.environ.copy()
    repo_root=a.builder_file.resolve().parents[3]
    env['PYTHONPATH']=str(repo_root)+os.pathsep+env.get('PYTHONPATH','')
    subprocess.run(cmd,check=True,env=env)
    for name in ['qc_config.json','qc_summary.json','qc_manifest.csv']:
        shutil.copy2(a.qc_cache/name,a.output_dir/name)
    provenance=dict(original_report_sha256=hashlib.sha256(a.original_report.read_bytes()).hexdigest(),
        builder_sha256=hashlib.sha256(a.builder_file.read_bytes()).hexdigest(),
        qc_config_sha256=digest, command=cmd, production_ready=False,
        required='Masked metrics and observation-validity review before training')
    (a.output_dir/'qc_dataset_provenance.json').write_text(json.dumps(provenance,indent=2))
if __name__=='__main__': main()
