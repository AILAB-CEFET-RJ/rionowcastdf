"""Run the complete reference audit, recording execution and exit status."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.corrdiff.builder.contract_common import load_config, read, sha


def save(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    temp.replace(path)


def run(args):
    os.chdir(ROOT)
    config=Path(args.contract_config).resolve()
    c=load_config(config)
    out=Path(args.output_dir).resolve()
    for key in ('output_dir','radar_cache_dir','radar_dir','era5_root','temporal_dir','pairing_dir','reuse_14j_dir'):
        if c.get(key):
            source=Path(c[key]).resolve()
            if out==source or source in out.parents or out in source.parents:
                raise ValueError('Execution output overlaps input: '+key)
    if out.exists():raise FileExistsError('Choose a new execution output: '+str(out))
    dataset=Path(c['output_dir'])
    summary=read(dataset/'summary.json')
    if not summary.get('complete') or not summary.get('zarr_created'):
        raise ValueError('Dataset construction is not complete')
    for key in ('radar_dir','era5_root','radar_cache_dir','pairing_dir','temporal_dir'):
        if not Path(c[key]).is_dir():raise FileNotFoundError(c[key])
    if not (dataset/'staging').is_dir():raise FileNotFoundError('Keep the complete dataset staging directory')
    for split in ('train','validation','test_primary','test_stress_ood'):
        if not (dataset/(split+'.zarr')).is_dir():raise FileNotFoundError(split+'.zarr')
    if args.check_only:
        print(json.dumps({'preflight_passed':True,'selected_hours':summary['selected_hours'],
                          'dataset':str(dataset),'output':str(out),
                          'note':'Path/basic completion checks only; no integrity verdict.'},indent=2))
        return 0
    out.mkdir(parents=True,exist_ok=False)
    cmd=[sys.executable,'-u',str(ROOT/'audit_reference_zarr.py'),
         '--contract-config',str(config),'--output-dir',str(out/'audit'),
         '--all-selected-pngs','--recompute-era5']
    meta={'started_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
          'command':cmd,'python':sys.executable,'python_version':sys.version,
          'config_sha256':sha(config),'auditor_sha256':sha(ROOT/'audit_reference_zarr.py'),
          'complete':False,'integrity_passed':False,'production_ready':False}
    git=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True)
    meta['git_commit']=git.stdout.strip() if git.returncode==0 else None
    freeze=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True)
    (out/'environment.txt').write_text(freeze.stdout if freeze.returncode==0 else freeze.stderr)
    save(out/'execution.json',meta)
    code=1
    try:
        print('Audit report: '+str(out/'audit/summary.json'),flush=True)
        code=subprocess.run(cmd,cwd=ROOT,check=False).returncode
        report=read(out/'audit/summary.json') if (out/'audit/summary.json').exists() else {}
        passed=(code==0 and report.get('complete') is True and report.get('integrity_passed') is True
                and report.get('all_selected_png_paths_checked') is True
                and report.get('all_selected_png_contents_decoded') is True
                and report.get('era5_raw_recomputed') is True)
        meta['integrity_passed']=passed
        if code==0 and not passed:code=2
        return code
    except Exception as exc:
        meta['error']=type(exc).__name__+': '+str(exc)
        raise
    finally:
        meta.update(complete=True,exit_code=code,finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        save(out/'execution.json',meta)
        print(json.dumps(meta,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract-config',default='configs/corrdiff_reference_zarr.json')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args()
    try:sys.exit(run(args))
    except Exception:
        traceback.print_exc()
        sys.exit(1)
