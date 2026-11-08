"""Selectable, resumable, sequential reference-dataset phase runner."""
import argparse
import contextlib
from datetime import datetime,timezone
import fcntl
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from .common import VERSION,NAMES,DEPS,Context,sha,dump
from .structure import phase0,phase15
from . import science
from .training import phase16,phase17,phase18,StageBlocked

FUNCTIONS={0:phase0,15:phase15,16:phase16,17:phase17,18:phase18}
FUNCTIONS.update({i:getattr(science,f'phase{i}') for i in range(1,15) if i not in (5,6)})
FUNCTIONS[5]=lambda c,o:science.calendar_phase(c,o,5)
FUNCTIONS[6]=lambda c,o:science.calendar_phase(c,o,6)


def selection(text):
    if text=='all':return list(range(19))
    result=set()
    for item in text.split(','):
        if '-' in item:
            first,last=map(int,item.split('-'))
            if first>last:raise ValueError('Reversed phase range')
            result.update(range(first,last+1))
        else:result.add(int(item))
    if not result or not result<=set(NAMES):raise ValueError('Phases must be between 0 and 18')
    return sorted(result)


def plan(phases,with_dependencies=True):
    order=[]
    def visit(p):
        if p in order:return
        for dep in DEPS[p]:
            if with_dependencies or dep in phases:visit(dep)
        order.append(p)
    for p in phases:visit(p)
    return order


def load_config(path):
    path=Path(path).resolve();c=json.loads(path.read_text())
    root=(path.parent/c.get('project_root','..')).resolve()
    for key in ('dataset_dir','output_dir','integrity_summary','reader_summary','ablation_config_dir'):
        c[key]=str((root/c[key]).resolve())
    native=c.get('native',{})
    if native.get('corrdiff_dir'):native['corrdiff_dir']=str((root/native['corrdiff_dir']).resolve())
    nonnegative=('phase0_max_hours_per_split','diagnostic_hours_per_split','baseline_train_hours','baseline_eval_hours')
    positive=('sample_pixels_per_hour','regime_clusters','regime_fit_max_hours')
    for key in nonnegative+positive:
        if not isinstance(c[key],int) or c[key]<(0 if key in nonnegative else 1):raise ValueError('Invalid config: '+key)
    if not c['lags_hours'] or any(not isinstance(v,int) or v<=0 for v in c['lags_hours']):raise ValueError('Invalid lags')
    if c['ridge_alpha']<=0:raise ValueError('ridge_alpha must be positive')
    return c


def preflight(c):
    dataset=Path(c['dataset_dir']);out=Path(c['output_dir'])
    if out==dataset or dataset in out.parents or out in dataset.parents:raise ValueError('Output and dataset directories must be separate')
    audit=json.loads(Path(c['integrity_summary']).read_text());reader=json.loads(Path(c['reader_summary']).read_text())
    flags=('complete','integrity_passed','all_patch_tensors_compared_to_staging','all_targets_compared_to_cache',
           'all_selected_png_contents_decoded','all_selected_png_paths_checked','era5_raw_recomputed','normalization_recomputed')
    if any(audit.get(k) is not True for k in flags):raise ValueError('Full source integrity audit required')
    if not reader.get('complete') or not reader.get('reader_check_passed'):raise ValueError('Real reader check required')
    summary=json.loads((dataset/'summary.json').read_text())
    if audit.get('selected_hours')!=summary['selected_hours']:raise ValueError('Audit dataset hour count mismatch')
    for split,stats in summary['splits'].items():
        if audit['splits'][split]['patches']!=stats['patches'] or reader['splits'][split]['patches']!=stats['patches']:raise ValueError('Evidence split count mismatch')
    for name in ('contract.json','normalization.json','selected_manifest.parquet','era5_identity.json'):
        if sha(dataset/name)!=summary['artifact_sha256'][name]:raise ValueError('Dataset artifact changed: '+name)
    paths=[Path(__file__).with_name(f) for f in ('common.py','runner.py','science.py','structure.py','training.py')]
    paths.append(Path(__file__).parents[1]/'reference_reader.py')
    fingerprint=dict(config=c,source_code={p.name:sha(p) for p in paths},dataset_summary=sha(dataset/'summary.json'),
        metadata=summary['artifact_sha256'],integrity_summary=sha(c['integrity_summary']),reader_summary=sha(c['reader_summary']),
        ablation_configs={p.name:sha(p) for p in Path(c['ablation_config_dir']).glob('phase17_*.json')},
        note='Audit evidence tied by metadata/counts, not an independent rescan of immutable Zarr chunks.')
    return fingerprint


class Tee:
    def __init__(self,*streams):self.streams=streams
    def write(self,text):
        for stream in self.streams:stream.write(text)
        return len(text)
    def flush(self):
        for stream in self.streams:stream.flush()


def verified(directory):
    p=directory/'summary.json'
    if not p.exists():return False
    record=json.loads(p.read_text())
    if record.get('status')!='completed':return False
    for name,digest in record.get('artifact_sha256',{}).items():
        if not (directory/name).is_file() or sha(directory/name)!=digest:return False
    return True


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/reference_phases.json')
    p.add_argument('--phases',default='0,15',help='0,15 or 1-4 or all')
    p.add_argument('--list',action='store_true');p.add_argument('--plan',action='store_true')
    p.add_argument('--resume',action='store_true');p.add_argument('--no-deps',action='store_true')
    p.add_argument('--output-dir');p.add_argument('--retry-failed',action='store_true')
    a=p.parse_args(argv)
    if a.list:
        for i in NAMES:print(f'{i:02d} {NAMES[i]} | dependencies={DEPS[i]}'+(' | external native integration, gated' if i==18 else ''))
        return 0
    c=load_config(a.config)
    if a.output_dir:c['output_dir']=str(Path(a.output_dir).resolve())
    phases=plan(selection(a.phases),not a.no_deps)
    print('Execution plan: '+' -> '.join(f'{i:02d}' for i in phases),flush=True)
    if a.plan:return 0
    fp=preflight(c);out=Path(c['output_dir'])
    if out.exists() and not a.resume:raise FileExistsError('Output exists; use --resume or a new --output-dir')
    if a.retry_failed and not a.resume:raise ValueError('--retry-failed requires --resume')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        snapshot=out/'run_config.json'
        if snapshot.exists():
            if json.loads(snapshot.read_text())!=fp:raise ValueError('Config/code/evidence changed; create a new output directory')
        else:dump(snapshot,fp)
        ctx=Context(c,out)
        results=[];started=datetime.now(timezone.utc).isoformat();exit_code=0
        for phase in phases:
            directory=ctx.directory(phase)
            for dep in DEPS[phase]:
                if not verified(ctx.directory(dep)):raise RuntimeError(f'Phase {phase} requires successful phase {dep}')
            if verified(directory):
                print(f'phase {phase:02d}: reused verified outputs',flush=True);results.append(dict(phase=phase,status='reused'));continue
            if directory.exists():
                if not a.retry_failed:raise RuntimeError(f'Incomplete phase {phase}; use --resume --retry-failed after reviewing log')
                archived=out/'failed_attempts';archived.mkdir(exist_ok=True)
                directory.rename(archived/(directory.name+'_'+str(time.time_ns())))
            directory.mkdir();begin=time.monotonic()
            report=dict(version=VERSION,phase=phase,name=NAMES[phase],status='running',complete=False,production_ready=False)
            dump(directory/'summary.json',report)
            try:
                with (directory/'execution.log').open('w') as log,contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(Tee(sys.stderr,log)):
                    payload=FUNCTIONS[phase](ctx,directory)
                report.update(payload,status='completed',complete=True)
            except BaseException as exc:
                report.update(status='blocked' if isinstance(exc,StageBlocked) else 'failed',error=f'{type(exc).__name__}: {exc}')
                exit_code=3 if isinstance(exc,StageBlocked) else 1
                import traceback
                with (directory/'execution.log').open('a') as log:traceback.print_exc(file=log)
                print(report['error'],file=sys.stderr,flush=True)
            report['elapsed_seconds']=round(time.monotonic()-begin,3)
            report['artifact_sha256']={str(f.relative_to(directory)):sha(f) for f in directory.rglob('*') if f.is_file() and f.name!='summary.json'}
            dump(directory/'summary.json',report);results.append(dict(phase=phase,status=report['status']))
            dump(out/'run_summary.json',dict(version=VERSION,started_utc=started,updated_utc=datetime.now(timezone.utc).isoformat(),
                selected_phases=phases,results=results,complete=exit_code==0 and len(results)==len(phases),
                exit_code=exit_code,production_ready=False,phase0_sampling=c['phase0_max_hours_per_split'],python=sys.version,platform=platform.platform()))
            if exit_code:break
        # Also records a resume where all requested phases were reused.
        dump(out/'run_summary.json',dict(version=VERSION,selected_phases=phases,results=results,complete=exit_code==0 and len(results)==len(phases),exit_code=exit_code,production_ready=False,
            started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,platform=platform.platform(),
            phase0_max_hours_per_split=c['phase0_max_hours_per_split'],diagnostic_hours_per_split=c['diagnostic_hours_per_split']))
        return exit_code


if __name__=='__main__':raise SystemExit(main())
