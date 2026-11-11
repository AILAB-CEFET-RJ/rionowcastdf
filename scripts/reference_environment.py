"""Audit a machine, build a frozen wheel bundle, install it into a NEW venv."""
import argparse
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import zipfile
from email.parser import BytesParser

EXPECTED_COMMIT = 'c15bd23eb56216c32bd3e30180593e5685be9bc2'


def run(argv, **kwargs):
    return subprocess.run([str(x) for x in argv], check=True, **kwargs)


def output(argv):
    return run(argv, capture_output=True, text=True).stdout.strip()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def host():
    return dict(python=list(sys.version_info[:2]), system=platform.system(),
                machine=platform.machine(), platform=sysconfig.get_platform())


def checkout_info(path, expected):
    path = Path(path).resolve()
    commit = output(['git','-C',path,'rev-parse','HEAD'])
    dirty = output(['git','-C',path,'status','--porcelain','--untracked-files=all'])
    if commit != expected: raise ValueError(f'Checkout commit {commit} differs from expected {expected}')
    return dict(path=str(path), commit=commit, dirty=bool(dirty), status=dirty)


def doctor(args):
    report = dict(schema='reference-environment-v1', host=host(), python=sys.version,
                  executable=sys.executable, checks={}, native_training_tested=False, ready_for_training=False,
                  dataset_modified=False, test_splits_accessed=False)
    checks = report['checks']
    try:
        report['checkout'] = checkout_info(args.physicsnemo_dir,args.expected_commit)
        checks['checkout_clean'] = not report['checkout']['dirty']
    except Exception as e:
        checks['checkout_clean']=False; report['checkout_error']=str(e)
    try:
        report['pip_check']=output([sys.executable,'-m','pip','check']); checks['dependencies']=True
    except subprocess.CalledProcessError as e:
        checks['dependencies']=False; report['pip_check']=e.stdout or str(e)
    try:
        import torch
        import physicsnemo
        from physicsnemo.models.diffusion_unets import CorrDiffRegressionUNet
        from physicsnemo.diffusion.preconditioners import EDMPrecondSuperResolution
        report['packages']={n:md.version(n) for n in ('torch','numpy','zarr','nvidia-physicsnemo')}
        report['physicsnemo_import_path']=str(Path(physicsnemo.__file__).resolve())
        report['torch_cuda']=torch.version.cuda
        checks['native_imports']=True
        report['gpus']=[]
        if not args.cpu_only:
            ids=[int(v) for v in args.gpus.split(',')]
            if len(set(ids)) != len(ids) or any(i < 0 for i in ids): raise ValueError('Invalid GPU IDs')
            if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
            for i in ids:
                with torch.cuda.device(i):
                    model=torch.nn.Conv2d(2,1,3,padding=1).to(f'cuda:{i}')
                    x=torch.randn(2,2,32,32,device=f'cuda:{i}',dtype=torch.float32)
                    loss=model(x).square().mean(); loss.backward(); torch.cuda.synchronize(i)
                    if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise RuntimeError(f'Non-finite GPU result: {i}')
                    prop=torch.cuda.get_device_properties(i)
                    report['gpus'].append(dict(logical_id=i,name=prop.name,memory_bytes=prop.total_memory,
                                               capability=list(torch.cuda.get_device_capability(i)),fp32_backward=True))
            checks['gpu_fp32']=True
        report['cuda_visible_devices']=os.environ.get('CUDA_VISIBLE_DEVICES')
    except Exception as e:
        checks['runtime']=False; report['runtime_error']=str(e)
    repo=Path(args.project_dir).resolve()
    report['disk_free_bytes']=shutil.disk_usage(repo).free
    report['cpu_count']=os.cpu_count()
    report['libc']=platform.libc_ver()
    try:
        sys.path.insert(0,str(repo))
        from adapter.reference_corrdiff_dataset import ReferenceCorrDiffDataset
        dataset=Path(args.dataset_dir)
        if not dataset.is_absolute(): dataset=repo/dataset
        report['dataset']=[ReferenceCorrDiffDataset(dataset,split).info() for split in ('train','validation')]
        checks['dataset_contract']=True
    except Exception as e:
        checks['dataset_contract']=False; report['dataset_error']=str(e)
    receipt=Path(sys.prefix)/'reference_installation.json'
    if receipt.is_file():
        report['installation_receipt']=json.loads(receipt.read_text())
        checks['installed_commit_receipt']=report['installation_receipt']['checkout']['commit']==args.expected_commit

    for pattern in ('test_reference_adapter.py','test_reference_masked_losses.py'):
        if not (repo/'tests'/pattern).is_file():
            checks[pattern]=False; continue
        p=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p',pattern,'-v'],
                         cwd=repo,capture_output=True,text=True)
        # A skipped test is reported explicitly, not silently treated as verified.
        checks[pattern]=p.returncode==0 and 'Ran 0 tests' not in p.stderr and 'skipped' not in p.stderr.lower()
        report[pattern]=dict(exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr)
    try: report['nvidia_smi']=output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'])
    except Exception as e: report['nvidia_smi']=str(e)
    report['passed']=bool(checks) and all(checks.values())
    report['gpu_checked']=not args.cpu_only
    report['installed_source_matches_checkout']='not established by imports; use bundle provenance'
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    # Never overwrite an earlier audit.
    with path.open('x') as f: json.dump(report,f,indent=2)
    print(json.dumps(dict(passed=report['passed'],report=str(path.resolve())),indent=2))
    return 0 if report['passed'] else 1


def pinned_requirements(text):
    result=[]
    for line in text.splitlines():
        line=line.strip()
        if not line or line.startswith('#'): continue
        if not re.fullmatch(r'[A-Za-z0-9_.-]+==[^\s;]+',line):
            raise ValueError(f'Cannot reproduce editable/direct/unpinned requirement: {line}')
        name=line.split('==')[0].lower().replace('_','-')
        if name != 'nvidia-physicsnemo': result.append(line)
    if not result: raise ValueError('Empty dependency snapshot')
    return result


def wheel_record(path):
    with zipfile.ZipFile(path) as z:
        names=[n for n in z.namelist() if n.endswith('.dist-info/METADATA')]
        if len(names)!=1: raise ValueError(f'Invalid wheel: {path}')
        meta=BytesParser().parsebytes(z.read(names[0]))
    return dict(file=path.name, name=meta['Name'],version=meta['Version'],sha256=digest(path))


def bundle(args):
    root=Path(args.output).resolve()
    if root.exists(): raise ValueError('Output must be a new directory')
    info=checkout_info(args.physicsnemo_dir,args.expected_commit)
    if info['dirty']: raise ValueError('Checkout has changes: preserve them in a commit and specify its SHA; do not discard files')
    frozen=output([sys.executable,'-m','pip','freeze','--all'])
    requirements=pinned_requirements(frozen)
    root.mkdir(parents=True); wheels=root/'wheels'; wheels.mkdir()
    (root/'source-freeze.txt').write_text(frozen+'\n')
    (root/'dependencies.txt').write_text('\n'.join(requirements)+'\n')
    indexes=[]
    for url in args.extra_index_url: indexes += ['--extra-index-url',url]
    run([sys.executable,'-m','pip','wheel','--no-deps','--wheel-dir',wheels,
         '-r',root/'dependencies.txt',*indexes])
    run([sys.executable,'-m','pip','wheel','--no-deps','--wheel-dir',wheels,info['path'],*indexes])
    # Ensure source did not change during packaging.
    after=checkout_info(args.physicsnemo_dir,args.expected_commit)
    if after['dirty']: raise ValueError('Checkout changed during build')
    records=[wheel_record(p) for p in sorted(wheels.glob('*.whl'))]
    names=[r['name'].lower().replace('_','-') for r in records]
    if len(names)!=len(set(names)) or names.count('nvidia-physicsnemo')!=1:
        raise ValueError('Duplicate wheels or absent PhysicsNeMo wheel')
    lock=''.join(f"{r['name']}=={r['version']} --hash=sha256:{r['sha256']}\n" for r in records)
    (root/'requirements.lock').write_text(lock)
    manifest=dict(schema='reference-wheel-bundle-v1',host=host(),checkout=info,wheels=records,
                  lock_sha256=digest(root/'requirements.lock'),validated_install=False)
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'Bundle created: {root}. Installation and doctor verification are still required.')
    return 0


def verify_bundle(root):
    root=Path(root).resolve(); m=json.loads((root/'manifest.json').read_text())
    if m['schema']!='reference-wheel-bundle-v1' or m['host']!=host():
        raise ValueError('Bundle requires the same Python minor, OS and machine architecture')
    if digest(root/'requirements.lock')!=m['lock_sha256']: raise ValueError('Lock hash mismatch')
    if not m['wheels']: raise ValueError('Empty wheel manifest')
    for w in m['wheels']:
        name=w['file']
        if Path(name).name!=name or not name.endswith('.whl'): raise ValueError('Unsafe wheel path')
        if digest(root/'wheels'/name)!=w['sha256']: raise ValueError(f'Wheel hash mismatch: {name}')
    return m


def install(args):
    root=Path(args.bundle).resolve(); m=verify_bundle(root)
    dest=Path(args.venv).resolve()
    if dest.exists(): raise ValueError('Refusing to modify an existing environment; choose a new venv path')
    run([sys.executable,'-m','venv',dest])
    python=dest/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    run([python,'-m','pip','install','--no-index','--find-links',root/'wheels',
         '--require-hashes','-r',root/'requirements.lock'])
    run([python,'-m','pip','check'])
    run([python,'-c','from physicsnemo.models.diffusion_unets import CorrDiffRegressionUNet; from physicsnemo.diffusion.preconditioners import EDMPrecondSuperResolution'])
    receipt=dict(bundle_manifest_sha256=digest(root/'manifest.json'),checkout=m['checkout'],
                 installed_packages=output([python,'-m','pip','freeze','--all']),gpu_verified=False)
    (dest/'reference_installation.json').write_text(json.dumps(receipt,indent=2))
    print(f'Environment installed: {python}. Run doctor with this Python before training.')
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    d=sub.add_parser('doctor'); d.add_argument('--project-dir',default='.')
    d.add_argument('--physicsnemo-dir',required=True); d.add_argument('--expected-commit',default=EXPECTED_COMMIT)
    d.add_argument('--dataset-dir',default='datasets/corrdiff_reference_v1'); d.add_argument('--gpus',default='0'); d.add_argument('--cpu-only',action='store_true'); d.add_argument('--output',required=True)
    b=sub.add_parser('bundle'); b.add_argument('--physicsnemo-dir',required=True)
    b.add_argument('--expected-commit',default=EXPECTED_COMMIT); b.add_argument('--output',required=True)
    b.add_argument('--extra-index-url',action='append',default=[])
    i=sub.add_parser('install'); i.add_argument('--bundle',required=True); i.add_argument('--venv',required=True)
    args=p.parse_args()
    try: return {'doctor':doctor,'bundle':bundle,'install':install}[args.command](args)
    except Exception as e:
        print(f'ERROR: {e}',file=sys.stderr); return 1

if __name__=='__main__': raise SystemExit(main())
