#!/usr/bin/env python3
"""Bounded REAL PhysicsNeMo model GPU smoke, with valid-pixel masked losses.

This is not a call to the deprecated stock CorrDiff train.py. It validates
forward/backward of its actual model classes, technical regression conditioning,
and the audited code/50 reference. It is not a scientific training campaign.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from adapter.reference_conditioning import reference_fingerprint, sha256

EXPECTED = 'c15bd23eb56216c32bd3e30180593e5685be9bc2'


def model_args(n_channels, stage, width):
    # Same PhysicsNeMo model families and positional embedding convention as
    # the pinned NVIDIA examples/weather/corrdiff training config.
    args = dict(img_resolution=32, img_out_channels=1, use_fp16=False,
                model_channels=width, channel_mult=[1, 2, 2],
                attn_resolutions=[16], N_grid_channels=4,
                embedding_type='zero', use_apex_gn=False)
    if stage == 'regression':
        args['img_in_channels'] = n_channels + 4
    else:
        args['img_in_channels'] = n_channels + 1 + 4
        args['gridtype'] = 'sinusoidal'
    return args


def check_contract(path):
    c = json.loads(Path(path).read_text())
    expected = dict(schema='phase18-reference-model-smoke-v1',
                    target_transform='target_code / 50',
                    patch_size=32, patches_per_hour=48,
                    raw_channels=12, derived_channels_C07=8,
                    physicsnemo_commit=EXPECTED,
                    native_legacy_train_py_compatible=False)
    if any(c.get(k) != value for k, value in expected.items()):
        raise ValueError('Versioned Phase-18 reference contract mismatch')
    if c.get('allowed_training_splits') != ['train', 'validation'] or set(c.get('locked_splits', [])) != {'test_primary','test_stress_ood'}:
        raise ValueError('Training split locks changed')
    return c


def check_checkout(path):
    root = Path(path).expanduser().resolve()
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != EXPECTED:
        raise RuntimeError(f'PhysicsNeMo HEAD {commit} != pinned {EXPECTED}')
    dirty = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if dirty.strip():
        raise RuntimeError('PhysicsNeMo tracked checkout is dirty')
    return commit


def create_dataset(root, exp, split, stats_path):
    if split not in ('train', 'validation'):
        raise ValueError('Locked split denied')
    if exp == 'C00':
        from adapter.reference_corrdiff_dataset import ReferenceCorrDiffDataset
        return ReferenceCorrDiffDataset(root, split)
    from adapter.reference_derived_dataset import ReferenceDerivedDataset
    return ReferenceDerivedDataset(root, split, stats_path=stats_path, allow_sampled_stats=True)


def batch_from(ds, idx, device):
    sample = ds[idx]
    return {k: sample[k].unsqueeze(0).to(device) for k in ('input', 'target', 'mask')}


def build_model(stage, n_channels, width):
    if stage == 'regression':
        from physicsnemo.models.diffusion_unets import CorrDiffRegressionUNet
        return CorrDiffRegressionUNet(**model_args(n_channels, 'regression', width))
    from physicsnemo.diffusion.preconditioners import EDMPrecondSuperResolution
    return EDMPrecondSuperResolution(**model_args(n_channels, 'diffusion', width))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset-dir', type=Path, default=Path('datasets/corrdiff_reference_v1'))
    ap.add_argument('--physicsnemo-dir', type=Path, default=Path.home()/'repositories/physicsnemo')
    ap.add_argument('--experiment', choices=['C00', 'C07'], required=True)
    ap.add_argument('--stage', choices=['regression', 'diffusion'], required=True)
    ap.add_argument('--stats-file', type=Path)
    ap.add_argument('--regression-checkpoint', type=Path)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=2)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--model-channels', type=int, choices=[32, 64], default=32)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--contract', type=Path, default=PROJECT / 'configs/phase18_reference_contract.json')
    args = ap.parse_args(argv)
    if not 1 <= args.steps <= 8 or args.gpu < 0:
        ap.error('Technical smoke permits 1..8 steps and nonnegative GPU index')
    check_contract(args.contract)
    if args.experiment == 'C07' and args.stats_file is None:
        ap.error('C07 requires --stats-file (train-only derived-channel statistics)')
    if args.stage == 'diffusion' and args.regression_checkpoint is None:
        ap.error('Diffusion smoke requires a matching --regression-checkpoint')
    import torch
    from adapter.reference_masked_flow import move_batch, train_step, validation_pass
    from adapter.reference_masked_losses import regression_loss, MaskedResidualEDMLoss
    if not torch.cuda.is_available() or args.gpu >= torch.cuda.device_count():
        raise RuntimeError('Requested GPU unavailable')
    root = args.dataset_dir.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    if out.exists():
        raise FileExistsError(f'Output already exists: {out}')
    if root == out or root in out.parents:
        raise ValueError('Cannot write inside reference dataset')
    out.mkdir(parents=True)
    report = dict(schema='phase18-reference-model-smoke-v1',status='FAIL',
                  experiment=args.experiment,stage=args.stage,full_training=False,
                  stock_train_py_executed=False,locked_splits_accessed=False,
                  gpu=args.gpu,seed=args.seed,steps_requested=args.steps,
                  target_transform='target_code / 50',loss='valid-pixel masked MSE / EDM',
                  run_type='technical_native_model_smoke')
    started=time.monotonic()
    try:
        report['physicsnemo_commit'] = check_checkout(args.physicsnemo_dir)
        report['reference_fingerprint'] = reference_fingerprint(root)
        stats_path = args.stats_file.expanduser().resolve() if args.stats_file else None
        train = create_dataset(root,args.experiment,'train',stats_path)
        valid = create_dataset(root,args.experiment,'validation',stats_path)
        if not len(train) or not len(valid):
            raise ValueError('Empty train/validation')
        report['train_info']=train.info()
        report['validation_info']=valid.info()
        torch.manual_seed(args.seed)
        torch.cuda.set_device(args.gpu)
        device=torch.device(f'cuda:{args.gpu}')
        n_channels=12 if args.experiment=='C00' else 20
        net = build_model(args.stage,n_channels,args.model_channels).to(device)
        regression=None
        if args.stage=='diffusion':
            checkpoint = args.regression_checkpoint.expanduser().resolve()
            payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
            meta=payload['metadata']
            for key,expected in [('experiment',args.experiment),('stage','regression'),
                                  ('reference_fingerprint',report['reference_fingerprint']),
                                  ('n_channels',n_channels)]:
                if meta.get(key)!=expected:
                    raise ValueError(f'Regression checkpoint does not match: {key}')
            regression=build_model('regression',n_channels,int(meta['model_channels'])).to(device)
            regression.load_state_dict(payload['model_state'],strict=True)
            regression.eval().requires_grad_(False)
            report['regression_checkpoint_sha256']=sha256(checkpoint)
            if args.experiment=='C07':
                if meta.get('stats_sha256')!=train.stats_info['file_sha256']:
                    raise ValueError('C07 regression checkpoint uses different stats')
        optim=torch.optim.Adam(net.parameters(),lr=1e-4)
        if args.stage=='regression':
            objective=lambda model,batch:regression_loss(model,batch)
        else:
            edmloss=MaskedResidualEDMLoss(hr_mean_conditioning=True)
            objective=lambda model,batch:edmloss(model,regression,batch)
        results=[]
        for i in range(args.steps):
            # Deterministic, bounded and only train: no random split or test access.
            idx=int(i*max(1,len(train)//args.steps))
            batch=batch_from(train,idx,device)
            step=train_step(net,optim,objective,batch)
            if not step['updated']:
                raise ValueError(f'No valid pixels for training step {i}')
            results.append(step)
            print(f'{args.experiment} {args.stage}: step {i+1}/{args.steps}, loss={step["loss"]:.6g}, valid={step["valid_pixels"]}',flush=True)
        validation_batch=batch_from(valid,0,device)
        torch.manual_seed(args.seed+10000)
        with torch.no_grad():
            val=validation_pass(net,objective,[validation_batch])
        torch.cuda.synchronize(args.gpu)
        report['train_steps']=results
        report['validation']=val
        report['peak_gpu_memory_bytes']=int(torch.cuda.max_memory_allocated(args.gpu))
        checkpoint=out/'technical_state_dict.pt'
        meta=dict(experiment=args.experiment,stage=args.stage,
                  reference_fingerprint=report['reference_fingerprint'],
                  n_channels=n_channels,model_channels=args.model_channels,seed=args.seed)
        if args.experiment=='C07':
            meta['stats_sha256']=train.stats_info['file_sha256']
        torch.save({'metadata':meta,'model_state':net.cpu().state_dict()},checkpoint)
        report['checkpoint']=str(checkpoint)
        report['checkpoint_sha256']=sha256(checkpoint)
        report['checkpoint_format']='PyTorch state_dict; NOT native NVIDIA .mdlus'
        report['status']='PASS'
    except Exception as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
        print(report['error'],file=sys.stderr,flush=True)
    finally:
        report['elapsed_seconds']=round(time.monotonic()-started,3)
        (out/'audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
        print(json.dumps({'status':report['status'],'audit':str(out/'audit.json'),'error':report.get('error')}))
    return 0 if report['status']=='PASS' else 2


if __name__=='__main__':
    raise SystemExit(main())
