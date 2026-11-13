#!/usr/bin/env python3
"""Technical CorrDiff generation smoke for reference-v1 masked checkpoint pair.

Uses official PhysicsNeMo regression & EDM models and its legacy deterministic
sampler, BUT loads the project's PyTorch state_dict checkpoints (not .mdlus).
Validation-only; technical inference, not a meteorological evaluation.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def sha256_stream(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def load_builder():
    """Reuse the exact architecture definitions from the approved smoke."""
    module_path = Path(__file__).with_name('18_reference_model_smoke.py')
    spec = importlib.util.spec_from_file_location('phase18_reference_model_builder', module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError('Cannot load reference model builder')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inspect_checkpoint(directory, expected_stage, experiment, reference_fp, stats_digest):
    """Verify audit, hash and metadata BEFORE loading untrusted checkpoint bytes."""
    directory = Path(directory).expanduser().resolve()
    audit_path = directory / 'audit.json'
    checkpoint = directory / 'technical_state_dict.pt'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    for name, wanted in {
        'schema': 'phase18-reference-model-smoke-v1',
        'status': 'PASS',
        'experiment': experiment,
        'stage': expected_stage,
        'reference_fingerprint': reference_fp,
        'target_transform': 'target_code / 50',
        'stock_train_py_executed': False,
        'locked_splits_accessed': False,
        'full_training': False,
        'checkpoint_format': 'PyTorch state_dict; NOT native NVIDIA .mdlus',
    }.items():
        if audit.get(name) != wanted:
            raise ValueError(f'{expected_stage} checkpoint audit mismatch: {name}')
    if Path(audit.get('checkpoint', '')).expanduser().resolve() != checkpoint:
        raise ValueError(f'{expected_stage} checkpoint path differs from signed audit')
    current_hash = sha256_stream(checkpoint)
    if current_hash != audit.get('checkpoint_sha256'):
        raise ValueError(f'{expected_stage} checkpoint sha256 mismatch')

    import torch
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or 'metadata' not in payload or 'model_state' not in payload:
        raise ValueError(f'{expected_stage} checkpoint schema mismatch')
    meta = payload['metadata']
    for name, wanted in {
        'experiment': experiment,
        'stage': expected_stage,
        'reference_fingerprint': reference_fp,
        'n_channels': 12 if experiment == 'C00' else 20,
    }.items():
        if meta.get(name) != wanted:
            raise ValueError(f'{expected_stage} checkpoint metadata mismatch: {name}')
    if meta.get('model_channels') not in (32, 64):
        raise ValueError('Unexpected technical model width')
    if experiment == 'C07' and meta.get('stats_sha256') != stats_digest:
        raise ValueError('C07 checkpoint uses a different set of derived statistics')
    return audit, meta, payload['model_state'], current_hash


def masked_diagnostics(prediction, target, mask):
    """Compute metrics only for valid radar pixels, never over masked zero padding."""
    import torch
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError('Generation/target/mask shapes differ')
    if mask.dtype != torch.bool:
        raise ValueError('Mask must be boolean')
    if not torch.isfinite(prediction).all() or not torch.isfinite(target[mask]).all():
        raise ValueError('Nonfinite generation or valid target')
    count = int(mask.sum().item())
    if count == 0:
        raise ValueError('Validation patch has no valid radar pixels')
    diff = (prediction[mask] - target[mask]).float()
    return dict(valid_pixels=count,
                masked_mse_code50=float(diff.square().mean().item()),
                masked_rmse_code50=float(diff.square().mean().sqrt().item()),
                masked_mae_code50=float(diff.abs().mean().item()),
                valid_prediction_outside_code_range_fraction=float(
                    ((prediction[mask] < 0) | (prediction[mask] > 1)).float().mean().item()),
                valid_no_echo_fraction=float((target[mask] == 0).float().mean().item()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', type=Path, default=Path('datasets/corrdiff_reference_v1'))
    parser.add_argument('--physicsnemo-dir', type=Path, default=Path.home()/'repositories/physicsnemo')
    parser.add_argument('--experiment', choices=('C00', 'C07'), required=True)
    parser.add_argument('--regression-run', type=Path, required=True)
    parser.add_argument('--diffusion-run', type=Path, required=True)
    parser.add_argument('--stats-file', type=Path)
    parser.add_argument('--sample-index', type=int, default=0)
    parser.add_argument('--num-steps', type=int, default=4)
    parser.add_argument('--sigma-min', type=float, default=0.01)
    parser.add_argument('--sigma-max', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)

    if args.sample_index < 0 or args.gpu < 0 or not 2 <= args.num_steps <= 16:
        parser.error('Require sample-index>=0, gpu>=0, 2<=num-steps<=16')
    if not 0 < args.sigma_min < args.sigma_max <= 8.0:
        parser.error('For technical bounded smoke: 0<sigma-min<sigma-max<=8.0')
    if args.experiment == 'C07' and args.stats_file is None:
        parser.error('C07 requires --stats-file')
    if args.experiment == 'C00' and args.stats_file is not None:
        parser.error('Do not specify --stats-file for C00')

    root = args.dataset_dir.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    if out.exists() or out == root or root in out.parents:
        raise ValueError('Output must be a NEW path outside the frozen dataset')
    out.mkdir(parents=True)
    began = time.monotonic()
    audit = dict(schema='phase18-reference-generation-smoke-v1', status='FAIL',
                 experiment=args.experiment, sample_split='validation',
                 sample_index=args.sample_index, model_training=False,
                 locked_splits_accessed=False, stock_generate_py_executed=False,
                 target_transform='target_code / 50', sampler='PhysicsNeMo deterministic_sampler',
                 sampler_parameters=dict(solver='heun', discretization='edm',
                                         num_steps=args.num_steps, sigma_min=args.sigma_min,
                                         sigma_max=args.sigma_max, seed=args.seed),
                 note='Functional inference only. Untrained/technical checkpoints; no skill claim.')
    try:
        import torch
        if not torch.cuda.is_available() or args.gpu >= torch.cuda.device_count():
            raise RuntimeError(f'GPU cuda:{args.gpu} unavailable')
        from adapter.reference_conditioning import reference_fingerprint
        builder = load_builder()
        builder.check_contract(PROJECT/'configs/phase18_reference_contract.json')
        audit['physicsnemo_commit'] = builder.check_checkout(args.physicsnemo_dir)
        fp = reference_fingerprint(root)
        audit['reference_fingerprint'] = fp
        stats_sha = sha256_stream(args.stats_file) if args.stats_file is not None else None
        reg_audit, reg_meta, reg_state, reg_sha = inspect_checkpoint(
            args.regression_run, 'regression', args.experiment, fp, stats_sha)
        dif_audit, dif_meta, dif_state, dif_sha = inspect_checkpoint(
            args.diffusion_run, 'diffusion', args.experiment, fp, stats_sha)
        if reg_meta['seed'] != dif_meta['seed']:
            raise ValueError('Stage seeds disagree')
        if reg_audit.get('physicsnemo_commit') != audit['physicsnemo_commit'] or dif_audit.get('physicsnemo_commit') != audit['physicsnemo_commit']:
            raise ValueError('Stage audits do not match approved PhysicsNeMo commit')
        # The diffusion audit records the precise regression checkpoint used.
        if dif_audit.get('regression_checkpoint_sha256') != reg_sha:
            raise ValueError('Diffusion checkpoint references a different regression model')
        audit['regression_sha256'] = reg_sha
        audit['diffusion_sha256'] = dif_sha
        audit['source_model_script_sha256'] = sha256_stream(Path(__file__).with_name('18_reference_model_smoke.py'))
        ds = builder.create_dataset(root, args.experiment, 'validation', args.stats_file)
        if not 0 <= args.sample_index < len(ds):
            raise IndexError('Validation sample index outside range')

        torch.cuda.set_device(args.gpu)
        torch.manual_seed(args.seed)
        device = torch.device(f'cuda:{args.gpu}')
        sample = ds[args.sample_index]
        x = sample['input'].unsqueeze(0).to(device=device, dtype=torch.float32)
        y = sample['target'].unsqueeze(0).to(device=device, dtype=torch.float32)
        mask = sample['mask'].unsqueeze(0).to(device=device, dtype=torch.bool)
        if x.shape != (1, 12 if args.experiment=='C00' else 20, 32, 32) or y.shape != (1, 1, 32, 32):
            raise ValueError('Invalid data shapes')

        n_chan = 12 if args.experiment=='C00' else 20
        regression = builder.build_model('regression', n_chan, reg_meta['model_channels']).to(device).eval()
        diffusion = builder.build_model('diffusion', n_chan, dif_meta['model_channels']).to(device).eval()
        regression.load_state_dict(reg_state, strict=True)
        diffusion.load_state_dict(dif_state, strict=True)
        regression.requires_grad_(False)
        diffusion.requires_grad_(False)

        from physicsnemo.diffusion.samplers import deterministic_sampler
        generator = torch.Generator(device=device).manual_seed(args.seed)
        latents = torch.randn((1, 1, 32, 32), generator=generator, device=device)
        with torch.inference_mode():
            mean = regression(torch.zeros((1, 1, 32, 32), device=device), x)
            residual = deterministic_sampler(
                diffusion, latents, img_lr=x, mean_hr=mean,
                num_steps=args.num_steps, sigma_min=args.sigma_min,
                sigma_max=args.sigma_max, solver='heun')
            prediction = mean + residual
            diag = masked_diagnostics(prediction, y, mask)
        torch.cuda.synchronize(args.gpu)
        if prediction.shape != (1, 1, 32, 32) or not torch.isfinite(mean).all() or not torch.isfinite(residual).all():
            raise ValueError('Unexpected generation shape or non-finite outputs')
        audit['metrics'] = diag
        audit['channels'] = list(ds.info()['channels'])
        audit['training_checkpoint_seeds'] = [reg_meta['seed'], dif_meta['seed']]
        audit['sample_metadata'] = sample.get('metadata', {})
        audit['peak_gpu_memory_bytes'] = int(torch.cuda.max_memory_allocated(args.gpu))
        prediction_path = out/'generated_validation_patch.npz'
        np.savez_compressed(prediction_path,
                            conditioning=x.detach().cpu().numpy(),
                            target_code50=y.detach().cpu().numpy(),
                            mask=mask.detach().cpu().numpy(),
                            regression_code50=mean.detach().cpu().numpy(),
                            residual_code50=residual.detach().cpu().numpy(),
                            prediction_code50_unclipped=prediction.detach().cpu().numpy())
        audit['generated_npz'] = str(prediction_path)
        audit['generated_npz_sha256'] = sha256_stream(prediction_path)
        audit['status']='PASS'
        print(f'{args.experiment} generation: PASS | valid={diag["valid_pixels"]} | masked_RMSE_code50={diag["masked_rmse_code50"]:.6g}', flush=True)
    except Exception as exc:
        audit['error'] = f'{type(exc).__name__}: {exc}'
        print(audit['error'], file=sys.stderr, flush=True)
    finally:
        audit['elapsed_seconds'] = round(time.monotonic()-began, 3)
        (out/'audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
        print(json.dumps({'status':audit['status'],'audit':str(out/'audit.json'),'error':audit.get('error')}))
    return 0 if audit['status']=='PASS' else 2


if __name__=='__main__':
    raise SystemExit(main())
