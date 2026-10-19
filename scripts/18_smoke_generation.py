#!/usr/bin/env python3
"""Reload native checkpoints and generate one explicit validation patch index."""
import argparse
from functools import partial
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--corrdiff-dir', required=True, type=Path)
    p.add_argument('--experiment-id', choices=['C00', 'C07'], required=True)
    p.add_argument('--regression-audit', required=True, type=Path)
    p.add_argument('--diffusion-audit', required=True, type=Path)
    p.add_argument('--sample-index', type=int, default=0)
    p.add_argument('--output-dir', type=Path, default=Path('analysis_outputs/18_nvidia_integration'))
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.output_dir / f'native_generation_audit_{args.experiment_id}.json'
    result = dict(phase_version='phase18-nvidia-corrdiff-native-integration-v2', status='FAIL',
                  experiment_id=args.experiment_id, mode='generation', sample_index=args.sample_index)
    try:
        import torch
        if not torch.cuda.is_available():
            result.update(status='BLOCKED', reason='CUDA required for native generation smoke.')
        else:
            records = [json.loads(path.read_text()) for path in (args.regression_audit, args.diffusion_audit)]
            for record, mode in zip(records, ('regression', 'diffusion')):
                if record['status'] != 'PASS' or record['experiment_id'] != args.experiment_id or record['mode'] != mode:
                    raise RuntimeError('Matching successful native stage audits required.')
                if len(record['checkpoints']) != 1:
                    raise RuntimeError('Expected one checkpoint per fresh tiny run.')
            if not records[0].get('fingerprint') or records[0]['fingerprint'] != records[1].get('fingerprint') or records[0]['seed'] != records[1]['seed']:
                raise RuntimeError('Regression/diffusion provenance mismatch.')
            corrdiff = args.corrdiff_dir.resolve()
            import subprocess
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=corrdiff, text=True).strip()
            if any(r['nvidia_commit'] != commit for r in records):
                raise RuntimeError('Checkout differs from native training audits.')
            sys.path.insert(0, str(corrdiff))
            from physicsnemo import Module
            from physicsnemo.diffusion.generate import regression_step, diffusion_step
            from physicsnemo.diffusion.samplers import deterministic_sampler
            spec = importlib.util.spec_from_file_location('phase18_runner', Path(__file__).with_name('18_run_native_smoke.py'))
            runner = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(runner)
            cfg = records[1]['resolved_configs']['diffusion']
            ds = runner.load_dataset(dict(cfg['dataset'], **cfg['validation']), corrdiff)
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from adapter.conditioning import evidence_fingerprint
            d = cfg['dataset']
            fingerprint = evidence_fingerprint(d['type'].split('::')[0], d['ablation_matrix_path'],
                Path(d['split_indices_path']).parent, Path(d['stats_path']).parent)
            if fingerprint != records[1]['fingerprint']:
                raise RuntimeError('Code/manifests/stats changed after native training.')
            if not 0 <= args.sample_index < len(ds):
                raise IndexError('Validation sample index out of range.')
            target, conditioning = ds[args.sample_index]
            device = torch.device('cuda')
            nets = [Module.from_checkpoint(r['checkpoints'][0], override_args={'use_apex_gn': False}).to(device).eval() for r in records]
            x = conditioning.unsqueeze(0).to(device)
            mean = regression_step(nets[0], x, (1, 1, 32, 32))
            residual = diffusion_step(net=nets[1],
                sampler_fn=partial(deterministic_sampler, num_steps=4, solver='heun'),
                img_shape=(32, 32), img_out_channels=1, rank_batches=[torch.tensor([42])],
                img_lr=x, rank=0, device=device,
                mean_hr=mean if cfg['model']['hr_mean_conditioning'] else None)
            prediction = (mean + residual).detach().cpu().numpy()
            if prediction.shape != (1, 1, 32, 32) or not np.isfinite(prediction).all():
                raise RuntimeError('Invalid generated tensor.')
            dbz = np.expm1(prediction.astype(np.float64))
            if not np.isfinite(dbz).all():
                raise RuntimeError('Inverse target transformation overflowed.')
            artifact = args.output_dir / f'generation_smoke_{args.experiment_id}.npz'
            np.savez(artifact, input=conditioning.numpy(), target_log1p=target.numpy(),
                     prediction_log1p=prediction, prediction_dbz_unclipped=dbz,
                     regression_log1p=mean.detach().cpu().numpy(), residual_log1p=residual.detach().cpu().numpy())
            result.update(status='PASS', nvidia_commit=commit, fingerprint=fingerprint, output=str(artifact),
                          sample_metadata=ds.sample_metadata(args.sample_index),
                          negative_dbz_fraction=float((dbz < 0).mean()),
                          rmse_log1p=float(np.sqrt(np.mean((prediction[0] - target.numpy()) ** 2))),
                          note='Technical smoke only; no clipping or scientific skill claim.',
                          sampler='native deterministic_sampler, heun, 4 steps; latent seed 42')
    except Exception as exc:
        result['reason'] = f'{type(exc).__name__}: {exc}'
    audit_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
