#!/usr/bin/env python3
"""Compose real NVIDIA configs, then run isolated native GPU smoke jobs.

Hydra composition does not import GPU/model dependencies. Native training invokes
unmodified train.py via the explicit seed bridge. Compose PASS is not train PASS.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

PHASE_VERSION = 'phase18-nvidia-corrdiff-native-integration-v2'
PROJECT = Path(__file__).resolve().parents[1]


def compose_config(corrdiff, experiment, stage):
    sys.path.insert(0, str(corrdiff))
    name = f'config_training_rionowcast_{experiment}_{stage}_smoke'
    with initialize_config_dir(config_dir=str(corrdiff / 'conf'), version_base='1.2'):
        cfg = compose(config_name=name)
    OmegaConf.resolve(cfg)
    result = OmegaConf.to_container(cfg, throw_on_missing=True)
    if result['dataset']['split_name'] != 'train' or result['validation']['split_name'] != 'validation':
        raise RuntimeError('Only train/validation are allowed.')
    if result['dataset']['allow_locked_split'] or result['validation']['allow_locked_split']:
        raise RuntimeError('Locked split override refused.')
    if result['dataset']['experiment_id'] != experiment:
        raise RuntimeError('Experiment mismatch.')
    if not 1 <= result['training']['hp']['training_duration'] <= 256:
        raise RuntimeError('Only a tiny smoke budget is allowed.')
    return result


def load_dataset(cfg, corrdiff):
    sys.path.insert(0, str(corrdiff))
    cfg = dict(cfg)
    path, name = cfg.pop('type').split('::')
    spec = importlib.util.spec_from_file_location('rio_native_smoke_dataset', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)(**cfg)


def verify_events(directory):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    acc = EventAccumulator(str(directory), size_guidance={'scalars': 0})
    acc.Reload()
    losses = {tag: [event.value for event in acc.Scalars(tag)]
              for tag in acc.Tags()['scalars'] if 'loss' in tag}
    if not losses or not any('training' in t for t in losses) or not any('validation' in t for t in losses):
        raise RuntimeError('Missing training/validation loss evidence in TensorBoard.')
    if not all(np.isfinite(values).all() for values in losses.values()):
        raise RuntimeError('Non-finite native losses.')
    return {tag: {'count': len(values), 'last': values[-1]} for tag, values in losses.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--corrdiff-dir', required=True, type=Path)
    p.add_argument('--experiment-id', choices=['C00', 'C07'], default='C00')
    p.add_argument('--mode', choices=['compose', 'regression', 'diffusion'], default='compose')
    p.add_argument('--regression-checkpoint', type=Path)
    p.add_argument('--seed', type=int, choices=[42, 43, 44], default=42)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    corrdiff = args.corrdiff_dir.expanduser().resolve()
    output = args.output or PROJECT / 'analysis_outputs/18_nvidia_integration' / f'native_smoke_audit_{args.experiment_id}_{args.mode}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    audit = dict(phase_version=PHASE_VERSION, mode=args.mode, experiment_id=args.experiment_id,
                 status='FAIL', corrdiff_dir=str(corrdiff), full_training=False, seed=args.seed)
    start = time.time()
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=corrdiff, text=True).strip()
        expected = json.loads((PROJECT / 'configs/phase18_contract.json').read_text())['nvidia_commit']
        if commit != expected:
            raise RuntimeError('NVIDIA checkout differs from the audited commit.')
        if subprocess.run(['git', 'diff', '--quiet', 'HEAD'], cwd=corrdiff).returncode:
            raise RuntimeError('Tracked NVIDIA files modified; use clean pinned checkout.')
        audit['nvidia_commit'] = commit
        if args.regression_checkpoint:
            ckpt = args.regression_checkpoint.expanduser().resolve()
            if not ckpt.is_file():
                raise FileNotFoundError(ckpt)
            os.environ['CORRDIFF_REGRESSION_CKPT'] = str(ckpt)
        elif args.mode == 'diffusion':
            raise RuntimeError('--regression-checkpoint is required for diffusion')
        else:
            os.environ['CORRDIFF_REGRESSION_CKPT'] = '/COMPOSE_ONLY_NOT_A_CHECKPOINT.mdlus'
        stages = ['regression', 'diffusion'] if args.mode == 'compose' else [args.mode]
        configs = {stage: compose_config(corrdiff, args.experiment_id, stage) for stage in stages}
        audit['resolved_configs'] = configs
        if args.mode == 'compose':
            audit.update(status='PASS', scope='Hydra composition only; no native imports/training')
        else:
            import torch
            if not torch.cuda.is_available():
                audit.update(status='BLOCKED', reason='Native GPU smoke requires CUDA.')
            else:
                cfg = configs[args.mode]
                checkpoint_root = Path(cfg['training']['io']['checkpoint_dir'])
                run_dir = checkpoint_root.parent
                if run_dir.exists() and any(run_dir.iterdir()):
                    raise RuntimeError('Output is not empty; use new output root to avoid stale evidence/resume.')
                evidence = json.loads((output.parent / 'dataset_smoke_audit.json').read_text())
                if evidence.get('status') != 'PASS' or evidence.get('phase_version') != PHASE_VERSION or evidence.get('all_train_validation_masks_valid') is not True:
                    raise RuntimeError('Run complete dataset/mask smoke before native training.')
                ds = load_dataset(cfg['dataset'], corrdiff)
                ds[0]
                sys.path.insert(0, str(PROJECT))
                from adapter.conditioning import evidence_fingerprint
                data_cfg = cfg['dataset']
                fingerprint = evidence_fingerprint(data_cfg['type'].split('::')[0],
                    data_cfg['ablation_matrix_path'], Path(data_cfg['split_indices_path']).parent,
                    Path(data_cfg['stats_path']).parent)
                if evidence.get('fingerprint') != fingerprint or evidence.get('data_path') != data_cfg['data_path']:
                    raise RuntimeError('Dataset audit stale; rerun dataset smoke.')
                audit['fingerprint'] = fingerprint
                audit['adapter_info'] = ds.info()
                log = output.with_suffix('.log')
                seed_audit = output.with_suffix('.seeds.json').resolve()
                seed_audit.unlink(missing_ok=True)
                command = [sys.executable, str(PROJECT / 'scripts/18_native_entrypoint.py'),
                           '--corrdiff-dir', str(corrdiff), '--seed', str(args.seed),
                           '--seed-audit', str(seed_audit),
                           f'--config-name=config_training_rionowcast_{args.experiment_id}_{args.mode}_smoke']
                with log.open('w') as stream:
                    proc = subprocess.run(command, cwd=corrdiff, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT)
                audit['result'] = dict(command=command, returncode=proc.returncode, log=str(log))
                if proc.returncode:
                    raise RuntimeError(f'Native train.py failed; inspect {log}')
                checkpoints = sorted(checkpoint_root.rglob('*.mdlus'))
                if not checkpoints:
                    raise RuntimeError('Native run did not produce a model checkpoint.')
                audit['loss_evidence'] = verify_events(run_dir / 'tensorboard')
                audit['checkpoints'] = [str(c) for c in checkpoints]
                seeds = json.loads(seed_audit.read_text())
                if seeds['model_seeds'] != [args.seed] or seeds['sampler_seeds'] != [args.seed]:
                    raise RuntimeError('Native seed bridge evidence mismatch')
                audit['seed_evidence'] = seeds
                audit['status'] = 'PASS'
    except Exception as exc:
        audit['reason'] = f'{type(exc).__name__}: {exc}'
    audit['elapsed_seconds'] = time.time() - start
    output.write_text(json.dumps(audit, indent=2))
    print(json.dumps({'status': audit['status'], 'audit': str(output), 'reason': audit.get('reason')}, indent=2))
    return 0 if audit['status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
