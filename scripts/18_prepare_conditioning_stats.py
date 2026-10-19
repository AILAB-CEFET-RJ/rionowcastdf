#!/usr/bin/env python3
"""Fit raw/derived statistics on Phase-15 train indices only."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import zarr
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapter.conditioning import ALL20, derive_batch, file_sha256, take, validate_indices, validate_raw

PHASE_VERSION = 'phase18-nvidia-corrdiff-native-integration-v2'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-dir', type=Path, default=Path('datasets/corrdiff_2011_2024'))
    p.add_argument('--phase15-dir', type=Path, default=Path('analysis_outputs/15_formal_splits'))
    p.add_argument('--phase17-dir', type=Path, default=Path('analysis_outputs/17_conditioning_ablations'))
    p.add_argument('--output-dir', type=Path, default=Path('analysis_outputs/18_nvidia_integration/conditioning_stats'))
    p.add_argument('--chunk-size', type=int, default=128)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    if args.chunk_size < 1:
        p.error('--chunk-size must be positive')
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError('Output is not empty; use --overwrite')
    audit_path = args.output_dir / 'conditioning_stats_audit.json'
    audit_path.unlink(missing_ok=True)
    root = zarr.open_group(str(args.dataset_dir / 'train.zarr'), mode='r')
    validate_raw(root)
    train_path = args.phase15_dir / 'train_patch_indices.npy'
    indices, _ = validate_indices(root, train_path, 'train')
    experiments = json.loads((args.phase17_dir / 'ablation_matrix.json').read_text())['experiments']
    if sorted(e['experiment_id'] for e in experiments) != [f'C{i:02}' for i in range(11)]:
        raise RuntimeError('Expected exactly C00..C10.')
    mean, m2, count = np.zeros(20), np.zeros(20), 0
    for start in range(0, len(indices), args.chunk_size):
        raw = take(root['input'], indices[start:start + args.chunk_size]).astype(np.float64)
        full = np.concatenate([raw, derive_batch(raw)], axis=1)
        if not np.isfinite(full).all():
            raise RuntimeError('Non-finite training predictors; statistics refused.')
        batch_n = full.shape[0] * full.shape[2] * full.shape[3]
        batch_mean = full.mean(axis=(0, 2, 3))
        batch_m2 = ((full - batch_mean[None, :, None, None]) ** 2).sum(axis=(0, 2, 3))
        delta, total = batch_mean - mean, count + batch_n
        m2 += batch_m2 + delta ** 2 * count * batch_n / total
        mean += delta * batch_n / total
        count = total
        if start % (args.chunk_size * 50) == 0:
            print(f'Processed {min(start + args.chunk_size, len(indices))}/{len(indices)}')
    std = np.sqrt(np.maximum(m2 / count, 1e-12))
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise RuntimeError('Non-finite statistics.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance = dict(fit_split=np.array('train'), train_indices_sha256=np.array(file_sha256(train_path)),
                      phase_version=np.array(PHASE_VERSION))
    np.savez(args.output_dir / 'all20_train_stats.npz', channel_names=np.array(ALL20),
             mean=mean, std=std, count=np.full(20, count), **provenance)
    emitted = []
    for exp in experiments:
        channels = exp['channels']
        positions = [ALL20.index(c) for c in channels]
        out = args.output_dir / f"{exp['experiment_id']}.npz"
        np.savez(out, experiment_id=np.array(exp['experiment_id']), channel_names=np.array(channels),
                 mean=mean[positions].astype(np.float32), std=std[positions].astype(np.float32),
                 count=np.full(len(channels), count), **provenance)
        emitted.append({'experiment_id': exp['experiment_id'], 'stats_file': str(out), 'n_channels': len(channels)})
    audit = dict(phase_version=PHASE_VERSION, status='PASS', fit_split='train',
                 n_train_patches=len(indices), train_indices_sha256=file_sha256(train_path),
                 all_means_finite=True, all_stds_positive_finite=True,
                 all20_channel_order=ALL20, experiments=emitted,
                 target_normalization='identity; stored log1p(clipped dBZ)',
                 manifest_sha256=file_sha256(args.phase17_dir / 'ablation_matrix.json'))
    audit_path.write_text(json.dumps(audit, indent=2))
    print(audit_path)


if __name__ == '__main__':
    main()
