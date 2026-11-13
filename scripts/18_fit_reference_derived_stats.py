#!/usr/bin/env python3
"""Fit eight derived ERA5 channel statistics on unique train grid cells/hours.

Default is a limited train-only sample for Phase-18 smoke tests. For the
complete experimental fitting, specify --all-train-hours (high I/O cost).
No validation/test arrays are read, and no patches are double-counted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapter import reference_conditioning
from adapter.reference_conditioning import DERIVED, derive_physical, reference_fingerprint, sha256


def choose_hours(hours, max_hours, all_train_hours):
    hours = np.asarray(hours, dtype=np.int64)
    if hours.ndim != 1 or not len(hours) or np.unique(hours).size != len(hours):
        raise ValueError('Train hours empty or duplicated')
    if all_train_hours:
        return hours, 'all_train_hours'
    if max_hours < 1:
        raise ValueError('--max-hours must be positive')
    n = min(max_hours, len(hours))
    return hours[np.linspace(0, len(hours) - 1, n, dtype=np.int64)], 'smoke_sampled_train_hours'


def fit(reader, selected):
    """Numerically stable population variance across unique covered cells."""
    count, mean, m2 = 0, np.zeros(8, dtype=np.float64), np.zeros(8, dtype=np.float64)
    for n, hour in enumerate(selected, 1):
        h = reader.hour(int(hour))
        covered = np.asarray(h['patch_count']) > 0
        if covered.shape != (70, 84) or not covered.any():
            raise ValueError('Bad or empty reference geometric coverage')
        physical = np.asarray(h['input'], dtype=np.float32)
        if physical.shape != (12, 70, 84):
            raise ValueError('Unexpected full-grid ERA5 schema')
        d = derive_physical(physical)[:, covered].astype(np.float64)
        if not np.isfinite(d).all():
            raise ValueError(f'Non-finite derived inputs in train hour {hour}')
        size = d.shape[1]
        avg = d.mean(axis=1)
        block_m2 = ((d - avg[:, None]) ** 2).sum(axis=1)
        delta = avg - mean
        total = count + size
        m2 += block_m2 + (delta ** 2) * count * size / total
        mean += delta * size / total
        count = total
        if n % 16 == 0 or n == len(selected):
            print(f'Train hours processed: {n}/{len(selected)}; grid cells: {count}', flush=True)
    std = np.sqrt(np.maximum(m2 / count, 0.0))
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 1e-10):
        raise ValueError('Derived channels have non-finite or effectively zero variance')
    return mean.astype(np.float32), std.astype(np.float32), count


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir', type=Path, default=Path('datasets/corrdiff_reference_v1'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-hours', type=int, default=96)
    p.add_argument('--all-train-hours', action='store_true')
    a = p.parse_args(argv)
    root = a.dataset_dir.expanduser().resolve()
    output = a.output.expanduser().resolve()
    if output.suffix != '.npz':
        p.error('--output must end in .npz')
    if output.exists() or output.with_suffix('.json').exists():
        raise FileExistsError(f'Output already exists: {output}')
    if root == output or root in output.parents:
        raise ValueError('Output must be outside frozen dataset')
    from src.corrdiff.diagnostics.reference_reader import ReferenceReader
    reader = ReferenceReader(root, 'train', input_mode='physical')
    selected, scope = choose_hours(reader.hours, a.max_hours, a.all_train_hours)
    mean, std, cells = fit(reader, selected)
    info = dict(schema='reference-derived-stats-v1', fit_split='train', scope=scope,
                train_hours_available=len(reader.hours), train_hours_evaluated=len(selected),
                unique_cells_seen=cells, reference_fingerprint=reference_fingerprint(root),
                train_hour_selection_sha256=hashlib.sha256(selected.tobytes()).hexdigest(),
                source_code_sha256=sha256(Path(__file__)),
                conditioning_sha256=sha256(Path(reference_conditioning.__file__)),
                use_for_full_training=scope == 'all_train_hours',
                tests_accessed=False, target_transform='target_code / 50')
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, mean=mean, std=std, derived_channels=np.array(DERIVED),
             **{k: np.array(v) for k, v in info.items()})
    output.with_suffix('.json').write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(json.dumps({'output':str(output),'scope':scope,'hours':len(selected),'cells':cells},indent=2))


if __name__ == '__main__':
    main()
