"""Shared physical-space conditioning and split checks for Phase 18."""
from pathlib import Path
import hashlib
import numpy as np

RAW_CHANNELS = ['tcwv', 't2m', 'u10', 'v10', 't_850', 'r_850', 'u_850', 'v_850',
                't_500', 'r_500', 'u_500', 'v_500']
DERIVED_CHANNELS = ['wind_speed_10', 'wind_speed_850', 'wind_speed_500',
                    'delta_t_500_850', 'delta_t_850_surface', 'delta_r_500_850',
                    'bulk_wind_diff_10_850', 'bulk_wind_diff_850_500']
ALL20 = RAW_CHANNELS + DERIVED_CHANNELS


def derive_batch(raw):
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 4 or raw.shape[1] != 12:
        raise ValueError(f'Expected [B,12,H,W], received {raw.shape}')
    r = {name: raw[:, i] for i, name in enumerate(RAW_CHANNELS)}
    return np.stack([
        np.hypot(r['u10'], r['v10']), np.hypot(r['u_850'], r['v_850']),
        np.hypot(r['u_500'], r['v_500']), r['t_500'] - r['t_850'],
        r['t_850'] - r['t2m'], r['r_500'] - r['r_850'],
        np.hypot(r['u10'] - r['u_850'], r['v10'] - r['v_850']),
        np.hypot(r['u_850'] - r['u_500'], r['v_850'] - r['v_500']),
    ], axis=1)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def take(arr, indices):
    return np.asarray(arr.oindex[(indices, *[slice(None)] * (arr.ndim - 1))])


def validate_raw(root):
    if list(root.attrs.get('channels', [])) != RAW_CHANNELS:
        raise RuntimeError('Zarr channel metadata differs from the frozen raw12 order.')
    if root['input'].shape[1:] != (12, 32, 32):
        raise RuntimeError('Expected input [N,12,32,32].')
    if root.attrs.get('target_transform') != 'log1p':
        raise RuntimeError('Expected stored log1p target.')


def validate_indices(root, path, split_name):
    """Read calendar/index metadata only, never locked input/target fields."""
    if split_name not in {'train', 'validation'}:
        raise RuntimeError(f'Locked split refused: {split_name}')
    path = Path(path)
    if path.name != f'{split_name}_patch_indices.npy':
        raise RuntimeError('Split filename does not match requested split.')
    indices = np.load(path, allow_pickle=False)
    n = root['input'].shape[0]
    if (indices.ndim != 1 or indices.dtype.kind not in 'iu' or not indices.size
            or indices.min() < 0 or indices.max() >= n
            or np.unique(indices).size != indices.size):
        raise RuntimeError('Invalid, empty, duplicated or out-of-range split indices.')
    ts = take(root['timestamps'], indices).astype('datetime64[s]')
    years = ts.astype('datetime64[Y]').astype(int) + 1970
    low, high = (2011, 2020) if split_name == 'train' else (2021, 2022)
    if np.any((years < low) | (years > high)):
        raise RuntimeError('Split calendar mismatch (or timestamps not Unix seconds).')
    if np.any(np.unique(ts, return_counts=True)[1] != 12):
        raise RuntimeError('Every selected timestamp must retain all 12 patches.')
    for other in ('train', 'validation', 'test_primary', 'test_stress_ood', 'excluded_purge'):
        if other == split_name:
            continue
        other_path = path.parent / f'{other}_patch_indices.npy'
        if not other_path.exists():
            raise FileNotFoundError(other_path)
        other_idx = np.load(other_path, allow_pickle=False)
        if (other_idx.ndim != 1 or other_idx.dtype.kind not in 'iu'
                or (other_idx.size and (other_idx.min() < 0 or other_idx.max() >= n))):
            raise RuntimeError(f'Invalid {other} indices.')
        if np.intersect1d(indices, other_idx).size:
            raise RuntimeError(f'{split_name} overlaps {other}.')
        if other_idx.size and np.intersect1d(ts, take(root['timestamps'], other_idx).astype('datetime64[s]')).size:
            raise RuntimeError(f'Timestamp split across {split_name} and {other}.')
    return indices.astype(np.int64), ts


def evidence_fingerprint(adapter_path, ablation_path, split_dir, stats_dir):
    adapter_path, split_dir, stats_dir = map(Path, (adapter_path, split_dir, stats_dir))
    paths = [adapter_path, adapter_path.with_name('conditioning.py'), Path(ablation_path)]
    paths += [split_dir / f'{s}_patch_indices.npy' for s in
              ('train', 'validation', 'test_primary', 'test_stress_ood', 'excluded_purge')]
    paths += [stats_dir / f'{s}.npz' for s in ('C00', 'C07')]
    return hashlib.sha256(''.join(file_sha256(p) for p in paths).encode()).hexdigest()
