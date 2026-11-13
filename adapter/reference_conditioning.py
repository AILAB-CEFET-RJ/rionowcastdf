"""Physical-space ERA5 derived features for the audited Zarr reference.

No access to test splits, no fitting, no transformation of the radar target.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

RAW = ('tcwv', 't2m', 'u10', 'v10', 't_850', 'r_850', 'u_850', 'v_850',
       't_500', 'r_500', 'u_500', 'v_500')
DERIVED = ('wind_speed_10', 'wind_speed_850', 'wind_speed_500',
           'delta_t_500_850', 'delta_t_850_surface', 'delta_r_500_850',
           'bulk_wind_diff_10_850', 'bulk_wind_diff_850_500')
ALL20 = RAW + DERIVED
VALID_CODES = np.array([0, .4, .5, .6, .7, .8, .9, 1.], dtype=np.float32)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def reference_fingerprint(root):
    """Bind derived-feature statistics to frozen reference metadata."""
    root = Path(root)
    keys = ('summary.json', 'contract.json', 'normalization.json')
    return hashlib.sha256(''.join(sha256(root / k) for k in keys).encode()).hexdigest()


def derive_physical(raw):
    """[12,H,W] or [B,12,H,W] -> eight derived fields, computed BEFORE scaling."""
    a = np.asarray(raw)
    squeeze = a.ndim == 3
    if squeeze:
        a = a[None, ...]
    if a.ndim != 4 or a.shape[1] != 12:
        raise ValueError(f'Expected [12,H,W] or [B,12,H,W]; found {a.shape}')
    a = a.astype(np.float64, copy=False)
    r = {n: a[:, i] for i, n in enumerate(RAW)}
    d = np.stack((
        np.hypot(r['u10'], r['v10']),
        np.hypot(r['u_850'], r['v_850']),
        np.hypot(r['u_500'], r['v_500']),
        r['t_500'] - r['t_850'],
        r['t_850'] - r['t2m'],
        r['r_500'] - r['r_850'],
        np.hypot(r['u10'] - r['u_850'], r['v10'] - r['v_850']),
        np.hypot(r['u_850'] - r['u_500'], r['v_850'] - r['v_500']),
    ), axis=1).astype(np.float32)
    return d[0] if squeeze else d


def read_stats(path, root, *, allow_sampled=False):
    """Read stats without pickle; require matching train-only provenance."""
    with np.load(path, allow_pickle=False) as f:
        required = {'schema', 'fit_split', 'scope', 'reference_fingerprint',
                    'derived_channels', 'mean', 'std', 'train_hours_evaluated',
                    'train_hours_available', 'conditioning_sha256'}
        if not required.issubset(f.files):
            raise ValueError('Derived statistics missing required fields')
        schema, split, scope = (str(f[k].item()) for k in ('schema', 'fit_split', 'scope'))
        fingerprint = str(f['reference_fingerprint'].item())
        if schema != 'reference-derived-stats-v1' or split != 'train':
            raise ValueError('Unknown statistics schema or non-training fit')
        if fingerprint != reference_fingerprint(root):
            raise ValueError('Statistics belong to a different reference dataset')
        if str(f['conditioning_sha256'].item()) != sha256(Path(__file__)):
            raise ValueError('Derived-feature formulas changed since statistics were fitted')
        if scope != 'all_train_hours' and not (allow_sampled and scope == 'smoke_sampled_train_hours'):
            raise ValueError('Sampled statistics are for smoke tests only, not full training')
        if tuple(f['derived_channels'].tolist()) != DERIVED:
            raise ValueError('Derived channel order differs from contract')
        mean = np.asarray(f['mean'], dtype=np.float32)
        std = np.asarray(f['std'], dtype=np.float32)
        if (mean.shape != (8,) or std.shape != (8,) or
                not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any()):
            raise ValueError('Invalid derived normalization')
        count = int(f['train_hours_evaluated'].item())
        total = int(f['train_hours_available'].item())
        if count < 1 or total < count:
            raise ValueError('Invalid training-hour counts')
        if scope == 'all_train_hours' and count != total:
            raise ValueError('Full-train scope requires all available training hours')
    return mean, std, dict(schema=schema, scope=scope, train_hours_evaluated=count,
                           reference_fingerprint=fingerprint, file_sha256=sha256(path))
