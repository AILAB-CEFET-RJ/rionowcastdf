"""Shared NumPy reader. Zero is a no-echo code, never measured zero dBZ."""
import hashlib
import json
import operator
from pathlib import Path

import numpy as np
import zarr

SPLITS = ('train', 'validation', 'test_primary', 'test_stress_ood')
CODES = np.array([0, 20, 25, 30, 35, 40, 45, 50], dtype='f4')
META = ('timestamps', 'source_timestamps', 'offset_seconds', 'patch_row',
        'patch_col', 'source_sha256')


def _read(path):
    return json.loads(path.read_text())


class ReferenceReader:
    """Open one split read-only; inputs default to physical units.

    batch() always returns a leading sample dimension. hour() reconstructs
    unique spatial points and rejects disagreements between overlapping patches.
    This reader does not validate raw sources or infer physical radar coverage.
    """
    def __init__(self, root, split='train', *, input_mode='physical', channels=None):
        if split not in SPLITS:
            raise ValueError(f'Unknown split: {split}')
        if input_mode not in ('physical', 'standardized'):
            raise ValueError('input_mode must be physical or standardized')
        self.root, self.split, self.input_mode = Path(root), split, input_mode
        self.summary = _read(self.root / 'summary.json')
        if not self.summary.get('complete'):
            raise ValueError('Incomplete reference dataset')
        for name in ('normalization.json', 'contract.json'):
            digest = hashlib.sha256((self.root / name).read_bytes()).hexdigest()
            if digest != self.summary['artifact_sha256'][name]:
                raise ValueError(f'Metadata hash mismatch: {name}')
        self.contract = _read(self.root / 'contract.json')
        digest = hashlib.sha256(json.dumps(self.contract, sort_keys=True, allow_nan=False).encode()).hexdigest()
        self.z = zarr.open_consolidated(str(self.root / f'{split}.zarr'), mode='r')
        a = self.z.attrs
        if (not a.get('complete') or a.get('version') != 'reference-zarr-v1'
                or a.get('split') != split or a.get('contract_sha256') != digest
                or a.get('input_normalized') is not False
                or a.get('target_transform') != 'target_code / 50'):
            raise ValueError('Unsupported or inconsistent Zarr contract')
        norm = _read(self.root / 'normalization.json')
        names = list(a['channels'])
        if names != norm['channels'] or names != self.summary['channels']:
            raise ValueError('Channel order mismatch')
        self.channels = list(names if channels is None else channels)
        if not self.channels or len(set(self.channels)) != len(self.channels):
            raise ValueError('Select at least one channel, without duplicates')
        self.channel_indices = [names.index(c) for c in self.channels]
        mean, std = (np.asarray(norm[k], dtype='f4') for k in ('input_mean', 'input_std'))
        if (mean.shape != (len(names),) or std.shape != mean.shape
                or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any()):
            raise ValueError('Invalid training normalization')
        self.mean = mean[self.channel_indices, None, None]
        self.std = std[self.channel_indices, None, None]
        n = self.z['input'].shape[0]
        for key, shape, dtype in [('input', (n, len(names), 32, 32), 'f4'),
                                 ('target', (n, 1, 32, 32), 'f4'),
                                 ('mask', (n, 1, 32, 32), 'bool')]:
            if self.z[key].shape != shape or self.z[key].dtype != np.dtype(dtype):
                raise ValueError(f'Unexpected tensor schema: {key}')
        for key in META:
            if self.z[key].shape != (n,):
                raise ValueError(f'Unexpected metadata shape: {key}')
        self.lat, self.lon, self.support = (self.z['grid_' + k][:] for k in ('lat', 'lon', 'support'))
        if any(v.shape != (70, 84) for v in (self.lat, self.lon, self.support)):
            raise ValueError('Unexpected grid shape')
        self._times = None
        self._hours = None

    def __len__(self):
        return self.z['input'].shape[0]

    def batch(self, start, stop):
        start, stop = operator.index(start), operator.index(stop)
        if not 0 <= start < stop <= len(self):
            raise IndexError('Require 0 <= start < stop <= split length')
        sl = slice(start, stop)
        x = self.z['input'][sl][:, self.channel_indices]
        y, mask = self.z['target'][sl], self.z['mask'][sl]
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Non-finite tensors')
        if not np.isin(y[mask], CODES / np.float32(50)).all() or np.any(y[~mask] != 0):
            raise ValueError('Invalid target encoding or unknown placeholder')
        meta = {k: self.z[k][sl] for k in META}
        if (np.any(meta['source_timestamps'] - meta['timestamps'] != meta['offset_seconds'])
                or np.any(np.abs(meta['offset_seconds'].astype('i8')) > 120)):
            raise ValueError('Temporal offset mismatch')
        if self.input_mode == 'standardized':
            x = (x - self.mean) / self.std
        # Invalid pixels are NaN in target_code, but remain zero in stored target.
        code = np.where(mask, np.rint(y * 50), np.nan).astype('f4')
        return dict(input=x, target=y, target_code=code, mask=mask,
                    no_echo=mask & (y == 0), metadata=meta)

    def __getitem__(self, index):
        index = operator.index(index)
        b = self.batch(index, index + 1)
        return {k: ({m: v[0] for m, v in value.items()} if k == 'metadata'
                    else value[0]) for k, value in b.items()}

    def iter_batches(self, batch_size=48):
        batch_size = operator.index(batch_size)
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        for start in range(0, len(self), batch_size):
            yield self.batch(start, min(start + batch_size, len(self)))

    @staticmethod
    def echo_at_least(batch, dbz=20):
        if dbz not in CODES[1:]:
            raise ValueError('Threshold must be a displayed class: 20,25,...,50')
        return batch['mask'] & (batch['target'] >= np.float32(dbz / 50))

    @property
    def hours(self):
        if self._times is None:
            self._times = self.z['timestamps'][:]
            if np.any(np.diff(self._times) < 0):
                raise ValueError('Patch timestamps must be ordered within each split')
        if self._hours is None:
            self._hours = np.unique(self._times)
        return self._hours.copy()

    def hour(self, timestamp_utc):
        """Reconstruct a Unix-second UTC hour without overlap weighting bias.

        NaN denotes uncovered input/invalid target_code. mask and patch_count
        distinguish valid no echo, invalid observations and uncovered positions.
        """
        if self._times is None:
            self.hours
        timestamp_utc = operator.index(timestamp_utc)
        lo = np.searchsorted(self._times, timestamp_utc, side='left')
        hi = np.searchsorted(self._times, timestamp_utc, side='right')
        if lo == hi:
            raise KeyError(timestamp_utc)
        b = self.batch(int(lo), int(hi))
        for key in ('source_timestamps', 'offset_seconds', 'source_sha256'):
            if len(np.unique(b['metadata'][key])) != 1:
                raise ValueError('Inconsistent source identity within hour')
        x = np.full((len(self.channels), 70, 84), np.nan, dtype='f4')
        y = np.zeros((1, 70, 84), dtype='f4')
        mask = np.zeros((1, 70, 84), dtype=bool)
        count = np.zeros((70, 84), dtype='i4')
        positions = set()
        for i, (r, c) in enumerate(zip(b['metadata']['patch_row'], b['metadata']['patch_col'])):
            r, c = int(r), int(c)
            if not (0 <= r <= 38 and 0 <= c <= 52) or (r, c) in positions:
                raise ValueError('Invalid or duplicate patch position')
            positions.add((r, c))
            sl = (slice(None), slice(r, r+32), slice(c, c+32))
            old = count[r:r+32, c:c+32] > 0
            for full, patch in ((x, b['input'][i]), (y, b['target'][i]), (mask, b['mask'][i])):
                if not np.array_equal(full[sl][:, old], patch[:, old]):
                    raise ValueError('Overlapping patches disagree')
                full[sl] = patch
            count[r:r+32, c:c+32] += 1
        return dict(input=x, target=y, target_code=np.where(mask, np.rint(y*50), np.nan),
                    mask=mask, no_echo=mask & (y == 0), patch_count=count,
                    timestamp_utc=timestamp_utc, metadata=b['metadata'])
