"""Reference-v1 PyTorch batches for a mask-aware CorrDiff training loop.

Deliberately not the legacy native (target, input, lead_time) tuple API.
"""
import hashlib
import operator
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from src.corrdiff.diagnostics.reference_reader import ReferenceReader

RAW_CHANNELS = ('tcwv', 't2m', 'u10', 'v10', 't_850', 'r_850',
                'u_850', 'v_850', 't_500', 'r_500', 'u_500', 'v_500')


class ReferenceCorrDiffDataset(Dataset):
    """Read-only, standardized inputs and unchanged code/50 targets.

    Each process opens its own reader. No patch filtering, fitting, imputation,
    augmentation, derived channels or test-split access is performed here.
    """
    def __init__(self, dataset_dir, split='train', *, channels=None):
        if split not in ('train', 'validation'):
            raise ValueError(f'Locked or unsupported training split: {split}')
        selected = tuple(RAW_CHANNELS if channels is None else channels)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError('Channels must be nonempty and unique')
        if any(c not in RAW_CHANNELS for c in selected):
            raise ValueError('Only reference raw channels are supported; derived channels need train-fit statistics')
        self.dataset_dir = str(Path(dataset_dir).resolve())
        self.split, self.channels = split, selected
        self._identity = self._fingerprint()
        reader = self._open()
        self._length = len(reader)
        self._reader, self._pid = None, None

    def _fingerprint(self):
        root = Path(self.dataset_dir)
        return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in ('summary.json', 'contract.json', 'normalization.json')}

    def _open(self):
        if self._fingerprint() != self._identity:
            raise ValueError('Reference metadata changed after adapter construction')
        reader = ReferenceReader(self.dataset_dir, self.split,
                                 input_mode='standardized', channels=self.channels)
        if tuple(reader.summary['channels']) != RAW_CHANNELS:
            raise ValueError('Expected the canonical reference twelve-channel contract')
        return reader

    def _get_reader(self):
        if self._reader is None or self._pid != os.getpid():
            reader = self._open()
            if len(reader) != self._length:
                raise ValueError('Split length changed after adapter construction')
            self._reader, self._pid = reader, os.getpid()
        return self._reader

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_reader=None, _pid=None)
        return state

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        index = operator.index(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        sample = self._get_reader()[index]
        meta = sample['metadata']
        row, col = int(meta['patch_row']), int(meta['patch_col'])
        if row not in (0, 8, 16, 24, 32, 38) or col not in (0, 8, 16, 24, 32, 40, 48, 52):
            raise ValueError('Invalid reference patch anchor')
        if not np.isfinite(sample['input']).all():
            raise ValueError('Non-finite standardized input')
        sha = meta['source_sha256']
        sha = sha.decode('ascii') if isinstance(sha, bytes) else str(sha)
        result = {key: torch.from_numpy(np.array(sample[key], copy=True, order='C'))
                  for key in ('input', 'target', 'mask')}
        result['metadata'] = {key: int(value) for key, value in meta.items()
                              if key != 'source_sha256'}
        result['metadata'].update(source_sha256=sha, source_index=index, split=self.split)
        return result

    def info(self):
        return dict(split=self.split, channels=list(self.channels), samples=len(self),
                    image_shape=[32, 32], target_transform='target_code / 50',
                    input_mode='standardized', mask_required=True,
                    batch_format='mapping', native_legacy_compatible=False,
                    metadata_sha256=dict(self._identity))
