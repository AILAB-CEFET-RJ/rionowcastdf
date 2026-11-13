"""Read-only, mask-aware 20-channel C07 dataset for the audited reference.

Raw ERA5 channels use authoritative full-train normalization. Derived channel
stats must be trained on train unique-grid hours; sampled stats are permitted
ONLY if explicitly requested for a technical smoke test.
"""
import os
from pathlib import Path
import operator
import numpy as np
import torch
from torch.utils.data import Dataset
from adapter.reference_conditioning import RAW, DERIVED, ALL20, derive_physical, read_stats, reference_fingerprint


class ReferenceDerivedDataset(Dataset):
    def __init__(self, dataset_dir, split='train', *, stats_path, allow_sampled_stats=False):
        if split not in ('train', 'validation'):
            raise ValueError(f'Locked or unsupported split: {split}')
        self.dataset_dir = str(Path(dataset_dir).expanduser().resolve())
        self.stats_path = str(Path(stats_path).expanduser().resolve())
        self.split = split
        self.reference_identity = reference_fingerprint(self.dataset_dir)
        self.derived_mean, self.derived_std, self.stats_info = read_stats(
            self.stats_path, self.dataset_dir, allow_sampled=allow_sampled_stats)
        reader = self._open()
        if tuple(reader.summary['channels']) != RAW:
            raise ValueError('Reference channel order mismatch')
        self.raw_mean = np.array(reader.mean, copy=True)
        self.raw_std = np.array(reader.std, copy=True)
        self._length = len(reader)
        self._reader, self._pid = None, None
        self.channels = ALL20

    def _open(self):
        if reference_fingerprint(self.dataset_dir) != self.reference_identity:
            raise ValueError('Reference dataset metadata changed')
        from src.corrdiff.diagnostics.reference_reader import ReferenceReader
        return ReferenceReader(self.dataset_dir, self.split, input_mode='physical')

    def _get_reader(self):
        if self._reader is None or self._pid != os.getpid():
            r = self._open()
            if len(r) != self._length:
                raise ValueError('Split length changed')
            self._reader, self._pid = r, os.getpid()
        return self._reader

    def __len__(self):
        return self._length

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_reader'], state['_pid'] = None, None
        return state

    def __getitem__(self, index):
        index = operator.index(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        s = self._get_reader()[index]
        raw = np.asarray(s['input'], dtype=np.float32)
        if raw.shape != (12, 32, 32) or not np.isfinite(raw).all():
            raise ValueError('Bad physical input')
        derived = derive_physical(raw)
        x = np.concatenate(((raw - self.raw_mean) / self.raw_std,
                            (derived - self.derived_mean[:, None, None]) / self.derived_std[:, None, None]), axis=0)
        target = np.asarray(s['target'], dtype=np.float32)
        mask = np.asarray(s['mask'], dtype=np.bool_)
        if target.shape != (1, 32, 32) or mask.shape != target.shape or not np.isfinite(x).all():
            raise ValueError('Bad target/mask schema or non-finite inputs')
        if not np.isfinite(target[mask]).all():
            raise ValueError('Invalid radar target')
        # ReferenceReader has already verified class codes and hidden zero placeholders.
        return dict(input=torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
                    target=torch.from_numpy(np.ascontiguousarray(target)),
                    mask=torch.from_numpy(np.ascontiguousarray(mask)),
                    metadata={k: (int(v) if k != 'source_sha256' else
                                  v.decode('ascii') if isinstance(v, bytes) else str(v))
                              for k, v in s['metadata'].items()})

    def info(self):
        return dict(split=self.split, channels=list(self.channels), samples=len(self),
                    target_transform='target_code / 50', mask_required=True,
                    sample_stats_scope=self.stats_info['scope'],
                    reference_fingerprint=self.reference_identity, stats_sha256=self.stats_info['file_sha256'],
                    native_legacy_tuple_compatible=False)
