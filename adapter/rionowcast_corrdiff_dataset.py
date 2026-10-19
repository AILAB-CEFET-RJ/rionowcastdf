"""Native CorrDiff adapter: train/validation only, strict valid masks.

The dataset contains different spatial slots, so one geographic grid cannot
represent every sample. Native train.py consumes the target/conditioning pair.
"""
from pathlib import Path
import importlib.util
import json
import numpy as np
import torch
import zarr
from datasets.base import DownscalingDataset, ChannelMetadata

_spec = importlib.util.spec_from_file_location(
    'rionowcast_conditioning', Path(__file__).with_name('conditioning.py'))
_common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_common)


class RioNowcastCorrDiffDataset(DownscalingDataset):
    def __init__(self, data_path, split_indices_path, ablation_matrix_path,
                 experiment_id, stats_path, split_name='train',
                 allow_locked_split=False, coordinate_mode='relative_patch',
                 grid_spacing_km=2.0, input_variables=None,
                 output_variables=None, invariant_variables=None):
        if allow_locked_split or split_name not in {'train', 'validation'}:
            raise RuntimeError(f'Locked split refused: {split_name}')
        if coordinate_mode != 'relative_patch':
            raise ValueError('Phase 18 supports relative_patch metadata only.')
        if invariant_variables:
            raise ValueError('Additional conditioning not registered in Phase 17.')
        self.data_path = str(Path(data_path).resolve())
        self.root = zarr.open_group(self.data_path, mode='r')
        _common.validate_raw(self.root)
        self.indices, self.timestamps = _common.validate_indices(self.root, split_indices_path, split_name)
        n = self.root['input'].shape[0]
        for key in ('target', 'mask'):
            if self.root[key].shape != (n, 1, 32, 32):
                raise RuntimeError(f'Invalid {key} shape.')
        experiments = json.loads(Path(ablation_matrix_path).read_text())['experiments']
        entries = [e for e in experiments if e['experiment_id'] == experiment_id]
        if len(entries) != 1:
            raise RuntimeError(f'Unknown/duplicated experiment: {experiment_id}')
        self.channels = entries[0]['channels']
        self.positions = [_common.ALL20.index(c) for c in self.channels]
        if len(set(self.channels)) != len(self.channels):
            raise RuntimeError('Duplicated conditioning channels.')
        if input_variables not in (None, []) and list(input_variables) != self.channels:
            raise RuntimeError('input_variables conflicts with Phase-17 channel order.')
        if output_variables not in (None, [], ['radar_reflectivity_log1p']):
            raise RuntimeError('Unexpected output_variables.')
        with np.load(stats_path, allow_pickle=False) as stats:
            if (stats['channel_names'].tolist() != self.channels
                    or str(stats['experiment_id'].item()) != experiment_id
                    or str(stats['fit_split'].item()) != 'train'):
                raise RuntimeError('Normalization does not match experiment/train contract.')
            train_path = Path(split_indices_path).parent / 'train_patch_indices.npy'
            if str(stats['train_indices_sha256'].item()) != _common.file_sha256(train_path):
                raise RuntimeError('Normalization belongs to another train manifest.')
            self.mean = np.asarray(stats['mean'], dtype=np.float32).reshape(-1, 1, 1)
            self.std = np.asarray(stats['std'], dtype=np.float32).reshape(-1, 1, 1)
        if (len(self.mean) != len(self.channels) or len(self.std) != len(self.channels)
                or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all()
                or np.any(self.std <= 0)):
            raise RuntimeError('Invalid normalization statistics.')
        self.experiment_id, self.split_name = experiment_id, split_name
        self.grid_spacing_km = float(grid_spacing_km)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        source = int(self.indices[index])
        if not np.all(np.asarray(self.root['mask'][source]) == 1):
            raise RuntimeError(f'Invalid radar pixels at index {source}; native loss is unmasked. '
                               'Stop and implement an approved masked loss; do not drop/fill silently.')
        raw = np.asarray(self.root['input'][source], dtype=np.float64)[None]
        full = np.concatenate([raw, _common.derive_batch(raw)], axis=1)[0]
        x = self.normalize_input(full[self.positions].astype(np.float32))
        y = np.asarray(self.root['target'][source], dtype=np.float32)
        if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(y < 0):
            raise RuntimeError(f'Invalid predictor or target at index {source}.')
        return torch.from_numpy(y.copy()), torch.from_numpy(np.ascontiguousarray(x))

    def input_channels(self):
        return [ChannelMetadata(name=c) for c in self.channels]

    def output_channels(self):
        return [ChannelMetadata(name='radar_reflectivity_log1p')]

    def image_shape(self):
        return (32, 32)

    def time(self):
        return self.timestamps.astype('datetime64[us]').astype(object).tolist()

    def latitude(self):
        return np.full(self.image_shape(), np.nan, dtype=np.float32)

    def longitude(self):
        return np.full(self.image_shape(), np.nan, dtype=np.float32)

    def normalize_input(self, x):
        return (x - self.mean) / self.std

    def denormalize_input(self, x):
        return x * self.std + self.mean

    def normalize_output(self, x):
        return x

    def denormalize_output(self, x):
        # Native space is stored log1p, not dBZ. Evaluation applies expm1.
        return x

    def sample_metadata(self, index):
        source = int(self.indices[index])
        return {'source_index': source, 'timestamp': str(self.timestamps[index]),
                'patch_row': int(self.root['patch_row'][source]),
                'patch_col': int(self.root['patch_col'][source])}

    def info(self):
        return {'experiment_id': self.experiment_id, 'split': self.split_name,
                'target_space': 'log1p(clipped radar dBZ)', 'mask_policy': 'require_all_valid',
                'absolute_position_conditioning': False,
                'geographic_coordinates': 'unavailable per dataset; latitude/longitude are NaN',
                'grid_spacing_km': self.grid_spacing_km,
                'generation_index': 'source_index + timestamp + patch_row + patch_col'}
