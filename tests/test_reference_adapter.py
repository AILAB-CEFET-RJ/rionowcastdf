import hashlib
import json
import pickle
import socket
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import DataLoader
from adapter.reference_corrdiff_dataset import RAW_CHANNELS, ReferenceCorrDiffDataset


def fixture(root):
    channels = list(RAW_CHANNELS)
    contract = {'coverage_state': 'provisional_geometric_region'}
    norm = dict(channels=channels, input_mean=list(range(12)), input_std=[2]*12)
    for name, value in [('contract.json', contract), ('normalization.json', norm)]:
        (root/name).write_text(json.dumps(value))
    summary = dict(complete=True, channels=channels, artifact_sha256={
        name: hashlib.sha256((root/name).read_bytes()).hexdigest()
        for name in ('contract.json', 'normalization.json')})
    (root/'summary.json').write_text(json.dumps(summary))
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True, allow_nan=False).encode()).hexdigest()
    groups = {}
    for split in ('train', 'validation'):
        z = zarr.open_group(str(root/(split+'.zarr')), mode='w')
        x = np.broadcast_to(np.arange(12, dtype='f4')[None,:,None,None]+4, (2,12,32,32)).copy()
        y = np.zeros((2,1,32,32), dtype='f4'); y[:,0,10,10] = .4
        mask = np.ones_like(y, dtype=bool); mask[:,0,0,0] = False
        values = dict(input=x, target=y, mask=mask, timestamps=np.array([1000,1000]),
                      source_timestamps=np.array([880,880]), offset_seconds=np.array([-120,-120]),
                      patch_row=np.array([0,0]), patch_col=np.array([0,8]),
                      source_sha256=np.array(['a'*64]*2, dtype='S64'),
                      grid_lat=np.ones((70,84)), grid_lon=np.ones((70,84)), grid_support=np.ones((70,84),dtype=bool))
        for key, value in values.items():
            z.create_dataset(key, data=value, shape=value.shape)
        z.attrs.update(complete=True, version='reference-zarr-v1', split=split,
                       contract_sha256=digest, input_normalized=False,
                       target_transform='target_code / 50', channels=channels)
        zarr.consolidate_metadata(str(root/(split+'.zarr')))
        groups[split] = z
    return groups


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.groups = fixture(self.root)

    def test_normalization_once_and_target_mask(self):
        item = ReferenceCorrDiffDataset(self.root)[0]
        self.assertTrue(torch.all(item['input'] == 2))
        self.assertEqual(item['target'].dtype, torch.float32)
        self.assertEqual(item['mask'].dtype, torch.bool)
        self.assertFalse(item['mask'][0,0,0])
        self.assertTrue(item['mask'][0,1,1])
        self.assertEqual(item['target'][0,0,0], item['target'][0,1,1])
        self.assertAlmostEqual(item['target'][0,10,10].item(), .4)

    def test_channel_order_and_validation(self):
        z = self.groups['validation']; z['input'][0,11,:,:] = 19
        item = ReferenceCorrDiffDataset(self.root, 'validation', channels=['v_500','tcwv'])[0]
        self.assertEqual(item['input'][0,0,0], 4)
        self.assertEqual(item['input'][1,0,0], 2)

    def test_locked_splits_before_io(self):
        for split in ('test_primary','test_stress_ood','../train'):
            with self.assertRaises(ValueError):
                ReferenceCorrDiffDataset('/nonexistent', split)

    def test_invalid_channels(self):
        for channels in ([], ['tcwv','tcwv'], ['wind_speed_10']):
            with self.assertRaises(ValueError):
                ReferenceCorrDiffDataset(self.root, channels=channels)

    def test_all_invalid_preserved(self):
        self.groups['train']['mask'][0] = False
        self.groups['train']['target'][0] = 0
        self.assertFalse(ReferenceCorrDiffDataset(self.root)[0]['mask'].any())

    def test_invalid_placeholder_rejected(self):
        self.groups['train']['target'][0,0,0,0] = .4
        with self.assertRaises(ValueError):
            ReferenceCorrDiffDataset(self.root)[0]

    def test_bounds_and_anchor(self):
        ds = ReferenceCorrDiffDataset(self.root)
        for i in (-1,2):
            with self.assertRaises(IndexError): ds[i]
        self.groups['train']['patch_row'][0] = 1
        with self.assertRaises(ValueError): ds[0]

    def test_metadata_change_before_worker_open(self):
        ds = ReferenceCorrDiffDataset(self.root)
        with (self.root/'normalization.json').open('a') as f: f.write(' ')
        with self.assertRaisesRegex(ValueError, 'changed'): ds[0]

    def test_pickle_and_single_process_collation(self):
        ds = ReferenceCorrDiffDataset(self.root); ds[0]
        restored = pickle.loads(pickle.dumps(ds))
        self.assertIsNone(restored._reader)
        torch.testing.assert_close(ds[0]['input'], restored[0]['input'])
        batch = next(iter(DataLoader(ds, batch_size=2, num_workers=0)))
        self.assertEqual(batch['input'].shape, (2,12,32,32))

    def test_spawn_collation(self):
        try:
            sock = socket.socket(socket.AF_UNIX)
            sock.close()
        except PermissionError:
            self.skipTest('Host prohibits Unix sockets used by PyTorch multiprocessing')
        ds = ReferenceCorrDiffDataset(self.root)
        batch = next(iter(DataLoader(ds, batch_size=2, num_workers=1, multiprocessing_context='spawn', timeout=20)))
        self.assertEqual(batch['input'].shape, (2,12,32,32))
        self.assertEqual(batch['mask'].shape, (2,1,32,32))
        self.assertEqual(batch['metadata']['source_sha256'], ['a'*64]*2)


if __name__ == '__main__':
    unittest.main()
