"""Offline CPU contract tests. Real Zarr/GPU/PhysicsNeMo need CEFET validation."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from adapter import reference_conditioning
from adapter.reference_conditioning import (
    RAW, DERIVED, ALL20, derive_physical, reference_fingerprint, read_stats, sha256)
from adapter.reference_derived_dataset import ReferenceDerivedDataset


def script(name):
    spec = importlib.util.spec_from_file_location(name, PROJECT / 'scripts' / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeReader:
    def __init__(self, root, split, *, input_mode):
        if split not in ('train', 'validation'):
            raise AssertionError('Test split opened')
        assert input_mode == 'physical'
        self.summary = {'channels': list(RAW)}
        self.mean = np.full((12, 1, 1), 3.0, dtype=np.float32)
        self.std = np.full((12, 1, 1), 2.0, dtype=np.float32)
        self.split = split

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        raw = np.ones((12, 32, 32), dtype=np.float32) * 5
        raw[2] = 3
        raw[3] = 4
        mask = np.ones((1, 32, 32), dtype=bool)
        mask[0, 0, 0] = False
        return {'input': raw, 'target': np.full((1, 32, 32), .4, dtype=np.float32),
                'mask': mask, 'metadata': {'patch_row': 0, 'patch_col': 0,
                'timestamps': 1295287200, 'source_sha256': b'abc'}}


class TestReferenceMigration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ('summary.json', 'contract.json', 'normalization.json'):
            (self.root / name).write_text(json.dumps({'name': name, 'complete': True}))
        self.stats_file = self.root.parent / (self.root.name + '-stats.npz')
        self.addCleanup(lambda: self.stats_file.unlink(missing_ok=True))
        self.write_stats()

    def write_stats(self, **updates):
        values = dict(schema=np.array('reference-derived-stats-v1'),
                      fit_split=np.array('train'),
                      scope=np.array('smoke_sampled_train_hours'),
                      reference_fingerprint=np.array(reference_fingerprint(self.root)),
                      derived_channels=np.array(DERIVED), mean=np.full(8, 2.0, dtype=np.float32),
                      std=np.full(8, 0.5, dtype=np.float32),
                      train_hours_evaluated=np.array(5),
                      train_hours_available=np.array(75),
                      conditioning_sha256=np.array(sha256(Path(reference_conditioning.__file__))))
        values.update(updates)
        np.savez(self.stats_file, **values)

    def test_raw_derived_formulas(self):
        x = np.zeros((1, 12, 1, 1), dtype=np.float32)
        x[:, 2] = 3; x[:, 3] = 4
        x[:, 1] = 290; x[:, 4] = 280; x[:, 8] = 250
        expected = np.array([5, 0, 0, -30, -10, 0, 5, 0], dtype=np.float32)
        np.testing.assert_allclose(derive_physical(x).reshape(-1), expected)
        np.testing.assert_allclose(derive_physical(x[0]).reshape(-1), expected)
        with self.assertRaises(ValueError):
            derive_physical(np.zeros((11, 32, 32)))

    def test_train_only_provenance_guard(self):
        with self.assertRaisesRegex(ValueError, 'smoke'):
            read_stats(self.stats_file, self.root)
        _, _, info = read_stats(self.stats_file, self.root, allow_sampled=True)
        self.assertEqual(info['scope'], 'smoke_sampled_train_hours')
        self.write_stats(fit_split=np.array('test_primary'))
        with self.assertRaisesRegex(ValueError, 'non-training'):
            read_stats(self.stats_file, self.root, allow_sampled=True)

    def test_conditioning_formula_hash_guard(self):
        self.write_stats(conditioning_sha256=np.array('different-formulas'))
        with self.assertRaisesRegex(ValueError, 'formulas changed'):
            read_stats(self.stats_file,self.root,allow_sampled=True)

    def test_all_train_stats_must_cover_all_hours(self):
        self.write_stats(scope=np.array('all_train_hours'))
        with self.assertRaisesRegex(ValueError, 'all available training hours'):
            read_stats(self.stats_file,self.root,allow_sampled=True)

    def test_metadata_fingerprint_guard(self):
        (self.root / 'normalization.json').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'different reference'):
            read_stats(self.stats_file, self.root, allow_sampled=True)

    def test_locked_split_refused_before_reader(self):
        for split in ('test_primary','test_stress_ood'):
            with self.assertRaisesRegex(ValueError, 'Locked'):
                ReferenceDerivedDataset(self.root, split, stats_path=self.stats_file, allow_sampled_stats=True)

    def test_twenty_channels_mask_and_raw_normalization(self):
        namespace = types.ModuleType('src.corrdiff.diagnostics.reference_reader')
        namespace.ReferenceReader = FakeReader
        with patch.dict(sys.modules, {'src.corrdiff.diagnostics.reference_reader': namespace}):
            ds = ReferenceDerivedDataset(self.root, 'train', stats_path=self.stats_file,
                                         allow_sampled_stats=True)
            result = ds[0]
            self.assertEqual(result['input'].shape, (20,32,32))
            self.assertEqual(result['target'].shape, (1,32,32))
            self.assertEqual(result['mask'].dtype, torch.bool)
            self.assertFalse(result['mask'][0,0,0])
            self.assertEqual(ds.info()['target_transform'], 'target_code / 50')
            # 5 normalized by (5-3)/2; channel u10=3 -> 0.
            self.assertEqual(float(result['input'][0,0,0]), 1.)
            self.assertEqual(float(result['input'][2,0,0]), 0.)
            self.assertEqual(float(result['input'][12,0,0]), 6.)  # (hypot(3,4)-2)/.5
            with self.assertRaises(IndexError):
                ds[len(ds)]

    def test_train_hour_selection_never_repeats(self):
        s = script('18_fit_reference_derived_stats')
        hours = np.array([11,12,14,19,24,25,30],dtype=np.int64)
        selected,scope = s.choose_hours(hours,3,False)
        self.assertEqual(scope,'smoke_sampled_train_hours')
        self.assertEqual(selected.tolist(), [11,19,30])
        selected,scope=s.choose_hours(hours,1,True)
        self.assertEqual(scope,'all_train_hours')
        self.assertEqual(len(selected),7)

    def test_fit_unique_cells_matches_direct_population_stats(self):
        s = script('18_fit_reference_derived_stats')
        rng = np.random.default_rng(417)
        grids = {}
        for t in (10, 20, 30):
            raw = rng.normal(loc=2 + t / 10, scale=1.0, size=(12,70,84)).astype('f4')
            covered = np.ones((70,84),dtype='i4')
            covered[0,:] = 0
            # Deliberately vary patch multiplicity: must have no effect on fit.
            covered[1:30,1:30] = t
            grids[t] = dict(input=raw,patch_count=covered)
        reader = types.SimpleNamespace(hour=lambda t:grids[t])
        avg,std,n=s.fit(reader,[10,20,30])
        direct=np.concatenate([derive_physical(grids[t]['input'])[:,grids[t]['patch_count']>0]
                               for t in (10,20,30)],axis=1).astype('f8')
        np.testing.assert_allclose(avg,direct.mean(axis=1),rtol=1e-5,atol=1e-5)
        np.testing.assert_allclose(std,direct.std(axis=1),rtol=1e-5,atol=1e-5)
        self.assertEqual(n,direct.shape[1])

    def test_contract_gate_rejects_obsolete_target(self):
        s=script('18_reference_model_smoke')
        c=PROJECT/'configs/phase18_reference_contract.json'
        self.assertEqual(s.check_contract(c)['target_transform'],'target_code / 50')
        bad=self.root/'wrong_contract.json'
        broken=json.loads(c.read_text());broken['target_transform']='log1p'
        bad.write_text(json.dumps(broken))
        with self.assertRaisesRegex(ValueError,'contract mismatch'):
            s.check_contract(bad)

    def test_model_shapes_and_cli_guards_without_gpu(self):
        s=script('18_reference_model_smoke')
        self.assertEqual(s.model_args(12,'regression',32)['img_in_channels'],16)
        self.assertEqual(s.model_args(20,'diffusion',32)['img_in_channels'],25)
        with self.assertRaises(ValueError):
            s.create_dataset(self.root,'C00','test_primary',None)


if __name__=='__main__':
    unittest.main()
