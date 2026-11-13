"""Offline CPU safety/unit checks for phase18 generation. No GPU/data required."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / 'scripts' / '18_reference_generation_smoke.py'
SPEC = importlib.util.spec_from_file_location('phase18_generation_smoke', MODULE_PATH)
GEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GEN)


class TestReferenceGeneration(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        self.root = Path(self.t.name)
        self.ref = 'a' * 64

    def record(self, stage='regression', exp='C00', stats_sha=None):
        d = self.root / stage
        d.mkdir(exist_ok=True)
        checkpoint = d/'technical_state_dict.pt'
        meta = dict(experiment=exp, stage=stage, reference_fingerprint=self.ref,
                    n_channels=12 if exp == 'C00' else 20, model_channels=32, seed=42)
        if exp=='C07':meta['stats_sha256']=stats_sha
        torch.save(dict(metadata=meta, model_state={'weight':torch.randn(2)}), checkpoint)
        digest = GEN.sha256_stream(checkpoint)
        audit = dict(schema='phase18-reference-model-smoke-v1', status='PASS',
                     experiment=exp, stage=stage, reference_fingerprint=self.ref,
                     target_transform='target_code / 50', stock_train_py_executed=False,
                     locked_splits_accessed=False, full_training=False,
                     checkpoint_format='PyTorch state_dict; NOT native NVIDIA .mdlus',
                     checkpoint=str(checkpoint), checkpoint_sha256=digest)
        (d/'audit.json').write_text(json.dumps(audit))
        return d,checkpoint

    def test_valid_checkpoint(self):
        directory, _ = self.record()
        audit,meta,state,digest = GEN.inspect_checkpoint(directory,'regression','C00',self.ref,None)
        self.assertEqual(meta['model_channels'],32)
        self.assertEqual(audit['checkpoint_sha256'],digest)
        self.assertIn('weight',state)

    def test_wrong_fingerprint_fails(self):
        directory,_ = self.record()
        with self.assertRaisesRegex(ValueError,'reference_fingerprint'):
            GEN.inspect_checkpoint(directory,'regression','C00','b'*64,None)

    def test_mutated_checkpoint_fails(self):
        directory,checkpoint = self.record()
        with checkpoint.open('ab') as stream:stream.write(b'tampered')
        with self.assertRaisesRegex(ValueError,'sha256'):
            GEN.inspect_checkpoint(directory,'regression','C00',self.ref,None)

    def test_wrong_stage_fails(self):
        directory,_ = self.record()
        with self.assertRaisesRegex(ValueError,'stage'):
            GEN.inspect_checkpoint(directory,'diffusion','C00',self.ref,None)

    def test_stats_mismatch_fails(self):
        directory,_ = self.record(exp='C07',stats_sha='a'*64)
        with self.assertRaisesRegex(ValueError,'statistics'):
            GEN.inspect_checkpoint(directory,'regression','C07',self.ref,'b'*64)

    def test_masked_rmse_ignores_unknown(self):
        prediction = torch.tensor([[[[1.,4.],[0.,10.]]]],dtype=torch.float32)
        target = torch.tensor([[[[1.,500.],[0.,500.]]]],dtype=torch.float32)
        mask = torch.tensor([[[[True,False],[True,False]]]],dtype=torch.bool)
        diag = GEN.masked_diagnostics(prediction,target,mask)
        self.assertEqual(diag['valid_pixels'],2)
        self.assertEqual(diag['masked_rmse_code50'],0.)
        self.assertEqual(diag['valid_no_echo_fraction'],0.5)

    def test_all_invalid_rejected(self):
        z=torch.zeros(1,1,1,1,dtype=torch.float32)
        with self.assertRaisesRegex(ValueError,'no valid'):
            GEN.masked_diagnostics(z,z,torch.zeros_like(z,dtype=torch.bool))

    def test_nonfinite_valid_rejected(self):
        target=torch.zeros(1,1,1,1,dtype=torch.float32)
        prediction=target.clone(); prediction[0,0,0,0]=float('nan')
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            GEN.masked_diagnostics(prediction,target,torch.ones_like(target,dtype=torch.bool))


if __name__=='__main__':unittest.main()
