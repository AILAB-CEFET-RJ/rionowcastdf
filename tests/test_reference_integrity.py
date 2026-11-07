import contextlib
import io
import tempfile
import unittest
from pathlib import Path
import numpy as np
import zarr
from test_reference_zarr import fixture
from src.corrdiff.builder.contract_cache import run as cache_run
from src.corrdiff.builder.contract_zarr import run as build_run
from src.corrdiff.builder.contract_common import read
from audit_reference_zarr import audit

class IntegrityTests(unittest.TestCase):
    def test_pass_and_detect_corrupt_patch(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);c=fixture(root);cache_run(c);build_run(c)
            result=audit(root/'config.json',root/'audit_ok',1)
            self.assertTrue(result['integrity_passed'])
            self.assertEqual(result['selected_hours'],6)
            self.assertGreater(result['raw_png_samples'],0)
            z=zarr.open_group(str(root/'dataset/train.zarr'),mode='a')
            z['target'][0,0,0,0]=np.float32(.413)
            with self.assertRaisesRegex(ValueError,'mismatch: target'):
                audit(root/'config.json',root/'audit_bad',1)
            self.assertFalse(read(root/'audit_bad/summary.json')['integrity_passed'])
    def test_reject_output_in_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root)
            with self.assertRaisesRegex(ValueError,'separate'):
                audit(root/'config.json',root/'dataset/audit',1)

if __name__=='__main__':unittest.main()
