import contextlib
import io
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
import numpy as np
from src.corrdiff.builder.contract_common import dump, sha
from test_reference_zarr import fixture
from src.corrdiff.builder.contract_cache import run as cache_run
from src.corrdiff.builder.contract_zarr import run as build_run
from src.corrdiff.builder.contract_common import read
from audit_reference_zarr import audit

class FullIntegrityTests(unittest.TestCase):
    def test_complete_sources(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);c=fixture(root);cache_run(c);build_run(c)
            result=audit(root/'config.json',root/'audit',all_selected_pngs=True,recompute_era5=True)
            self.assertTrue(result['integrity_passed'])
            self.assertTrue(result['all_selected_png_paths_checked'])
            self.assertEqual(result['raw_png_paths_checked'],6)
            self.assertEqual(result['era5_raw_hours_checked'],6)
            self.assertTrue((root/'audit/era5_sources_verified.json').exists())
            # Same-content PNGs must each have their acquisition path checked.
            png=sorted((root/'radar').rglob('*.png'))[-1]
            png.write_bytes(b'broken')
            with self.assertRaisesRegex(ValueError,'PNG path checksum'):
                audit(root/'config.json',root/'bad_png',all_selected_pngs=True)
            self.assertFalse(read(root/'bad_png/summary.json')['integrity_passed'])

    def test_changed_era5_rejected(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);c=fixture(root);cache_run(c);build_run(c)
            parquet=next((root/'era5').rglob('*.parquet'))
            parquet.write_bytes(parquet.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'ERA5 content changed'):
                audit(root/'config.json',root/'bad_era5',all_selected_pngs=True,recompute_era5=True)
            self.assertFalse(read(root/'bad_era5/summary.json')['integrity_passed'])

    def test_raw_values_compared_even_if_stage_checksum_updated(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);c=fixture(root);cache_run(c);build_run(c)
            stage=next((root/'dataset/staging').glob('*/input.npy'))
            a=np.load(stage,mmap_mode='r+');a[0,0,0,0]+=1;a.flush();del a
            record=read(stage.parent/'complete.json')
            record['sha256']['input.npy']=sha(stage)
            dump(stage.parent/'complete.json',record)
            with self.assertRaisesRegex(ValueError,'ERA5 raw/staging mismatch'):
                audit(root/'config.json',root/'bad_stage',recompute_era5=True)

    def test_launcher_records_success_and_refuses_reuse(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);c=fixture(root);cache_run(c);build_run(c)
            script=Path(__file__).resolve().parents[1]/'scripts/run_reference_audit.py'
            cmd=[sys.executable,str(script),'--contract-config',str(root/'config.json'),
                 '--output-dir',str(root/'execution')]
            check=subprocess.run(cmd+['--check-only'],capture_output=True,text=True)
            self.assertEqual(check.returncode,0,check.stderr)
            self.assertFalse((root/'execution').exists())
            run=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            self.assertTrue(read(root/'execution/execution.json')['integrity_passed'])
            again=subprocess.run(cmd,capture_output=True,text=True)
            self.assertNotEqual(again.returncode,0)

if __name__=='__main__':unittest.main()
