"""Integration fixture uses the actual frozen cache/Zarr builders, no CEFET data."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from test_reference_zarr import fixture,cache_run,build_run
from src.corrdiff.diagnostics.reference_phases.common import Context,split_for
from src.corrdiff.diagnostics.reference_phases.runner import main,selection,plan,FUNCTIONS,load_config
from src.corrdiff.diagnostics.reference_phases.structure import phase15
from src.corrdiff.diagnostics.reference_phases.training import StageBlocked

REPO=Path(__file__).resolve().parents[1]


class PhaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        with contextlib.redirect_stdout(io.StringIO()):
            cfg=fixture(cls.root);cache_run(cfg);build_run(cfg)
        summary=json.loads((cls.root/'dataset/summary.json').read_text())
        evidence=dict(complete=True,integrity_passed=True,all_patch_tensors_compared_to_staging=True,
            all_targets_compared_to_cache=True,all_selected_png_contents_decoded=True,all_selected_png_paths_checked=True,
            era5_raw_recomputed=True,normalization_recomputed=True,selected_hours=6,splits=summary['splits'])
        (cls.root/'audit.json').write_text(json.dumps(evidence))
        (cls.root/'reader.json').write_text(json.dumps(dict(complete=True,reader_check_passed=True,splits=summary['splits'])))
        c=json.loads((REPO/'configs/reference_phases.json').read_text())
        c.update(project_root=str(cls.root),dataset_dir='dataset',output_dir='results',integrity_summary='audit.json',reader_summary='reader.json',
            ablation_config_dir=str(REPO/'configs'),diagnostic_hours_per_split=3,baseline_train_hours=3,baseline_eval_hours=2,sample_pixels_per_hour=32,regime_clusters=2)
        cls.config=cls.root/'phases.json';cls.config.write_text(json.dumps(c))
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_01_all_diagnostics_and_resume(self):
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            result=main(['--config',str(self.config),'--phases','0-17'])
        self.assertEqual(result,0,[(p.name,p.read_text()) for p in (self.root/'results').glob('*/summary.json') if 'failed' in p.read_text()])
        d=pd.read_parquet(self.root/'results/00_structure/hour_metrics.parquet')
        self.assertEqual(len(d),6)
        # Unique grid denominators, never 48 * 32 * 32 per hour.
        self.assertTrue((d.valid_pixels<=5880).all())
        s=json.loads((self.root/'results/15_split_audit/summary.json').read_text())
        self.assertEqual(s['hours'],6)
        with contextlib.redirect_stdout(io.StringIO()):
            result=main(['--config',str(self.config),'--phases','0,15','--resume'])
        self.assertEqual(result,0)
        report=json.loads((self.root/'results/run_summary.json').read_text())
        self.assertTrue(all(x['status']=='reused' for x in report['results']))
        exp=json.loads((self.root/'results/17_ablations/experiment_manifest.json').read_text())
        self.assertEqual(len(exp['cases']),66)
        self.assertTrue(all(case['tests_locked'] for case in exp['cases']))

    def test_02_native_is_blocked_without_command(self):
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            result=main(['--config',str(self.config),'--phases','18','--resume'])
        self.assertEqual(result,3)
        r=json.loads((self.root/'results/18_native_integration/summary.json').read_text())
        self.assertFalse(r['complete']);self.assertEqual(r['status'],'blocked')

    def test_03_changed_configuration_refuses_resume(self):
        config=json.loads(self.config.read_text());config['seed']=7
        path=self.root/'changed.json';path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError,'changed'),contextlib.redirect_stdout(io.StringIO()):
            main(['--config',str(path),'--phases','15','--resume'])

    def test_selection_and_dependencies(self):
        self.assertEqual(selection('0,1-3,15'),[0,1,2,3,15])
        self.assertEqual(plan([0,15]),[15,0])
        self.assertEqual(plan([14]),[15,0,12,13,14])
        for value in ('19','-1','3-1'):
            with self.assertRaises(ValueError):selection(value)

    def test_purge_edges_and_source_offset(self):
        t=pd.to_datetime(['2020-12-29T23:00Z','2020-12-30T00:00Z','2021-01-02T23:00Z','2021-01-03T00:00Z'],utc=True)
        self.assertEqual(split_for((t.astype('int64')//10**9).to_numpy()).tolist(),['train','excluded_purge','excluded_purge','validation'])

    def test_04_wrong_patch_source_rejected(self):
        import zarr
        ctx=Context(load_config(self.config),self.root/'unused')
        z=zarr.open_group(str(self.root/'dataset/train.zarr'),mode='a')
        before=int(z['source_timestamps'][0]);z['source_timestamps'][0]=before+60
        output=self.root/'negative';output.mkdir()
        try:
            with self.assertRaisesRegex(ValueError,'metadata mismatch'):
                phase15(ctx,output)
        finally:z['source_timestamps'][0]=before


if __name__=='__main__':unittest.main()
