import copy,importlib.util,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import pandas as pd
import zarr
from PIL import Image
from radar_qc import DEFAULT,classify,load_policy
from qc_dataset import split_codes

PACKAGE=Path(__file__).resolve().parents[1]

def run(script,*args,ok=True):
    proc=subprocess.run([sys.executable,str(PACKAGE/script),*map(str,args)],text=True,capture_output=True)
    if ok and proc.returncode:raise AssertionError(proc.stdout+'\n'+proc.stderr)
    return proc

class Pipeline(unittest.TestCase):
    def test_review_and_dry_policy(self):
        p=copy.deepcopy(DEFAULT);p['dry_rgba']=[[0,0,0,0]]
        with tempfile.TemporaryDirectory() as d:
            file=Path(d)/'policy.json';file.write_text(json.dumps(p))
            with self.assertRaises(ValueError):load_policy(file)
            p['dry_evidence']='Synthetic fixture only, not a real source contract';file.write_text(json.dumps(p))
            p=load_policy(file)
            rgba=np.zeros((2,2,4),np.uint8)
            grid,info=classify(rgba,p)
            self.assertTrue(np.isnan(grid).all())
            grid,info=classify(rgba,p,'accept');np.testing.assert_array_equal(grid,0)
    def test_purge(self):
        ts=pd.to_datetime(['2020-12-29T23:00Z','2020-12-30T00:00Z','2021-01-02T23:00Z','2021-01-03T00:00Z'],utc=True)
        np.testing.assert_array_equal(split_codes(ts),[1,0,0,2])
    def test_cache_sources(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);source=d/'radar'/'2011/01/01';source.mkdir(parents=True)
            for hour,color in [(0,[242,239,233,255]),(1,[227,6,5,255]),(2,[0,0,0,0])]:
                Image.fromarray(np.tile(np.array(color,np.uint8),(8,8,1))).save(source/f'2011_01_01_{hour:02d}_00.png')
            cache=d/'qc'
            run('14d_rebuild_radar_cache.py','--radar-dir',d/'radar','--output-dir',cache,
                '--begin','2011-01-01','--end','2011-01-01 03:00','--image-width',8,'--image-height',8,'--max-panels',0)
            s=json.loads((cache/'qc_summary.json').read_text())
            self.assertEqual(s['counts'],{'quarantine_map_review':1,'accepted_full':1,'quarantine_no_decodable_observation':1,'missing_png':1})
            self.assertEqual(len(list(cache.glob('*.npy'))),1)
            np.testing.assert_array_equal(np.load(cache/'20110101_01_00.npy'),45)
            again=run('14d_rebuild_radar_cache.py','--radar-dir',d/'radar','--output-dir',cache,ok=False)
            self.assertNotEqual(again.returncode,0)
            review=d/'review.csv'
            pd.DataFrame([dict(timestamp='2011-01-01T01:00:00Z',source_sha256='0'*64,action='accept',reason='fixture',reviewer='test')]).to_csv(review,index=False)
            bad=run('14d_rebuild_radar_cache.py','--radar-dir',d/'radar','--output-dir',d/'bad',
                '--begin','2011-01-01 01:00','--end','2011-01-01 01:00','--image-width',8,'--image-height',8,'--max-panels',0,'--reviews',review,ok=False)
            self.assertEqual(bad.returncode,2)
    def test_masked_metrics_and_locked_tests(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);dataset=d/'dataset';root=zarr.open_group(str(dataset/'train.zarr'),mode='w')
            ts=pd.to_datetime(['2019-06-01T00:00Z','2019-06-01T01:00Z','2020-12-30T00:00Z',
                '2021-06-01T00:00Z','2021-06-01T01:00Z','2023-06-01T00:00Z','2024-06-01T00:00Z'],utc=True)
            y=np.full((len(ts),1,4,4),np.log1p(np.float32(99)),np.float32)
            m=np.zeros_like(y);m[:,:,0,:2]=1;y[:,:,0,0]=np.log1p(np.float32(45));y[:,:,0,1]=0
            root.create_dataset('target',data=y);root.create_dataset('mask',data=m)
            x=np.random.default_rng(3).normal(size=(len(ts),2,4,4)).astype(np.float32)
            root.create_dataset('input',data=x);root.create_dataset('timestamps',data=ts.asi8)
            root.create_dataset('patch_row',data=np.zeros(len(ts),np.int16));root.create_dataset('patch_col',data=np.zeros(len(ts),np.int16))
            audit=d/'audit';run('14f_masked_audit.py','--dataset-dir',dataset,'--output-dir',audit,'--batch-size',2)
            metrics=pd.read_csv(audit/'yearly_masked_metrics.csv')
            self.assertTrue((metrics.pixel_fraction_ge_45==.5).all())
            np.testing.assert_array_equal(np.load(audit/'train_patch_indices.npy'),[0,1])
            np.testing.assert_array_equal(np.load(audit/'validation_patch_indices.npy'),[3,4])
            # Poison the held-out targets AFTER the descriptive audit. Baselines must not read them.
            root['target'][5:]=np.nan
            baseline=d/'baseline';run('16_masked_baselines.py','--dataset-dir',dataset,'--splits-dir',audit,'--output-dir',baseline,'--batch-size',2)
            score=pd.read_csv(baseline/'validation_metrics.csv')
            glob=score[(score.baseline=='train_global_mean_field')&(score.cohort=='common_training_support')&(score.threshold_dbz==45)].iloc[0]
            self.assertEqual(glob.n_pixels,4);self.assertEqual(glob.csi,1);self.assertAlmostEqual(glob.rmse_dbz,0)
            self.assertEqual(glob.fss_2km,1)
            persist=score[(score.baseline=='persistence_1h_radar_reference')&(score.threshold_dbz==45)].iloc[0]
            self.assertEqual(persist.n_pixels,2);self.assertEqual(persist.csi,1)
            norm=np.load(baseline/'train_input_normalization.npz')
            np.testing.assert_allclose(norm['mean'],x[:2].mean(axis=(0,2,3)),atol=1e-7)
            np.save(audit/'train_patch_indices.npy',np.array([0,1,5]))
            bad=run('16_masked_baselines.py','--dataset-dir',dataset,'--splits-dir',audit,'--output-dir',d/'badbaseline',ok=False)
            self.assertNotEqual(bad.returncode,0)
if __name__=='__main__':unittest.main()
