import argparse
import contextlib
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import prepare as p
from dataset import RadarDataset,masked_mse,residual_inputs,nearest_category
from pair_contract import CHANNELS,digest,dump

def fixture(root):
    policy=root/'policy';pair=root/'pair';cache=root/'cache'
    for d in (policy,pair,cache):d.mkdir()
    lat,lon=np.meshgrid(np.linspace(-23.5,-22.3,70),np.linspace(-44,-42.6,84),indexing='ij')
    np.savez(policy/'coverage.npz',lat=lat,lon=lon,observed_region_mask=np.ones((70,84),bool))
    np.savez(cache/'geometry.npz',lat=lat,lon=lon)
    dump(cache/'config.json',{'reference_contract':{'palette_dbz':[50,45,40,35,30,25,20]}})
    stamps=pd.to_datetime(['2011-01-17T02:00Z','2011-01-18T03:00Z','2021-06-01T00:00Z'],utc=True)
    single=[];pressure=[]
    for i,t in enumerate(stamps):
        shift=[0,10,10000][i]
        for la in (-24.,-22.):
            for lo in (-45.,-42.):
                single.append(dict(valid_time=t,latitude=la,longitude=lo,tcwv=shift+la,t2m=shift+280+la,u10=shift+lo,v10=shift+la))
                for level in (500,850):pressure.append(dict(valid_time=t,latitude=la,longitude=lo,pressure_level=level,
                                                          t=shift+la+270,r=shift+lo+70,u=shift+lo,v=shift+la))
    db=sqlite3.connect(pair/'era5_inventory_snapshot.sqlite')
    db.executescript('CREATE TABLE metrics(timestamp_utc TEXT,channel TEXT,path TEXT);CREATE TABLE files(path TEXT,fingerprint TEXT,status TEXT,details TEXT);')
    for kind,rows,variables in [('single',single,['tcwv','t2m','u10','v10']),('pressure',pressure,['t','r','u','v'])]:
        path=root/(kind+'.parquet');pd.DataFrame(rows).to_parquet(path,index=False)
        s=path.stat();details={'time_column':'valid_time','columns':list(rows[0]),'variable_columns':{v:v for v in variables},
                             'pressure_column':'pressure_level' if kind=='pressure' else None,'metadata':{}}
        db.execute('INSERT INTO files VALUES (?,?,?,?)',(str(path),f'{s.st_size}:{s.st_mtime_ns}','ok',json.dumps(details)))
        for t in stamps:
            for ch in CHANNELS:
                if ('_' in ch)==(kind=='pressure'):db.execute('INSERT INTO metrics VALUES (?,?,?)',(t.isoformat(),ch,str(path)))
    db.commit();db.close()
    pd.DataFrame({'timestamp_utc':stamps}).to_parquet(pair/'paired_manifest.parquet',index=False)
    dump(pair/'run_config.json',{'inventory_config':{'pressure_unit':'hPa'},'radar_config_sha256':digest(cache/'config.json'),
      'geometry_sha256':digest(cache/'geometry.npz'),'era5_snapshot_sha256':digest(pair/'era5_inventory_snapshot.sqlite')})
    dump(pair/'summary.json',{'complete':True,'provisional':False,'era5_error_files':0})
    records=[]
    for i,t in enumerate(stamps):
        raw=np.zeros((70,84),np.float32);mask=np.ones((70,84),bool);cl=np.ones((70,84),np.uint8)
        if i==1:
            raw[20:40,20:40]=30;cl[20:40,20:40]=2;raw[0,0]=np.nan;mask[0,0]=False;cl[0,0]=0
        rel=f'targets/{i}.npz';path=policy/rel;path.parent.mkdir(exist_ok=True)
        np.savez(path,target_code=raw,observation_mask=mask,observation_class=cl)
        records.append({'timestamp_utc':t,'split':'train' if i<2 else 'validation','locked':i==2,
          'policy_pair_candidate':True,'policy_train_candidate':i<2,'era5_technical_candidate':True,
          'in_archive_period':True,'local_filename_ambiguous':False,'observation_pixels':int(mask.sum()),
          'target_path':rel,'target_sha256':digest(path)})
    pd.DataFrame(records).to_parquet(policy/'policy_manifest.parquet',index=False)
    dump(policy/'policy_config.json',{'pairing_dir':str(pair),'cache_dir':str(cache),
      'pairing_run_config_sha256':digest(pair/'run_config.json'),'paired_manifest_sha256':digest(pair/'paired_manifest.parquet')})
    dump(policy/'summary.json',{'version':'14n-producer-policy-v1','complete':True,'object_error_hours':0,
      'coverage_state':'provisional_geometric_region','spatial_validity_confirmed':False,
      'artifact_sha256':{n:digest(policy/n) for n in ['policy_manifest.parquet','coverage.npz','policy_config.json']}})
    return argparse.Namespace(policy_dir=str(policy),output_dir=str(root/'prepared'),pilot_hours=0,resume=False)

class ContractTests(unittest.TestCase):
    def run_quiet(self,a):
        with contextlib.redirect_stdout(io.StringIO()):p.run(a)

    def test_full_prepare_train_only_stats_and_loader(self):
        with tempfile.TemporaryDirectory() as td:
            a=fixture(Path(td));self.run_quiet(a);out=Path(a.output_dir)
            s=json.loads((out/'summary.json').read_text());self.assertEqual(s['patches'],144)
            stats=json.loads((out/'normalization.json').read_text())
            self.assertEqual(stats['count_per_channel'],2*70*84)
            self.assertAlmostEqual(stats['mean'][0],5+(-23.5-22.3)/2,places=5)
            ds=RadarDataset(out);self.assertEqual(len(ds),96)
            b=ds[48];self.assertEqual(tuple(b['conditioning'].shape),(12,32,32))
            self.assertFalse(b['mask'][0,0,0]);self.assertEqual(b['target'][0,0,0],0)
            self.assertEqual(int((b['target']>0).sum()),12*12)
            self.assertEqual(len(RadarDataset(out,fully_supported=True)),95)
            self.assertEqual(len(RadarDataset(out,'validation')),48)
            self.assertFalse(s['production_ready']);self.assertFalse(s['corrdiff_native_tested'])

    def test_resume_integrity_and_changed_source(self):
        with tempfile.TemporaryDirectory() as td:
            a=fixture(Path(td));self.run_quiet(a);out=Path(a.output_dir)
            with self.assertRaisesRegex(ValueError,'exists'):p.run(a)
            a.resume=True
            with self.assertRaisesRegex(ValueError,'already complete'):p.run(a)
            # Simulate interruption after committed monthly shards, before final report.
            dump(out/'summary.json',{'complete':False});before=digest(out/'shards/2011-01/conditioning.npy')
            self.run_quiet(a);self.assertEqual(before,digest(out/'shards/2011-01/conditioning.npy'))
            dump(out/'summary.json',{'complete':False})
            with (Path(td)/'single.parquet').open('ab') as f:f.write(b'changed')
            with self.assertRaisesRegex(ValueError,'Changed ERA5'):self.run_quiet(a)

    def test_corrupt_shard_refused(self):
        with tempfile.TemporaryDirectory() as td:
            a=fixture(Path(td));self.run_quiet(a);out=Path(a.output_dir)
            ds=RadarDataset(out)
            with (out/'shards/2011-01/target_code.npy').open('ab') as f:f.write(b'changed')
            with self.assertRaisesRegex(ValueError,'checksum'):ds[0]

    def test_pilot_never_uses_validation(self):
        with tempfile.TemporaryDirectory() as td:
            a=fixture(Path(td));a.pilot_hours=1;self.run_quiet(a)
            s=json.loads((Path(a.output_dir)/'summary.json').read_text())
            self.assertEqual(s['split_hours'],{'train':1});self.assertTrue(s['pilot_only'])

    def test_masked_loss_and_residual_unknown_invariance(self):
        pred=torch.tensor([[[[1.,999.]]]],requires_grad=True)
        y=torch.tensor([[[[0.,float('nan')]]]]);m=torch.tensor([[[[True,False]]]])
        loss=masked_mse(pred,y,m);loss.backward()
        self.assertEqual(loss.item(),1);self.assertEqual(pred.grad[0,0,0,1],0)
        r,n=residual_inputs(y,torch.zeros_like(y),m,torch.ones(1,1,1,1),torch.ones_like(y))
        self.assertEqual(r[0,0,0,1],0);self.assertEqual(n[0,0,0,1],0)
        self.assertTrue(torch.isfinite(n).all())
        with self.assertRaises(ValueError):masked_mse(pred,y,torch.zeros_like(m))

    def test_stride8_full_coverage(self):
        ys=p.patch_starts(70,32,8);xs=p.patch_starts(84,32,8)
        self.assertEqual(ys,[0,8,16,24,32,38])
        self.assertEqual(xs,[0,8,16,24,32,40,48,52])
        cover=np.zeros((70,84),int)
        for y in ys:
            for x in xs:cover[y:y+32,x:x+32]+=1
        self.assertTrue((cover>0).all());self.assertEqual(len(ys)*len(xs),48)

    def test_reindex_preserves_fields_and_normalization(self):
        import reindex
        with tempfile.TemporaryDirectory() as td:
            a=fixture(Path(td));self.run_quiet(a);source=Path(a.output_dir)
            h=digest(source/'shards/2011-01/conditioning.npy')
            out=Path(td)/'reindexed'
            with contextlib.redirect_stdout(io.StringIO()):
                reindex.run(argparse.Namespace(prepared_dir=str(source),output_dir=str(out)))
            self.assertEqual(digest(out/'normalization.json'),digest(source/'normalization.json'))
            self.assertEqual(h,digest(out/'shards/2011-01/conditioning.npy'))
            self.assertTrue((out/'shards').is_symlink())
            self.assertEqual(len(RadarDataset(out)),96)

    def test_numeric_decode_and_stats(self):
        np.testing.assert_array_equal(nearest_category([-2,0,9,10,11,20,22.5,51]),[0,0,0,0,20,20,20,50])
        fields=np.arange(12*70*84,dtype='float32').reshape(12,70,84)
        n,mu,m2=p.stats_merge(0,np.zeros(12),np.zeros(12),fields)
        n,mu,m2=p.stats_merge(n,mu,m2,fields+4)
        expected=np.stack([fields,fields+4])
        np.testing.assert_allclose(mu,expected.mean(axis=(0,2,3),dtype=float))
        np.testing.assert_allclose(np.sqrt(m2/n),expected.std(axis=(0,2,3),dtype=float))

if __name__=='__main__':unittest.main()
