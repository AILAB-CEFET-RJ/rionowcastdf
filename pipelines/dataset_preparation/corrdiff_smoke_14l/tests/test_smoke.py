import argparse
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
import smoke_test as s
import pair_contract as k

class Tests(unittest.TestCase):
    def test_interpolation_linear_and_outside(self):
        rows=pd.DataFrame([(a,b,2*a+3*b+4) for a in (-24.,-22.) for b in (-45.,-42.)],columns=['lat','lon','value'])
        lat=np.array([[-23.5,-23.0]]);lon=np.array([[-44.,-43.]])
        np.testing.assert_allclose(s.regrid(rows,lat,lon),2*lat+3*lon+4)
        with self.assertRaises(ValueError):s.regrid(rows,np.array([[-25.]]),np.array([[-44.]]))
        with self.assertRaises(ValueError):s.regrid(pd.concat([rows,rows.iloc[[0]]]),lat,lon)
    def test_mask_nan_and_gradient_invariance(self):
        p=torch.tensor([[[[.1,.2],[.3,.4]]]],requires_grad=True)
        t=torch.tensor([[[[1.,float('nan')],[2.,999.]]]])
        m=torch.tensor([[[[True,False],[True,False]]]])
        loss=s.masked_mse(p,t,m);loss.backward();grad=p.grad.clone()
        self.assertTrue(torch.isfinite(loss));self.assertEqual(float(grad[~m].abs().sum()),0.)
        p2=p.detach().clone().requires_grad_(True);t2=t.clone();t2[~m]=-1e9
        loss2=s.masked_mse(p2,t2,m);loss2.backward()
        torch.testing.assert_close(loss,loss2);torch.testing.assert_close(grad,p2.grad)
        with self.assertRaises(ValueError):s.masked_mse(p,t,torch.zeros_like(m))
    def test_patch_layout(self):
        x=np.zeros((12,70,84),np.float32);y=np.ones((70,84),np.float32);mask=np.ones_like(y,bool)
        items,records=s.patches(x,y,mask)
        self.assertEqual(len(items),12)
        mask[:]=False
        items,records=s.patches(x,y,mask)
        self.assertEqual(len(records),12);self.assertEqual(len(items),0)
    def test_selection_rejects_test_and_ambiguous(self):
        df=pd.DataFrame({'timestamp_utc':['2012-01-01T00:00Z','2023-01-01T00:00Z','2012-02-01T00:00Z'],
          'split':['train','test_primary','train'],'locked':[False,True,False],
          'train_smoke_candidate':[True,True,True],'era5_technical_candidate':[True]*3,
          'local_filename_ambiguous':[False,False,True],'status':[s.RADAR_STATUS]*3})
        self.assertEqual(len(s.select_train(df,8)),1)
        df.loc[1,['split','locked']]=['train',False]
        with self.assertRaises(ValueError):s.select_train(df,8)
    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);single=root/'single';pressure=root/'pressure';single.mkdir();pressure.mkdir()
            stamp=pd.Timestamp('2012-12-11T19:00Z')
            rows=[];prs=[]
            for la in (-24.,-22.):
                for lo in (-45.,-42.):
                    rows.append(dict(valid_time=stamp,latitude=la,longitude=lo,tcwv=20+la,t2m=280+la,u10=lo,v10=la))
                    for lev in (850,500):prs.append(dict(valid_time=stamp,latitude=la,longitude=lo,pressure_level=lev,t=270+la,r=70+lo,u=lo,v=la))
            pd.DataFrame(rows).to_parquet(single/'s.parquet');pd.DataFrame(prs).to_parquet(pressure/'p.parquet')
            inv=root/'inv';inv.mkdir();(inv/'.writer.lock').touch()
            k.index_era5(argparse.Namespace(single_dir=str(single),pressure_dir=str(pressure),pressure_unit='hPa',rescan=False),inv,stamp,stamp)
            cache=root/'cache';cache.mkdir();(cache/'decoded').mkdir()
            lat,lon=np.meshgrid(np.linspace(-23.5,-22.3,70),np.linspace(-44,-42.6,84),indexing='ij')
            gm=np.ones((70,84),bool)
            np.savez(cache/'geometry.npz',lat=lat,lon=lon,geometric_mask=gm)
            target=np.full((70,84),np.nan,np.float32);target[20:50,20:60]=30
            cl=np.zeros((70,84),np.uint8);cl[np.isfinite(target)]=5
            alpha=np.zeros_like(cl);alpha[np.isfinite(target)]=255
            decoded=cache/'decoded/x.npz';np.savez(decoded,grid_dbz_review=target,grid_class=cl,grid_alpha=alpha,geometric_mask=gm)
            k.dump(cache/'config.json',{'version':'14j-intermediate-cache-v1','cadence_minutes':60,'start_utc':stamp.isoformat(),'end_utc':stamp.isoformat(),
              'reference_contract':{'palette_dbz':[50,45,40,35,30,25,20]}})
            k.dump(cache/'summary.json',{'complete':True})
            db=sqlite3.connect(cache/'manifest.sqlite')
            db.executescript('CREATE TABLE frames(timestamp_utc TEXT,timestamp_local TEXT,source_sha256 TEXT,local_filename_ambiguous INTEGER,status TEXT); CREATE TABLE sources(sha256 TEXT,decoded_path TEXT,decoded_sha256 TEXT);')
            db.execute('INSERT INTO frames VALUES (?,?,?,?,?)',(stamp.isoformat(),'2012-12-11T17:00:00-02:00','x',0,s.RADAR_STATUS))
            db.execute('INSERT INTO sources VALUES (?,?,?)',('x','decoded/x.npz',s.digest(decoded)));db.commit();db.close()
            pair=root/'pair';pair.mkdir()
            k.pair(argparse.Namespace(radar_cache_dir=str(cache),inventory_dir=str(inv),allow_partial=False,purge_hours=48),pair,stamp,stamp)
            args=argparse.Namespace(pairing_dir=str(pair),radar_cache_dir=str(cache),output_dir=str(root/'smoke'),allow_provisional=False,max_timestamps=1,steps=3,batch_size=2,device='cpu',seed=42)
            result=s.run(args)
            self.assertTrue(result['technical_smoke_passed']);self.assertFalse(result['production_ready']);self.assertFalse(result['corrdiff_native_tested'])
            with np.load(root/'smoke/smoke_tensors.npz') as z:
                self.assertTrue(np.isfinite(z['conditioning']).all())
                self.assertTrue(np.isnan(z['target_raw_dbz'][~z['palette_support']]).all())
            decoded.write_bytes(b'corrupt')
            args.output_dir=str(root/'smoke_bad')
            with self.assertRaisesRegex(ValueError,'checksum'):s.run(args)

if __name__=='__main__':unittest.main()
