"""Small end-to-end fixtures: no CEFET files or GPU required."""
import contextlib
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import numpy as np
import pandas as pd
from PIL import Image
from src.corrdiff.builder.contract_common import *
from src.corrdiff.builder.reference_geometry import geometry
from src.corrdiff.builder.contract_cache import run as cache_run,target
from src.corrdiff.builder.contract_zarr import run as build_run
from src.corrdiff.builder.contract_dataset import ReferenceDataset
from src.corrdiff.builder.contract_era5 import regrid


def fixture(root):
    g=geometry();np.savez(root/'geometry.npz',**g)
    np.savez(root/'coverage.npz',lat=g['lat'],lon=g['lon'],observed_region_mask=g['geometric_mask'])
    dump(root/'spatial.json',dict(no_echo_code=0,grid_shape=[70,84],source_palette_dbz=[20,25,30,35,40,45,50],target_normalization='target_code / 50',patch_stride=8,
        policy_artifacts={'coverage.npz':sha(root/'coverage.npz')},coverage_state='provisional_geometric_region',spatial_validity_confirmed=False))
    td=root/'temporal';td.mkdir();pair=root/'pairing';pair.mkdir()
    tc=dict(strategy='nearest2',aggregation='none',no_echo_code=0,tolerance_seconds=120,source_timezone='America/Sao_Paulo')
    dump(td/'temporal_contract.json',tc)
    stamps=pd.to_datetime(['2011-01-17T03:00Z','2011-01-17T04:00Z','2012-02-25T05:00Z','2021-02-01T03:00Z','2023-02-01T03:00Z','2024-02-01T03:00Z'],utc=True)
    db=sqlite3.connect(pair/'era5_inventory_snapshot.sqlite')
    db.execute('CREATE TABLE files(path TEXT PRIMARY KEY, fingerprint TEXT, status TEXT,details TEXT)')
    db.execute('CREATE TABLE metrics(path TEXT,timestamp_utc TEXT,channel TEXT)')
    dfrows=[]
    for i,t in enumerate(stamps):
        source=t+pd.Timedelta(minutes=-2 if i==1 else 0);rel=source.tz_convert('America/Sao_Paulo').strftime('%Y/%m/%d/%Y_%m_%d_%H_%M.png')
        p=root/'radar'/rel;p.parent.mkdir(parents=True,exist_ok=True)
        rgba=np.zeros((654,656,4),np.uint8)
        if i%2:
            gm=g['geometric_mask'];r,c=np.argwhere(gm)[200]
            rgba[g['py'][r,c],g['px'][r,c]]=[0,167,12,255]
            r,c=np.argwhere(gm)[1000];rgba[g['py'][r,c],g['px'][r,c]]=[1,2,3,255]
        Image.fromarray(rgba).save(p)
        dfrows.append(dict(timestamp_utc=t,source_utc=source,year=t.year,split=expected_split(t),strategy='nearest2',era5_ready=True,paired=True,radar_found=True,
            offset_seconds=int((source-t).total_seconds()),path=rel,sha=sha(p),policy_status='valid_echo_and_no_echo' if i%2 else 'valid_transparent_no_echo'))
    # One file per month/type, matching the actual ETL partition structure.
    for month,ts in pd.Series(stamps).groupby(stamps.strftime('%Y%m')):
        for kind,variables,levels in [('single',['tcwv','t2m','u10','v10'],[None]),('pressure',['t','r','u','v'],[500,850])]:
            rows=[]
            for t in ts:
                shift=10000 if t.year>=2021 else 0
                for lev in levels:
                    for lat in [-24.,-23.,-22.]:
                        for lon in [-45.,-44.,-43.,-42.]:
                            d=dict(valid_time=t,latitude=lat,longitude=lon)
                            if lev:d['pressure_level']=lev
                            for j,v in enumerate(variables):d[v]=lat*2+lon*3+j+shift+t.hour
                            rows.append(d)
            data=pd.DataFrame(rows);p=root/'era5'/kind/f'year={ts.iloc[0].year}'/f'month={ts.iloc[0].month}'/'part.parquet'
            p.parent.mkdir(parents=True,exist_ok=True);data.to_parquet(p,index=False)
            stat=p.stat();details=dict(columns=list(data.columns),time_column='valid_time',variable_columns={v:v for v in variables},pressure_column='pressure_level' if kind=='pressure' else None)
            db.execute('INSERT INTO files VALUES (?,?,?,?)',(str(p),f'{stat.st_size}:{stat.st_mtime_ns}','ok',json.dumps(details)))
            for t in ts:
                for lev in levels:
                    for v in variables:db.execute('INSERT INTO metrics VALUES (?,?,?)',(str(p),t.isoformat(),v if lev is None else f'{v}_{lev}'))
    db.commit();db.close();dump(pair/'run_config.json',dict(inventory_config={'pressure_unit':'hPa'},era5_snapshot_sha256=sha(pair/'era5_inventory_snapshot.sqlite')))
    pd.DataFrame(dfrows).to_parquet(td/'selected_manifest.parquet',index=False)
    dump(td/'summary.json',dict(complete=True,temporal_rule_finalized=True,selected_hours=len(stamps),artifact_sha256={n:sha(td/n) for n in ['temporal_contract.json','selected_manifest.parquet']}))
    c=dict(version=VERSION,project_root=str(root),temporal_dir='temporal',geometry_file='geometry.npz',coverage_file='coverage.npz',spatial_contract_file='spatial.json',
           radar_dir='radar',reuse_14j_dir=None,radar_cache_dir='cache',pairing_dir='pairing',era5_root='era5',output_dir='dataset',workers=2)
    dump(root/'config.json',c)
    return load_config(root/'config.json')


class ReferenceTests(unittest.TestCase):
    def test_end_to_end_and_resume(self):
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp);c=fixture(root);cache_run(c);build_run(c)
            s=read(root/'dataset/summary.json');self.assertTrue(s['complete']);self.assertEqual(s['selected_hours'],6)
            n=read(root/'dataset/normalization.json');self.assertEqual(n['train_hours'],3)
            self.assertTrue(all(v<0 for v in n['input_mean'])) # validation +10000 must not enter train stats
            ds=ReferenceDataset(root/'dataset');self.assertEqual(len(ds),144)
            a=ds[0];self.assertTrue(np.isfinite(a['input']).all());self.assertTrue((a['target']==0).all());self.assertTrue(a['mask'].any())
            z=ds.z;self.assertEqual(int(z['offset_seconds'][48]),-120)
            self.assertTrue(np.isclose(z['target'][:],0.4).any());self.assertTrue((~z['mask'][:]).any())
            import xarray as xr
            with xr.open_zarr(root/'dataset/train.zarr',chunks=None) as opened:
                self.assertEqual(opened.input.shape,(144,12,32,32))
                self.assertEqual(opened.target.shape,(144,1,32,32))
                self.assertEqual(opened['mask'].dtype,bool)
                self.assertTrue((opened.target.isel(sample=0).values==0).all())
            before=read(root/'dataset/train.complete.json');cache_run(c,resume=True);build_run(c,resume=True)
            self.assertEqual(before,read(root/'dataset/train.complete.json'))
            # A partial write is safely rebuilt from hourly staging, not accepted as complete.
            import shutil
            shutil.rmtree(root/'dataset/test_primary.zarr');(root/'dataset/test_primary.complete.json').unlink()
            (root/'dataset/.test_primary.zarr.build').mkdir();build_run(c,resume=True)
            self.assertEqual(read(root/'dataset/test_primary.complete.json')['patches'],48)
            # Content-addressed cache damage is detected before a rebuild.
            with sqlite3.connect(root/'cache/index.sqlite') as db:rel=db.execute('SELECT path FROM objects LIMIT 1').fetchone()[0]
            (root/'cache'/rel).write_bytes(b'broken')
            with self.assertRaises(ValueError):cache_run(c,resume=True)
    def test_reuse_14j_and_portable_era5(self):
        import shutil
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp);c=fixture(root);old=root/'old14j';old.mkdir()
            shutil.copyfile(root/'geometry.npz',old/'geometry.npz');dump(old/'summary.json',{'complete':True})
            db=sqlite3.connect(old/'manifest.sqlite');db.execute('CREATE TABLE sources(sha256 TEXT,decoded_path TEXT,decoded_sha256 TEXT)')
            df,g,_=inputs(c)
            for row in df.drop_duplicates('sha').itertuples(index=False):
                rgba=np.asarray(Image.open(root/'radar'/row.path));a=rgba[:,:,3]
                native=np.full(a.shape,255,np.uint8);native[a==0]=0
                for i,rgb in enumerate(REFERENCE['palette_rgb'],1):native[(a==255)&(rgba[:,:,:3]==rgb).all(2)]=i
                gm=g['geometric_mask'];classes=np.full((70,84),253,np.uint8);alpha=np.zeros((70,84),np.uint8)
                classes[gm]=native[g['py'][gm],g['px'][gm]];alpha[gm]=a[g['py'][gm],g['px'][gm]]
                name=row.sha+'.npz';np.savez_compressed(old/name,grid_class=classes,grid_alpha=alpha,native_class=native,metadata_json=np.array('{}'))
                db.execute('INSERT INTO sources VALUES (?,?,?)',(row.sha,name,sha(old/name)))
            db.commit();db.close();c['reuse_14j_dir']=str(old)
            shutil.rmtree(root/'radar');cache_run(c);build_run(c)
            with sqlite3.connect(root/'cache/index.sqlite') as db:self.assertEqual(db.execute('SELECT DISTINCT origin FROM objects').fetchall(),[('14j',)])
            copied=root/'copied_era5';shutil.copytree(root/'era5',copied,copy_function=shutil.copyfile)
            c['era5_root']=str(copied);c['output_dir']=str(root/'portable_output')
            with self.assertRaisesRegex(ValueError,'metadata changed'):build_run(c)
            c['era5_identity_file']=str(root/'dataset/era5_identity.json');build_run(c,resume=True)
            self.assertEqual(read(root/'dataset/train.complete.json'),read(root/'portable_output/train.complete.json'))

    def test_masks_and_patch_edges(self):
        g=geometry();g['support']=g['geometric_mask'];classes=np.zeros((70,84),np.uint8);alpha=np.zeros_like(classes)
        classes[3,3]=255;alpha[3,3]=255;classes[4,4]=7;alpha[4,4]=255
        code,mask=target(classes,alpha,g,'valid_echo_and_no_echo',1)
        self.assertFalse(mask[3,3]);self.assertTrue(np.isnan(code[3,3]));self.assertEqual(code[4,4],20)
        with self.assertRaises(ValueError):target(classes,alpha,g,'reviewed_map_no_echo',1)
        self.assertEqual(starts(70),[0,8,16,24,32,38]);self.assertEqual(starts(84),[0,8,16,24,32,40,48,52])
    def test_interpolation_and_no_extrapolation(self):
        rows=pd.DataFrame([(a,b,2*a+3*b) for a in [0,1] for b in [0,1]],columns=['lat','lon','value'])
        self.assertAlmostEqual(float(regrid(rows,np.array([[.25]]),np.array([[.5]]))[0,0]),2.)
        with self.assertRaises(ValueError):regrid(rows,np.array([[2.]]),np.array([[.5]]))
        with self.assertRaises(ValueError):regrid(rows.iloc[:-1],np.array([[.25]]),np.array([[.5]]))
    def test_tampered_manifest_and_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);c=fixture(root);td=root/'temporal'
            df=pd.read_parquet(td/'selected_manifest.parquet');df.loc[0,'offset_seconds']=300;df.to_parquet(td/'selected_manifest.parquet',index=False)
            with self.assertRaisesRegex(ValueError,'checksum'):inputs(c)
            raw=read(root/'config.json');raw['output_dir']='radar';dump(root/'config.json',raw)
            with self.assertRaisesRegex(ValueError,'separate'):load_config(root/'config.json')

if __name__=='__main__':unittest.main()
