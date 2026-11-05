import argparse
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import apply_policy as p
from pair_contract import dump,digest,CHANNELS


def fixture(root):
    """Small 14j/14k-shaped fixture with real Parquet inputs and a SQLite ERA5 index."""
    cache=root/'cache';pair=root/'pair';cache.mkdir();pair.mkdir()
    shape=(70,84);geom=np.ones(shape,bool)
    lat,lon=np.meshgrid(np.linspace(-23.5,-22.3,70),np.linspace(-44,-42.6,84),indexing='ij')
    np.savez(cache/'geometry.npz',lat=lat,lon=lon,geometric_mask=geom)
    dump(cache/'config.json',dict(version='14j-intermediate-cache-v1',cadence_minutes=60,reference_contract={'palette_dbz':p.PALETTE}))
    dump(cache/'summary.json',dict(complete=True))
    rdb=sqlite3.connect(pair/'radar_manifest_snapshot.sqlite')
    rdb.executescript('CREATE TABLE frames(timestamp_utc TEXT,source_sha256 TEXT,status TEXT,local_filename_ambiguous INTEGER); CREATE TABLE sources(sha256 TEXT,decoded_path TEXT,decoded_sha256 TEXT);')
    edb=sqlite3.connect(pair/'era5_inventory_snapshot.sqlite')
    edb.executescript('CREATE TABLE metrics(timestamp_utc TEXT,channel TEXT,path TEXT); CREATE TABLE files(path TEXT,fingerprint TEXT,status TEXT,details TEXT);')
    statuslist=['missing_png','transparent_ambiguous','palette_sampled_observation_validity_unconfirmed','map_screen_review','no_exact_palette_review','palette_in_png_not_sampled_in_corrected_grid','transparent_ambiguous']
    stamps=pd.to_datetime(['2011-01-17T01:00Z','2011-01-17T02:00Z','2012-12-11T19:00Z','2012-02-25T06:00Z','2013-01-01T00:00Z','2014-01-01T00:00Z','2023-06-01T00:00Z'],utc=True)
    frame_rows=[]
    for i,(stamp,status) in enumerate(zip(stamps,statuslist)):
        sha=f'{i:064x}' if status!='missing_png' else None
        rel=f'decoded/{i}.npz' if sha else None;checksum=None
        if sha:
            classes=np.zeros(shape,np.uint8);alpha=np.zeros(shape,np.uint8);grid=np.full(shape,np.nan,np.float32)
            palcount=0
            if i==2:
                classes[20:45,20:45]=5;alpha[20:45,20:45]=255;grid[20:45,20:45]=30;palcount=625
                classes[0,0]=255;alpha[0,0]=255
                classes[0,1]=254;alpha[0,1]=100
            if i in (3,4):classes[:]=255;alpha[:]=255
            if i==5:palcount=3 # colors in native image, none selected on grid
            path=cache/rel;path.parent.mkdir(exist_ok=True)
            meta=dict(status=status,source_size=[656,654],source_frames=1,exact_palette_pixels=palcount)
            np.savez_compressed(path,grid_class=classes,grid_alpha=alpha,grid_dbz_palette=grid,
                grid_dbz_review=grid,geometric_mask=geom,metadata_json=np.array(json.dumps(meta)))
            checksum=digest(path)
            rdb.execute('INSERT INTO sources VALUES (?,?,?)',(sha,rel,checksum))
        rdb.execute('INSERT INTO frames VALUES (?,?,?,?)',(stamp.isoformat(),sha,status,0))
        split='test_primary' if stamp.year==2023 else 'train'
        frame_rows.append(dict(timestamp_utc=stamp,timestamp_local=stamp.tz_convert('America/Sao_Paulo').isoformat(),
            source_path=f'/raw/{stamp}.png',source_sha256=sha,decoded_path=rel,decoded_sha256=checksum,status=status,error=None,
            local_filename_ambiguous=False,split=split,locked=split!='train',era5_complete_12=True,era5_technical_candidate=True,
            paired_palette_candidate=i==2,train_smoke_candidate=i==2,**{ch:'finite_rectangular_cover' for ch in CHANNELS}))
    rdb.commit();rdb.close()
    single=[];pressure=[]
    for stamp in stamps:
        for la in (-24.,-22.):
            for lo in (-45.,-42.):
                single.append(dict(valid_time=stamp,latitude=la,longitude=lo,tcwv=20+la,t2m=280+la,u10=lo,v10=la))
                for lev in (850,500):pressure.append(dict(valid_time=stamp,latitude=la,longitude=lo,pressure_level=lev,t=270+la,r=70+lo,u=lo,v=la))
    for kind,rows,variables in [('single',single,{'tcwv':'tcwv','t2m':'t2m','u10':'u10','v10':'v10'}),('pressure',pressure,{'t':'t','r':'r','u':'u','v':'v'})]:
        path=root/(kind+'.parquet');pd.DataFrame(rows).to_parquet(path)
        st=path.stat();details=dict(time_column='valid_time',columns=list(rows[0]),variable_columns=variables,pressure_column='pressure_level',metadata={})
        edb.execute('INSERT INTO files VALUES (?,?,?,?)',(str(path),f'{st.st_size}:{st.st_mtime_ns}','ok',json.dumps(details)))
        for stamp in stamps:
            for ch in CHANNELS:
                if ('_' in ch)==(kind=='pressure'):edb.execute('INSERT INTO metrics VALUES (?,?,?)',(stamp.isoformat(),ch,str(path)))
    edb.commit();edb.close()
    pd.DataFrame(frame_rows).to_parquet(pair/'paired_manifest.parquet',index=False)
    dump(pair/'summary.json',dict(version='14k-pairing-v1',complete=True,provisional=False,radar_snapshot_complete=True,era5_error_files=0))
    dump(pair/'run_config.json',dict(args={'purge_hours':48},inventory_config={'pressure_unit':'hPa'},
        radar_config_sha256=digest(cache/'config.json'),geometry_sha256=digest(cache/'geometry.npz'),
        radar_snapshot_sha256=digest(pair/'radar_manifest_snapshot.sqlite'),era5_snapshot_sha256=digest(pair/'era5_inventory_snapshot.sqlite')))
    return argparse.Namespace(radar_cache_dir=str(cache),pairing_dir=str(pair),output_dir=str(root/'policy'),archive_start_local='2011-01-17',
        coverage_mask=None,coverage_metadata=None,map_review=None,resume=False)


class Tests(unittest.TestCase):
    def test_pixel_semantics(self):
        cl=np.array([[0,5,254,255,0]],np.uint8);alpha=np.array([[0,255,100,255,0]],np.uint8)
        geom=np.ones(cl.shape,bool);support=geom.copy();support[0,-1]=False
        a=p.interpret(cl,alpha,geom,support,'palette_sampled_observation_validity_unconfirmed',1)
        np.testing.assert_array_equal(a['observation_class'],[[1,2,0,0,0]])
        self.assertEqual(a['target_code'][0,0],0);self.assertTrue(np.isnan(a['displayed_dbz'][0,0]))
        self.assertEqual(a['displayed_dbz'][0,1],30)
        self.assertTrue(np.isnan(a['target_code'][0,2:]).all())
        self.assertIsNone(p.interpret(cl,alpha,geom,support,'no_exact_palette_review',0))
        self.assertIsNone(p.interpret(cl,alpha,geom,support,'map_screen_review',0))
        with self.assertRaisesRegex(ValueError,'palette pixels'):p.interpret(cl,alpha,geom,support,'map_screen_review',1,True)

    def test_policy_resume_period_and_map(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);args=fixture(root)
            inputs={str(x):digest(x) for x in root.rglob('*') if x.is_file()}
            r=p.run(args)
            self.assertEqual(r['archive_boundary_utc'],'2011-01-17T02:00:00+00:00')
            self.assertEqual(r['outside_archive_period_hours'],1)
            self.assertEqual(r['policy_train_candidate_hours'],3)
            self.assertEqual(r['paired_transparent_hours'],2)
            df=pd.read_parquet(root/'policy/policy_manifest.parquet')
            self.assertFalse(df.iloc[0].policy_pair_candidate)
            self.assertFalse(df.loc[df.split.eq('test_primary'),'policy_train_candidate'].any())
            self.assertEqual(df.loc[df.status.eq('map_screen_review'),'observation_pixels'].iloc[0],0)
            self.assertTrue(all(digest(Path(k))==v for k,v in inputs.items()))
            args.resume=True;r2=p.run(args);self.assertEqual(r['policy_pair_candidate_hours'],r2['policy_pair_candidate_hours'])
            args.archive_start_local='2011-01-18'
            with self.assertRaisesRegex(ValueError,'before supplied'):p.run(args)
            args.archive_start_local='2011-01-17'
            review=root/'review.csv'
            pd.DataFrame([dict(source_sha256=f'{3:064x}',decision='accept_no_echo',evidence='Fixture producer confirmation of valid map-only frame',reviewer='fixture',reviewed_at='2026-11-05')],columns=p.MAP_COLUMNS).to_csv(review,index=False)
            args.output_dir=str(root/'map_policy');args.resume=False;args.map_review=str(review)
            rm=p.run(args);self.assertEqual(rm['policy_train_candidate_hours'],4)
            self.assertEqual(rm['policy_status_counts']['reviewed_map_no_echo'],1)

    def test_external_coverage_and_corruption(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);args=fixture(root)
            with np.load(root/'cache/geometry.npz') as g:
                m=g['geometric_mask'].copy();m[:,42:]=False
                np.savez(root/'mask.npz',observed_region_mask=m,lat=g['lat'],lon=g['lon'])
            dump(root/'mask.json',dict(semantics='effective_observed_region',evidence='fixture-only',reviewer='tester',reviewed_at='2026-11-05'))
            args.coverage_mask=str(root/'mask.npz');args.coverage_metadata=str(root/'mask.json')
            r=p.run(args);self.assertTrue(r['spatial_validity_confirmed'])
            df=pd.read_parquet(root/'policy/policy_manifest.parquet')
            self.assertEqual(df.loc[df.status.eq('transparent_ambiguous'),'observation_pixels'].iloc[0],2940)
            (root/'cache/decoded/1.npz').write_bytes(b'corrupt')
            args.output_dir=str(root/'corrupt')
            r=p.run(args);self.assertEqual(r['object_error_hours'],1)
            df=pd.read_parquet(root/'corrupt/policy_manifest.parquet')
            self.assertFalse(df.loc[df.policy_status.eq('policy_object_error'),'policy_pair_candidate'].any())

    def test_smoke_end_to_end(self):
        import smoke_policy as s
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);args=fixture(root);p.run(args)
            a=argparse.Namespace(policy_dir=str(root/'policy'),output_dir=str(root/'smoke'),max_timestamps=4,steps=3,batch_size=2,device='cpu',seed=42)
            r=s.run(a)
            self.assertTrue(r['technical_smoke_passed']);self.assertGreater(r['selected_no_echo_only_hours'],0);self.assertGreater(r['selected_echo_hours'],0)
            self.assertFalse(r['production_ready']);self.assertFalse(r['corrdiff_native_tested'])
            self.assertEqual(r['patches_used'],36);self.assertEqual(r['empty_patches_skipped'],0)
            with np.load(root/'smoke/smoke_tensors.npz') as z:
                m=z['observation_support'];self.assertTrue(np.isnan(z['target_code'][~m]).all());self.assertTrue(np.isfinite(z['conditioning']).all())
            (root/'policy/coverage.npz').write_bytes(b'changed')
            a.output_dir=str(root/'smoke_bad')
            with self.assertRaisesRegex(ValueError,'artifact changed'):s.run(a)

    def test_ambiguous_local_excluded(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);args=fixture(root);pair=root/'pair'
            df=pd.read_parquet(pair/'paired_manifest.parquet')
            df.loc[1,'local_filename_ambiguous']=True
            df.to_parquet(pair/'paired_manifest.parquet',index=False)
            with sqlite3.connect(pair/'radar_manifest_snapshot.sqlite') as db:
                db.execute('UPDATE frames SET local_filename_ambiguous=1 WHERE source_sha256=?',(f'{1:064x}',))
            conf=json.loads((pair/'run_config.json').read_text())
            conf['radar_snapshot_sha256']=digest(pair/'radar_manifest_snapshot.sqlite');dump(pair/'run_config.json',conf)
            r=p.run(args)
            df=pd.read_parquet(root/'policy/policy_manifest.parquet')
            row=df[df.local_filename_ambiguous].iloc[0]
            self.assertGreater(row.observation_pixels,0)
            self.assertFalse(row.policy_pair_candidate)
            self.assertFalse(row.policy_train_candidate)

    def test_git_housekeeping_no_deletions(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);subprocess.run(['git','init','-q',str(root)],check=True)
            f=root/'analysis_outputs_restored/x/a.sqlite';f.parent.mkdir(parents=True);f.write_bytes(b'keep')
            subprocess.run(['git','-C',str(root),'add','.'],check=True)
            script=Path(p.__file__).with_name('repository_report.py')
            subprocess.run([sys.executable,str(script),'--repo-root',str(root),'--output-dir',str(root/'report'),'--apply-gitignore'],check=True,stdout=subprocess.DEVNULL)
            self.assertEqual(f.read_bytes(),b'keep')
            self.assertIn('a.sqlite',subprocess.check_output(['git','-C',str(root),'ls-files'],text=True))
            df=pd.read_csv(root/'report/tracked_artifacts_review.csv');self.assertEqual(len(df),1)

if __name__=='__main__':unittest.main()
