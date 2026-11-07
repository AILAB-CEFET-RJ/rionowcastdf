import argparse
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
import duckdb
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pair_inventory as m

class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.start=m.utc('2012-12-11T19:00Z');self.end=self.start
        self.engine=duckdb.connect();self.engine.execute("SET TimeZone='UTC'")
    def tearDown(self): self.engine.close();self.tmp.cleanup()
    def frame(self,pressure=False):
        rows=[]
        for level in ((850,500) if pressure else (None,)):
            for lat in (-24.,-22.):
                for lon in (315.,318.):
                    row={'valid_time':pd.Timestamp('2012-12-11T19:00'),'latitude':lat,'longitude':lon}
                    if pressure: row.update(pressure_level=level,t=280.,r=70.,u=2.,v=-1.)
                    else: row.update(tcwv=20.,t2m=290.,u10=1.,v10=-2.)
                    rows.append(row)
        return pd.DataFrame(rows)
    def test_time_and_longitude(self):
        df=self.frame();df.valid_time=pd.to_datetime(['2012-12-11T17:00:00-02:00']*len(df))
        path=self.root/'a.parquet';df.to_parquet(path)
        rows,meta=m.scan_file(self.engine,path,'single',self.start,self.end,'hPa')
        self.assertEqual(len(rows),4);self.assertEqual(rows[0][1],'2012-12-11T19:00:00+00:00')
        self.assertEqual(rows[0][-2:],(-45.,-42.))
    def test_duplicates_nonfinite_and_levels(self):
        df=self.frame(True);df.loc[0,'t']=np.inf
        df=pd.concat([df,df.iloc[[0]]],ignore_index=True)
        path=self.root/'b.parquet';df.to_parquet(path)
        rows,_=m.scan_file(self.engine,path,'pressure',self.start,self.end,'hPa')
        row=next(x for x in rows if x[2]=='t_850')
        self.assertEqual(row[3],5);self.assertEqual(row[4],4);self.assertEqual(row[7],3)
        df.pressure_level*=100;df.to_parquet(path)
        rows,_=m.scan_file(self.engine,path,'pressure',self.start,self.end,'Pa')
        self.assertEqual(len(rows),8)
    def test_ambiguous_schema_rejected(self):
        df=self.frame();df['time']=df.valid_time
        path=self.root/'a.parquet';df.to_parquet(path)
        with self.assertRaises(ValueError):m.scan_file(self.engine,path,'single',self.start,self.end,'hPa')
    def test_purge_edges(self):
        ts=pd.date_range('2020-12-29', '2021-01-04',freq='h',tz='UTC')
        _,s,_=m.split_labels(ts,48)
        self.assertEqual(sum(s=='excluded_purge'),96)
        self.assertEqual(s[ts.get_loc('2020-12-29T23:00Z')],'train')
        self.assertEqual(s[ts.get_loc('2021-01-03T00:00Z')],'validation')
    def test_end_to_end_resume_pair_missing_and_overlap(self):
        single=self.root/'single';pressure=self.root/'pressure';single.mkdir();pressure.mkdir()
        self.frame().to_parquet(single/'s.parquet');self.frame(True).to_parquet(pressure/'p.parquet')
        inv=self.root/'inventory';inv.mkdir();(inv/'.writer.lock').touch()
        args=argparse.Namespace(single_dir=str(single),pressure_dir=str(pressure),pressure_unit='hPa',rescan=False)
        self.assertEqual(m.index_era5(args,inv,self.start,self.end)['metrics_rows'],12)
        self.assertEqual(m.index_era5(args,inv,self.start,self.end)['metrics_rows'],12)
        cache=self.root/'cache';cache.mkdir()
        m.dump(cache/'config.json',{'version':'14j-intermediate-cache-v1','cadence_minutes':60,'start_utc':self.start.isoformat(),'end_utc':self.end.isoformat()})
        m.dump(cache/'summary.json',{'complete':True})
        np.savez(cache/'geometry.npz',lat=np.array([[-23.5,-23.5],[-22.3,-22.3]]),lon=np.array([[-44,-42.6],[-44,-42.6]]))
        db=sqlite3.connect(cache/'manifest.sqlite')
        db.executescript('CREATE TABLE frames(timestamp_utc TEXT,timestamp_local TEXT,source_sha256 TEXT,local_filename_ambiguous INTEGER,status TEXT); CREATE TABLE sources(sha256 TEXT,decoded_path TEXT,decoded_sha256 TEXT);')
        db.execute('INSERT INTO frames VALUES (?,?,?,?,?)',(self.start.isoformat(),'2012-12-11T17:00:00-02:00','x',0,'palette_sampled_observation_validity_unconfirmed'))
        db.execute("INSERT INTO sources VALUES ('x','decoded/x.npz','y')");db.commit();db.close()
        out=self.root/'paired';out.mkdir()
        pargs=argparse.Namespace(radar_cache_dir=str(cache),inventory_dir=str(inv),allow_partial=False,purge_hours=48)
        summary=m.pair(pargs,out,self.start,self.end)
        self.assertEqual(summary['train_smoke_candidate_hours'],1)
        f=pd.read_parquet(out/'paired_manifest.parquet');self.assertFalse(f.training_ready.any())
        # Overlapping version must not silently win or multiply paired timestamps.
        self.frame().to_parquet(single/'duplicate.parquet')
        m.index_era5(args,inv,self.start,self.end)
        out2=self.root/'paired2';out2.mkdir()
        summary=m.pair(pargs,out2,self.start,self.end)
        self.assertEqual(summary['paired_palette_candidate_hours'],0)
        self.assertEqual(len(pd.read_parquet(out2/'paired_manifest.parquet')),1)
    def test_full_split_counts(self):
        ts=pd.date_range('2011-01-01','2024-12-31T23:00',freq='h',tz='UTC')
        _,labels,_=m.split_labels(ts,48)
        self.assertEqual(len(ts),122736)
        self.assertEqual(sum(labels=='excluded_purge'),288)
        self.assertEqual(sum(labels=='train'),87624)

    def test_missing_channels_and_corrupt_file(self):
        single=self.root/'single';pressure=self.root/'pressure';single.mkdir();pressure.mkdir()
        self.frame().drop(columns=['t2m','u10','v10']).to_parquet(single/'s.parquet')
        (pressure/'broken.parquet').write_bytes(b'not parquet')
        inv=self.root/'inventory';inv.mkdir();(inv/'.writer.lock').touch()
        args=argparse.Namespace(single_dir=str(single),pressure_dir=str(pressure),pressure_unit='hPa',rescan=False)
        summary=m.index_era5(args,inv,self.start,self.end)
        self.assertEqual(summary['error_files'],1);self.assertEqual(summary['metrics_rows'],1)

if __name__=='__main__':unittest.main()
