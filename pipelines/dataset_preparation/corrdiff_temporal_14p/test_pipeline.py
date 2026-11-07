import unittest
import tempfile
import sqlite3
import json
from pathlib import Path
from argparse import Namespace
from datetime import datetime,timezone
import pandas as pd
import pipeline as p

class PipelineTests(unittest.TestCase):
    def test_dst(self):
        self.assertIsNone(p.utc_seconds('2018-11-04T00:30:00'))
        self.assertIsNone(p.utc_seconds('2019-02-16T23:30:00'))
        self.assertEqual(p.utc_seconds('2012-02-25T03:00:00'),int(datetime(2012,2,25,5,tzinfo=timezone.utc).timestamp()))

    def test_policy(self):
        self.assertTrue(p.policy('transparent_ambiguous','x',{}, {})[1])
        self.assertFalse(p.policy('no_exact_palette_review','x',{}, {})[1])
        with self.assertRaises(ValueError):
            p.policy('map_screen_review','x',{'x':{'decision':'accept_no_echo'}},{'palette_pixels':1})
        self.assertTrue(p.policy('map_screen_review','x',{'x':{'decision':'accept_no_echo'}},{'palette_pixels':0})[1])

    def test_nearest(self):
        db=sqlite3.connect(':memory:')
        db.execute('CREATE TABLE candidates(sec,path,sha,policy,split)')
        db.executemany('INSERT INTO candidates VALUES(?,?,?,?,?)',[(880,'a','a','dry','train'),(1120,'b','b','dry','train'),(1000,'c','c','dry','validation')])
        self.assertEqual(p.select(db,1000,120,'train')[1],'a')
        self.assertIsNone(p.select(db,1000,0,'train'))
        self.assertIsNone(p.select(db,1000,119,'train'))

    def test_split_boundary(self):
        b=int(datetime(2023,1,1,tzinfo=timezone.utc).timestamp())
        self.assertEqual(p.split_at(b-48*3600),'excluded_purge')
        self.assertEqual(p.split_at(b+48*3600),'test_primary')
        self.assertEqual(p.split_at(b-48*3600-1),'validation')

    def test_review_evidence_required(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'review.csv'
            f.write_text('source_sha256,decision,evidence,reviewer,reviewed_at\n'+'a'*64+',accept_no_echo,,,\n')
            with self.assertRaises(ValueError): p.reviews(f)

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=root/'audit';k=root/'pair';a.mkdir();k.mkdir()
            (a/'.audit.lock').touch()
            db=sqlite3.connect(a/'audit.sqlite')
            db.executescript('CREATE TABLE objects(sha,status,metrics); CREATE TABLE files(path,sha,status,local_time,time_state,year,utc_time,seen_run);')
            # Exact transparent; nearby echo; unknown excluded; map grouped; duplicate time excluded.
            examples=[('2012-01-02T10:00:00','transparent_ambiguous'),('2012-01-02T10:58:00','palette_sampled_observation_validity_unconfirmed'),('2012-01-02T12:00:00','no_exact_palette_review'),('2012-01-02T13:00:00','map_screen_review'),('2012-01-02T14:00:00','transparent_ambiguous'),('2012-01-02T14:00:00','transparent_ambiguous')]
            for i,(local,status) in enumerate(examples):
                h=f'{i:064x}';db.execute('INSERT INTO objects VALUES(?,?,?)',(h,status,json.dumps({'palette_pixels':0})))
                db.execute('INSERT INTO files VALUES(?,?,?,?,?,?,?,?)',(f'{i}.png',h,status,local,'unique','2012',None,'run'))
            db.commit();db.close()
            p.dump(a/'summary.json',{'version':'14m-full-png-audit-v1','complete':True,'processed_files':len(examples),'run':'run'})
            p.dump(a/'config.json',{})
            p.dump(k/'summary.json',{'complete':True,'era5_error_files':0})
            pd.DataFrame({'timestamp_utc':['2012-01-02T12:00:00+00:00','2012-01-02T13:00:00+00:00','2012-01-02T14:00:00+00:00','2012-01-02T16:00:00+00:00'], 'split':['train']*4,'era5_technical_candidate':[True]*4}).to_parquet(k/'paired_manifest.parquet')
            before=p.digest(a/'audit.sqlite');out=root/'out'
            p.analyze(Namespace(audit_dir=a,pairing_dir=k,output_dir=out,reviews=None,radar_dir=None,max_samples=40))
            self.assertEqual(before,p.digest(a/'audit.sqlite'))
            r=pd.read_parquet(out/'temporal_comparison.parquet')
            self.assertEqual(int(r[r.strategy.eq('exact')].paired.sum()),1)
            self.assertEqual(int(r[r.strategy.eq('nearest2')].paired.sum()),2)
            cache=root/'cache';cache.mkdir()
            p.dump(cache/'summary.json',{'complete':True})
            with sqlite3.connect(cache/'manifest.sqlite') as c:
                c.execute('CREATE TABLE sources(sha256,decoded_path,decoded_sha256)')
                c.execute('INSERT INTO sources VALUES(?,?,?)',(f'{0:064x}','objects/zero.npz','z'))
            p.freeze(Namespace(analysis_dir=out,output_dir=root/'final',strategy='nearest2',rationale='Fixture decision',radar_cache_dir=cache))
            self.assertEqual(len(pd.read_csv(root/'final'/'additional_sources_to_decode.csv')),1)
            self.assertEqual(p.read_json(root/'final'/'summary.json')['selected_hours'],2)
            with self.assertRaises(FileExistsError):
                p.freeze(Namespace(analysis_dir=out,output_dir=root/'final',strategy='exact',rationale='Fixture',radar_cache_dir=None))
            with (out/'temporal_comparison.parquet').open('ab') as f:f.write(b'tamper')
            with self.assertRaises(ValueError):
                p.freeze(Namespace(analysis_dir=out,output_dir=root/'bad',strategy='exact',rationale='Fixture',radar_cache_dir=None))

if __name__=='__main__':unittest.main()
