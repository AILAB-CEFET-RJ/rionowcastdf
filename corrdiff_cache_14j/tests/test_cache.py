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
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import build_cache as c
from alignment_reference import geometry, utc_candidates
from datetime import datetime

class CacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.g=geometry()
    def png(self, a):
        b=io.BytesIO();Image.fromarray(a).save(b,format='PNG');return b.getvalue()
    def test_lossless_rgba_and_unknown(self):
        a=np.zeros((654,656,4),np.uint8);a[0,0]=[99,88,77,0];a[1,1]=[197,0,197,128];a[2,2]=[1,2,3,255]
        info, arrays=c.decode(self.png(a),self.g)
        np.testing.assert_array_equal(a,arrays['native_rgba'])
        self.assertEqual(arrays['native_class'][0,0],0)
        self.assertEqual(arrays['native_class'][1,1],254)
        self.assertEqual(arrays['native_class'][2,2],255)
        self.assertTrue(np.isnan(arrays['grid_dbz_palette']).all())
    def test_geographic_palette_and_map_review(self):
        a=np.zeros((654,656,4),np.uint8)
        y,x=self.g['py'][20,20],self.g['px'][20,20]
        a[y,x]=[227,6,5,255]
        _,v=c.decode(self.png(a),self.g)
        self.assertEqual(v['grid_dbz_palette'][20,20],45)
        a[:100]=[242,239,233,255]
        info,v=c.decode(self.png(a),self.g)
        self.assertEqual(info['status'],'map_screen_review')
        self.assertTrue(np.isnan(v['grid_dbz_review']).all())
        self.assertEqual(v['grid_dbz_palette'][20,20],45)
    def test_wrong_dimensions_preserved(self):
        a=np.zeros((3,4,4),np.uint8)
        info,v=c.decode(self.png(a),self.g)
        self.assertEqual(info['status'],'unexpected_dimensions_review')
        self.assertNotIn('grid_dbz_palette',v)
        np.testing.assert_array_equal(v['native_rgba'],a)
    def test_dst(self):
        self.assertEqual(c.png_path('/radar',c.parse_utc('2012-12-11T19:00:00Z')).name,'2012_12_11_17_00.png')
        self.assertEqual(len(utc_candidates(datetime(2018,2,17,23,30))),2)
        self.assertEqual(len(utc_candidates(datetime(2018,11,4,0,30))),0)
    def test_manifest_resume_changes_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            radar=Path(tmp)/'radar';radar.mkdir();out=Path(tmp)/'out'
            t=c.parse_utc('2012-12-11T19:00:00Z')
            a=np.zeros((654,656,4),np.uint8);data=self.png(a)
            for hour in (0,1):
                p=c.png_path(radar,t+c.timedelta(hours=hour));p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
            args=argparse.Namespace(radar_dir=str(radar),output_dir=str(out),start_utc=t.isoformat(),end_utc='2012-12-11T21:00:00Z',cadence_minutes=60,resume=False,progress_every=10)
            with contextlib.redirect_stdout(io.StringIO()): self.assertEqual(c.run(args),0)
            s=json.loads((out/'summary.json').read_text())
            self.assertTrue(s['complete']);self.assertEqual(s['counts']['missing_png'],1);self.assertEqual(s['unique_source_objects'],1)
            args.resume=True
            with contextlib.redirect_stdout(io.StringIO()): c.run(args)
            db=sqlite3.connect(out/'manifest.sqlite')
            self.assertEqual(db.execute('SELECT COUNT(*) FROM frames').fetchone()[0],3)
            p=c.png_path(radar,t);p.write_bytes(b'not a png')
            with contextlib.redirect_stdout(io.StringIO()): self.assertEqual(c.run(args),2)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM changes').fetchone()[0],1)
            row=db.execute("SELECT original_path FROM sources WHERE status='decode_error'").fetchone()
            self.assertEqual((out/row[0]).read_bytes(),b'not a png')
            decoded=db.execute('SELECT decoded_path FROM sources WHERE decoded_path IS NOT NULL').fetchone()[0]
            (out/decoded).write_bytes(b'corrupt')
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError,'missing/corrupt'): c.run(args)
            db.close()
    def test_resume_contract_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'radar').mkdir()
            args=argparse.Namespace(radar_dir=str(root/'radar'),output_dir=str(root/'out'),start_utc='2024-01-01T00:00:00Z',end_utc='2024-01-01T00:00:00Z',cadence_minutes=60,resume=False,progress_every=10)
            with contextlib.redirect_stdout(io.StringIO()):c.run(args)
            args.resume=True;args.cadence_minutes=2
            with self.assertRaisesRegex(ValueError,'contract differs'):c.run(args)

if __name__=='__main__':unittest.main()
