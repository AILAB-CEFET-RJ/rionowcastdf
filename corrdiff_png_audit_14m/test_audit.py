import csv
import fcntl
import importlib.util
import io
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

SCRIPT=Path(__file__).with_name('audit_pngs.py')
spec=importlib.util.spec_from_file_location('audit',SCRIPT)
a=importlib.util.module_from_spec(spec); spec.loader.exec_module(a)


def png(array,fmt='PNG'):
    b=io.BytesIO(); Image.fromarray(array).save(b,format=fmt); return b.getvalue()


class AuditTests(unittest.TestCase):
    def test_pixel_classes_and_geometry(self):
        x=np.zeros((654,656,4),dtype='uint8')
        d=a.decode(png(x))
        self.assertEqual(d['status'],'transparent_ambiguous')
        self.assertEqual(d['transparent_pixels'],429024)
        x[:]=[242,239,233,255]
        x[10,10]=[*a.PALETTE[0],255]
        d=a.decode(png(x))
        self.assertEqual(d['status'],'map_screen_review')
        self.assertEqual(d['palette_pixels'],1)
        x[:]=[1,2,3,255]
        self.assertEqual(a.decode(png(x))['status'],'no_exact_palette_review')
        x[:]=0
        x[0,0]=[*a.PALETTE[0],255]
        self.assertEqual(a.decode(png(x))['status'],'palette_in_png_not_sampled_in_corrected_grid')
        x[:]=0
        x[a.GRID_ROWS[0],a.GRID_COLS[0]]=[*a.PALETTE[0],255]
        self.assertEqual(a.decode(png(x))['status'],'palette_sampled_observation_validity_unconfirmed')
        x[a.GRID_ROWS[0],a.GRID_COLS[0],3]=128
        d=a.decode(png(x))
        self.assertEqual(d['palette_pixels'],0)
        self.assertEqual(d['partial_alpha_pixels'],1)
        self.assertEqual(a.GRID_ROWS.size,5880)

    def test_decode_failures_and_dimensions(self):
        self.assertEqual(a.decode(b'not an image')['status'],'decode_error')
        x=np.zeros((768,1024,4),dtype='uint8')
        self.assertEqual(a.decode(png(x))['status'],'unexpected_dimensions_review')
        data=png(x)
        self.assertEqual(a.decode(data[:100])['status'],'decode_error')
        x=np.zeros((654,656,3),dtype='uint8')
        self.assertEqual(a.decode(png(x,'JPEG'))['status'],'not_png_format_review')

    def test_local_time_dst(self):
        t,state,year,utc=a.timestamp_info(Path('2012_02_25_03_00.png'))
        self.assertEqual(utc,'2012-02-25T05:00:00+00:00')
        self.assertEqual(a.timestamp_info(Path('2012_02_25_23_30.png'))[1],'ambiguous')
        self.assertEqual(a.timestamp_info(Path('2012_10_21_00_30.png'))[1],'nonexistent')
        self.assertEqual(a.timestamp_info(Path('bogus.png'))[1],'unparsed')

    def test_end_to_end_resume_changed_removed_and_lock(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'raw'; root.mkdir(); out=Path(td)/'results'
            x=np.zeros((654,656,4),dtype='uint8'); data=png(x)
            p1=root/'2012_02_25_03_00.png'; p1.write_bytes(data)
            p2=root/'2012_02_25_03_02.png'; p2.write_bytes(data)
            bad=root/'2012_02_25_03_04.png'; bad.write_bytes(b'bad')
            cmd=[sys.executable,str(SCRIPT),'--radar-dir',str(root),'--output-dir',str(out)]
            def run(extra=(),expected=0):
                r=subprocess.run(cmd+list(extra),capture_output=True,text=True)
                self.assertEqual(r.returncode,expected,r.stdout+r.stderr)
                return json.loads((out/'summary.json').read_text())
            s=run(); self.assertTrue(s['complete']); self.assertEqual(s['processed_files'],3)
            self.assertEqual(s['decoded_unique_this_run'],2)
            self.assertEqual(s['reused_by_hash'],1)
            self.assertTrue((out/'review_bundle.zip').is_file())
            s=run(['--resume']); self.assertEqual(s['reused_by_stat'],3)
            x[:]=[242,239,233,255]; p1.write_bytes(png(x)); p2.unlink()
            s=run(['--resume']); self.assertEqual(s['processed_files'],2)
            self.assertEqual(s['source_changes_this_run'],1)
            self.assertEqual(s['counts']['map_screen_review'],1)
            with (out/'files_absent_on_rescan.csv').open() as f:
                self.assertEqual(len(list(csv.DictReader(f))),1)
            s=run(['--resume','--rehash']); self.assertEqual(s['reused_by_stat'],0)
            self.assertEqual(s['reused_by_hash'],2)
            with (out/'.audit.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                run(['--resume'],2)
            (root/'link.png').symlink_to(p1)
            s=run(['--resume'],2); self.assertFalse(s['complete'])
            self.assertEqual(s['counts']['symlink_file_skipped'],1)

if __name__=='__main__': unittest.main(verbosity=2)
