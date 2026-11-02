import sys
import unittest
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment import (local_to_utc,utc_candidates,parse_utc,png_path,geometry,
                       project_pixels,analyze_array,CONTRACT)

class AlignmentTests(unittest.TestCase):
    def test_local_time_and_historical_dst(self):
        self.assertEqual(local_to_utc('2024_01_01_12_00').hour,15)
        self.assertEqual(local_to_utc('2018_01_01_12_00').hour,14)
        self.assertEqual(local_to_utc('2012_02_25_03_00').hour,5)
    def test_utc_filename_date_rollover(self):
        self.assertEqual(str(png_path('/radar',parse_utc('2024-01-01T01:00Z'))),
                         '/radar/2023/12/31/2023_12_31_22_00.png')
    def test_dst_ambiguity_and_nonexistent(self):
        self.assertEqual(len(utc_candidates(datetime(2018,2,17,23,30))),2)
        self.assertEqual(len(utc_candidates(datetime(2018,11,4,0,30))),0)
        with self.assertRaises(ValueError):local_to_utc('2018_02_17_23_30')
        with self.assertRaises(ValueError):local_to_utc('2018_11_04_00_30')
    def test_utc_required(self):
        with self.assertRaises(ValueError):parse_utc('2024-01-01T01:00')
    def test_projection_center_orientation(self):
        p=CONTRACT['radar_latitude'];l=CONTRACT['radar_longitude']
        row,col,inside=project_pixels(np.array([p,p+.1,p]),np.array([l,l,l+.1]))
        self.assertAlmostEqual(row[0],326.5,places=5)
        self.assertAlmostEqual(col[0],327.5,places=5)
        self.assertTrue(row[1]<row[0]);self.assertTrue(col[2]>col[0]);self.assertTrue(inside.all())
    def test_geometry_roi(self):
        g=geometry();self.assertEqual(g['lat'].shape,(70,84))
        self.assertTrue(g['geometric_mask'].all())
        self.assertLess(g['native_roi'].sum(),654*656)
        self.assertTrue(np.any(g['px']!=g['legacy_px']))
        self.assertTrue(np.any(g['py']!=g['legacy_py']))
    def test_transparency_and_map_never_dry(self):
        g=geometry();a=np.zeros((654,656,4),np.uint8)
        _,corrected,_,review,r=analyze_array(a,g)
        self.assertEqual(r['status'],'transparent_ambiguous');self.assertTrue(np.isnan(review).all())
        a[:]=[242,239,233,255]
        _,_,_,review,r=analyze_array(a,g)
        self.assertEqual(r['status'],'map_screen_review');self.assertTrue(np.isnan(review).all())
    def test_corrected_pixel_and_missing_alpha(self):
        g=geometry();a=np.zeros((654,656,4),np.uint8);i,j=30,30
        y,x=g['py'][i,j],g['px'][i,j];a[y,x]=[227,6,5,255]
        _,z,_,_,r=analyze_array(a,g)
        self.assertEqual(z[i,j],45);self.assertEqual(r['native_roi_palette_pixels'],1)
        a[y,x,3]=0
        _,z,_,_,r=analyze_array(a,g);self.assertTrue(np.isnan(z).all())
    def test_shape_rejected(self):
        with self.assertRaises(ValueError):analyze_array(np.zeros((768,1024,4),np.uint8),geometry())

if __name__=='__main__':unittest.main()
