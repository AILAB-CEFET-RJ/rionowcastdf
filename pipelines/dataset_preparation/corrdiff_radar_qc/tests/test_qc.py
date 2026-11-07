import unittest
import numpy as np
from radar_qc import decode,classify,COLORS,VALUES,event_mask
class QC(unittest.TestCase):
    def test_palette(self):np.testing.assert_array_equal(decode(COLORS[None]),VALUES[None])
    def test_map_unknown(self):self.assertTrue(np.isnan(decode(np.array([[[242,239,233],[221,221,221]]]))).all())
    def test_alpha(self):self.assertTrue(np.isnan(decode(np.array([[[227,6,5,0],[227,6,5,128]]]))).all())
    def test_dry_not_invented(self):self.assertTrue(np.isnan(decode(np.array([[[0,0,0],[255,255,255]]]))).all())
    def test_quarantine(self):
        grid,info=classify(np.tile([242,239,233,255],(70,84,1)))
        self.assertEqual(info['status'],'quarantine_map_review');self.assertTrue(np.isnan(grid).all())
    def test_valid_red(self):
        grid,info=classify(np.tile([227,6,5,255],(2,2,1)))
        self.assertEqual(info['status'],'accepted_full');self.assertEqual(info['extreme_pixels'],4)
    def test_threshold_roundtrip(self):
        for th in [20,30,40,45,50]:
            x=np.log1p(np.array([th,0],np.float32));np.testing.assert_array_equal(event_mask(x,[True,False],th),[True,False])
if __name__=='__main__':unittest.main()
