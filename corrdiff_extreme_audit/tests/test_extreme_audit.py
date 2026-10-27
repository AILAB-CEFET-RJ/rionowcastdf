import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location('audit',Path(__file__).parents[1]/'scripts/14b_audit_extreme_episodes.py')
a=importlib.util.module_from_spec(spec); spec.loader.exec_module(a)

class AuditTests(unittest.TestCase):
    def test_overlap_and_mask(self):
        p=np.full((2,2,2),np.float32(np.log1p(45)),np.float32)
        m=np.ones_like(p,bool)
        r=a.summarize_frame(pd.Timestamp('2024-01-01',tz='UTC'),p,m,np.array([0,0]),np.array([0,1]))
        self.assertEqual(r['patch_count_45'],8)
        self.assertEqual(r['unique_count_45'],6)
        p[1,0,0]=0
        with self.assertRaises(ValueError): a.reconstruct(p,m,[0,0],[0,1])

    def test_gap_dry_and_repeat(self):
        df=pd.DataFrame({'timestamp':pd.to_datetime(['2024-01-01T00:00Z','2024-01-01T01:00Z','2024-01-01T03:00Z','2024-01-01T04:00Z','2024-01-01T05:00Z']), 'unique_count_45':[1,1,1,0,1]})
        self.assertEqual(a.assign_episodes(df,45).tolist(),[1,1,2,0,3])

    def test_mask_and_boundary(self):
        cut=np.float32(np.log1p(45))
        p=np.array([[[cut,np.nextafter(cut,np.float32(0))],[cut,0]]],np.float32)
        m=np.array([[[1,1],[0,1]]],bool)
        r=a.summarize_frame(pd.Timestamp('2024-01-01',tz='UTC'),p,m,np.array([0]),np.array([0]))
        self.assertEqual(r['unique_count_45'],1)
        self.assertEqual(r['unique_valid'],3)

    def test_batch_boundaries(self):
        import zarr
        with tempfile.TemporaryDirectory() as tmp:
            root=zarr.open_group(tmp,mode='w')
            y=np.full((6,1,2,2),np.float32(np.log1p(45)),np.float32)
            for name,value in {'target':y,'mask':np.ones_like(y),'timestamps':np.array([1704067200]*2+[1704070800]*2+[1704078000]*2),'patch_row':np.zeros(6,int),'patch_col':np.tile([0,1],3)}.items(): root.create_dataset(name,data=value)
            df=a.scan(root,3)
            self.assertEqual(df.unique_count_45.tolist(),[6]*3)
            self.assertEqual(df.same_field_previous_exact_1h.tolist(),[False,True,False])
            self.assertEqual(df.end_row.tolist(),[2,4,6])
            df['episode_id']=a.assign_episodes(df,45)
            df.attrs['pixel_km']=2
            ep=a.episode_table(df,45)
            self.assertEqual(ep.pixel_hours.tolist(),[12,6])
            out=a.yearly(df,'removed',{1})
            self.assertEqual(out[0]['retained_hours'],1)

if __name__=='__main__': unittest.main()
