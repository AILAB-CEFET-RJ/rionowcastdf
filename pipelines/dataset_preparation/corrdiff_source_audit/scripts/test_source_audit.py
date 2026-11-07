"""Synthetic integration test; pass the local builder as the sole argument."""
import importlib.util,sys,tempfile,subprocess,json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import zarr
from audit14b_helpers import summarize_frame
spec=importlib.util.spec_from_file_location('audit',Path(__file__).with_name('14c_audit_radar_sources.py')); mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
builder=Path(sys.argv[1]).resolve()
with tempfile.TemporaryDirectory() as td:
 d=Path(td); (d/'raw/2024/01/01').mkdir(parents=True);(d/'cache').mkdir();(d/'14b').mkdir()
 C=mod.load_converter(builder);c=C(d/'raw',d/'tmp',2,[-23.5,-22.25],[-44,-42.5],pd.Timestamp('2024-01-01',tz='UTC'),pd.Timestamp('2024-01-02',tz='UTC'))
 c.build_grid();c.precompute_pixel_map()
 rgba=np.zeros((654,656,4),np.uint8);rgba[:,:,:3]=[227,6,5];rgba[:,:,3]=255;rgba[:100,:,3]=0
 grid=c.rgb_to_reflectivity(rgba[c.py,c.px,:3]); z=zarr.open_group(str(d/'train.zarr'),mode='w')
 times=pd.date_range('2024-01-01',periods=3,freq='h',tz='UTC');y=np.log1p(grid[:32,:32]).astype(np.float32)
 z.create_dataset('target',data=np.stack([y]*3)[:,None]);z.create_dataset('mask',data=np.ones((3,1,32,32),np.uint8))
 z.create_dataset('timestamps',data=times.as_unit('s').asi8);z.create_dataset('patch_row',data=np.zeros(3,int));z.create_dataset('patch_col',data=np.zeros(3,int))
 records=[]
 for i,t in enumerate(times):
  r=summarize_frame(t,y[None],np.ones((1,32,32),bool),np.array([0]),np.array([0]));r.update(first_row=i,end_row=i+1);records.append(r)
  g=grid.copy()
  if i==1:g[0,0]+=1
  np.save(d/'cache'/t.strftime('%Y%m%d_%H_%M.npy'),g)
  if i<2:Image.fromarray(rgba).save(d/'raw'/t.strftime('%Y/%m/%d/%Y_%m_%d_%H_%M.png'))
 pd.DataFrame(records).to_csv(d/'14b/timestamp_metrics.csv',index=False)
 pd.DataFrame([dict(peak_timestamp=str(times[0]),peak_year=2024)]).to_csv(d/'14b/episodes_ranked.csv',index=False)
 subprocess.run([sys.executable,str(Path(__file__).with_name('14c_audit_radar_sources.py')),'--dataset-dir',str(d),'--audit14b-dir',str(d/'14b'),'--radar-dir',str(d/'raw'),'--radar-cache-dir',str(d/'cache'),'--builder-file',str(builder),'--output-dir',str(d/'out')],check=True)
 checks=pd.read_csv(d/'out/source_checks.csv');summary=json.loads((d/'out/summary.json').read_text())
 assert summary['errors']==0 and summary['missing_png']==1
 assert json.loads(checks.iloc[0].png_cache)['unequal_finite']==0
 assert json.loads(checks.iloc[1].png_cache)['unequal_finite']==1
 assert json.loads(checks.iloc[1].cache_target)['unequal_finite']==1
 assert len(pd.read_csv(d/'out/extreme_mask_hash_repeats.csv'))==3
 assert len(list((d/'out').glob('*.png')))==2
 # Save one render outside fixture for manual QA when explicitly requested.
 print('PASS: source identity, cache divergence, missing PNG, repeats, panels')
