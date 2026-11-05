#!/usr/bin/env python3
"""Bounded loader check on actual prepared training data; no native model claim."""
import argparse
from pathlib import Path
import numpy as np
import torch
from dataset import RadarDataset
from pair_contract import dump

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-dir',required=True)
    p.add_argument('--output-dir',required=True);p.add_argument('--max-patches',type=int,default=48);a=p.parse_args()
    if a.max_patches<=0:p.error('max-patches must be positive')
    out=Path(a.output_dir)
    if out.exists():raise ValueError('Use a new report directory')
    ds=RadarDataset(a.data_dir,'train')
    if not len(ds):raise ValueError('No train samples')
    ids=np.unique(np.linspace(0,len(ds)-1,min(a.max_patches,len(ds))).astype(int))
    support=echo=0
    for i in ids:
        b=ds[int(i)];x=b['conditioning'];y=b['target'];m=b['mask']
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():raise ValueError('Nonfinite tensor')
        if not torch.isin(y[m],torch.tensor([0,.4,.5,.6,.7,.8,.9,1.])).all():raise ValueError('Unexpected target code')
        if (y[~m]!=0).any():raise ValueError('Invalid tensor placeholder')
        support+=int(m.sum());echo+=int(((y>0)&m).sum())
    out.mkdir(parents=True)
    report={'complete':True,'loader_check_passed':True,'sampled_train_patches':len(ids),
      'supported_pixel_occurrences':support,'echo_pixel_occurrences':echo,'production_ready':False,
      'corrdiff_native_tested':False,'pilot_only':ds.summary['pilot_only']}
    dump(out/'summary.json',report);print(report)
