"""Prepared 14o patches; masks are mandatory, never implicit zero labels."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pair_contract import digest, CHANNELS

class RadarDataset(Dataset):
    def __init__(self,root,split='train',fully_supported=False):
        self.root=Path(root).resolve();s=json.loads((self.root/'summary.json').read_text())
        if s.get('version')!='14o-prepared-v1' or not s.get('complete'): raise ValueError('Incomplete 14o')
        for n,h in s['artifact_sha256'].items():
            if digest(self.root/n)!=h: raise ValueError('Changed artifact: '+n)
        self.contract=json.loads((self.root/'contract.json').read_text()); self.summary=s
        if split not in ('train','validation','test_primary','test_stress_ood'): raise ValueError('Invalid split')
        stats=json.loads((self.root/'normalization.json').read_text())
        if stats['channels']!=CHANNELS: raise ValueError('Channel order mismatch')
        self.mean=np.array(stats['mean'],dtype='float32')[:,None,None]
        self.scale=np.array(stats['scale'],dtype='float32')[:,None,None]
        if not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or (self.scale<=0).any():raise ValueError('Invalid normalization')
        p=pd.read_parquet(self.root/'patches.parquet');p=p[p.split.eq(split)]
        if fully_supported:p=p[p.fully_supported]
        self.rows=p.reset_index(drop=True);self.loaded={};self.checked=set()
        with np.load(self.root/'coverage.npz',allow_pickle=False) as g:
            # Fixed geographical coordinates, independent of train/test values.
            self.coordinates=np.stack(((g['lat']+23.5)/1.25*2-1,(g['lon']+44)/1.5*2-1)).astype('float32')
    def __len__(self):return len(self.rows)
    def __getstate__(self):
        d=self.__dict__.copy();d['loaded']={};return d
    def __getitem__(self,index):
        r=self.rows.iloc[index];key=r.shard
        if key not in self.loaded:
            folder=self.root/'shards'/key
            if key not in self.checked:
                meta=json.loads((folder/'complete.json').read_text())
                for n,h in meta['sha256'].items():
                    if digest(folder/n)!=h:raise ValueError('Shard checksum mismatch: '+str(folder/n))
                self.checked.add(key)
            # Bound number of mappings; slices returned below are copies.
            if len(self.loaded)>=2:self.loaded.pop(next(iter(self.loaded)))
            self.loaded[key]=tuple(np.load(folder/n,mmap_mode='r') for n in ['conditioning.npy','target_code.npy','mask.npy'])
        x,y,m=self.loaded[key];i=int(r.hour_index);iy=int(r.row_start);ix=int(r.col_start)
        sl=np.s_[iy:iy+32,ix:ix+32]
        xx=((x[i,:,iy:iy+32,ix:ix+32]-self.mean)/self.scale).astype('float32')
        mask=np.array(m[i][sl],copy=True)[None]
        raw=np.array(y[i][sl],copy=True)[None]
        if not np.isfinite(xx).all() or not np.isfinite(raw[mask]).all():raise ValueError('Nonfinite sample')
        yy=np.where(mask,raw/50.,0.).astype('float32')
        pos=self.coordinates[:,iy:iy+32,ix:ix+32].copy()
        return {'conditioning':torch.from_numpy(xx),'target':torch.from_numpy(yy),
                'mask':torch.from_numpy(mask),'coordinates':torch.from_numpy(pos),'index':int(index)}

def nearest_category(code_score):
    codes=np.array([0,20,25,30,35,40,45,50],dtype=float)
    a=np.asarray(code_score)
    if not np.isfinite(a).all():raise ValueError('Nonfinite prediction')
    return codes[np.abs(a[...,None]-codes).argmin(axis=-1)]

def masked_mse(pred,target,mask,weight=None):
    if pred.shape!=target.shape or mask.shape!=target.shape or mask.dtype!=torch.bool or not mask.any():raise ValueError('Invalid loss support')
    p=pred[mask];t=target[mask]
    if not torch.isfinite(p).all() or not torch.isfinite(t).all():raise ValueError('Nonfinite supported prediction/target')
    loss=(p-t).square()
    if weight is not None:loss=loss*torch.broadcast_to(weight,target.shape)[mask]
    return loss.mean()

def residual_inputs(target,mean,mask,sigma,noise):
    # No target or random value from unsupported locations enters the denoiser.
    residual=torch.where(mask,target-mean,torch.zeros_like(target))
    noisy=torch.where(mask,residual+sigma*noise,torch.zeros_like(target))
    return residual,noisy
