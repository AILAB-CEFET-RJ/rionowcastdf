"""Shared timestamp, split and lineage checks for QC reprocessing."""
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd

SPLITS=['excluded_purge','train','validation','test_primary','test_stress_ood']
BOUNDARIES=['2021-01-01','2023-01-01','2024-01-01']


def timestamps(root):
    raw=np.asarray(root['timestamps'][:])
    if not len(raw):raise ValueError('Empty dataset')
    if np.issubdtype(raw.dtype,np.integer):
        scale=int(np.median(np.abs(raw)))
        unit='ns' if scale>=10**17 else 'us' if scale>=10**14 else 'ms' if scale>=10**11 else 's'
        ts=pd.DatetimeIndex(pd.to_datetime(raw,unit=unit,utc=True))
    else:ts=pd.DatetimeIndex(pd.to_datetime(raw,utc=True))
    if ts.isna().any() or not ((ts.year>=2011)&(ts.year<=2024)).all():raise ValueError('Invalid timestamps')
    if not ts.is_monotonic_increasing:raise ValueError('Timestamps must be sorted')
    if not (ts==ts.floor('h')).all():raise ValueError('Dataset must use exact hourly timestamps')
    if root['target'].shape!=root['mask'].shape or len(ts)!=len(root['target']):raise ValueError('Misaligned target/mask/timestamps')
    pos=np.stack([root['patch_row'][:],root['patch_col'][:]],axis=1)
    keys=pd.DataFrame({'timestamp':ts,'row':pos[:,0],'col':pos[:,1]})
    if keys.duplicated().any():raise ValueError('Duplicate timestamp/patch position')
    return ts,pos


def split_codes(ts):
    codes=np.where(ts.year<=2020,1,np.where(ts.year<=2022,2,np.where(ts.year==2023,3,4)))
    for boundary in BOUNDARIES:
        b=pd.Timestamp(boundary,tz='UTC')
        codes[(ts>=b-pd.Timedelta(hours=48))&(ts<b+pd.Timedelta(hours=48))]=0
    return codes


def index_fingerprint(ts,pos):
    h=hashlib.sha256()
    h.update(ts.asi8.astype('<i8').tobytes());h.update(np.asarray(pos,dtype='<i8').tobytes())
    return h.hexdigest()
