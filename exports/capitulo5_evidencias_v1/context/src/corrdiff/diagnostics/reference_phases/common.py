from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from ..reference_reader import ReferenceReader, SPLITS, CODES

VERSION = 'reference-phases-v1'
NAMES = {0:'structure',1:'univariate',2:'derived',3:'joint',4:'extremes',
5:'seasonality',6:'diurnal',7:'temporal_lags',8:'persistence',9:'spatial_structure',
10:'spatial_era5_radar',11:'spatial_scales',12:'multivariate',13:'regimes',
14:'distribution_shift',15:'split_audit',16:'baselines',17:'ablations',18:'native_integration'}
DEPS = {15:[],0:[15],1:[0],2:[0],3:[0,2],4:[0],5:[0],6:[0],7:[0],8:[0],
9:[0],10:[0],11:[0],12:[0],13:[0,12],14:[0,13],16:[0,15],17:[15],18:[15,17]}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):return [clean(v) for v in value]
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def dump(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(clean(obj),indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)


def csv(path,rows):
    pd.DataFrame(rows).to_csv(path,index=False)


def split_for(seconds):
    t=pd.DatetimeIndex(pd.to_datetime(np.asarray(seconds),unit='s',utc=True))
    years=t.year.to_numpy()
    s=np.where(years<=2020,'train',np.where(years<=2022,'validation',np.where(years==2023,'test_primary','test_stress_ood')))
    s=s.astype(object)
    for year in (2021,2023,2024):
        b=pd.Timestamp(f'{year}-01-01',tz='UTC').timestamp()
        s[(np.asarray(seconds)>=b-48*3600)&(np.asarray(seconds)<b+48*3600)]='excluded_purge'
    return s


def sample_hours(hours,limit,seed):
    if limit==0 or len(hours)<=limit:return np.asarray(hours)
    return np.sort(np.random.default_rng(seed).choice(hours,limit,replace=False))


def corr(a,b):
    m=np.isfinite(a)&np.isfinite(b);a,b=np.asarray(a)[m],np.asarray(b)[m]
    return float(np.corrcoef(a,b)[0,1]) if len(a)>1 and np.std(a)>0 and np.std(b)>0 else None


def derived(x,names):
    v={name:x[i] for i,name in enumerate(names)}
    return {
        'wind_speed_10':np.hypot(v['u10'],v['v10']),
        'wind_speed_850':np.hypot(v['u_850'],v['v_850']),
        'wind_speed_500':np.hypot(v['u_500'],v['v_500']),
        'delta_t_500_850':v['t_500']-v['t_850'],
        'delta_t_850_surface':v['t_850']-v['t2m'],
        'delta_r_500_850':v['r_500']-v['r_850'],
        'bulk_wind_diff_10_850':np.hypot(v['u_850']-v['u10'],v['v_850']-v['v10']),
        'bulk_wind_diff_850_500':np.hypot(v['u_500']-v['u_850'],v['v_500']-v['v_850'])}


class Context:
    def __init__(self,config,output):
        self.c=config;self.output=Path(output);self._readers={}
    def directory(self,phase):return self.output/f'{phase:02d}_{NAMES[phase]}'
    def reader(self,split):
        if split not in self._readers:self._readers[split]=ReferenceReader(self.c['dataset_dir'],split)
        return self._readers[split]
    def table(self):return pd.read_parquet(self.directory(0)/'hour_metrics.parquet')
    def selected(self,split='train',limit=None):
        if limit is None:limit=self.c['diagnostic_hours_per_split']
        return sample_hours(self.reader(split).hours,limit,self.c['seed'])
    def fields(self,split='train',limit=None):
        r=self.reader(split)
        for t in self.selected(split,limit):yield int(t),r.hour(int(t))


def describe(values):
    v=np.asarray(values,dtype='f8');v=v[np.isfinite(v)]
    if not len(v):return {'n':0}
    return dict(n=len(v),mean=v.mean(),std=v.std(),min=v.min(),max=v.max(),
                q01=np.quantile(v,.01),q50=np.quantile(v,.5),q95=np.quantile(v,.95),q99=np.quantile(v,.99))
