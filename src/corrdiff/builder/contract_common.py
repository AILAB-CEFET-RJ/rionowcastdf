"""Shared immutable contracts for the reference Zarr and compact radar cache."""
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from .reference_geometry import geometry as reference_geometry, CONTRACT as REFERENCE

VERSION = 'reference-zarr-v1'
CHANNELS = ['tcwv','t2m','u10','v10','t_850','r_850','u_850','v_850','t_500','r_500','u_500','v_500']
SPLITS = ['train','validation','test_primary','test_stress_ood']
POLICIES = ['valid_transparent_no_echo','valid_no_echo_on_sampled_grid','valid_echo_and_no_echo','reviewed_map_no_echo']
PATHS = ['temporal_dir','geometry_file','coverage_file','spatial_contract_file','radar_dir','reuse_14j_dir','radar_cache_dir','pairing_dir','era5_root','output_dir','era5_identity_file']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        json.dump(obj,f,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def digest_obj(obj): return hashlib.sha256(json.dumps(obj,sort_keys=True,allow_nan=False).encode()).hexdigest()


def inside(root,relative):
    root=Path(root).resolve();p=(root/relative).resolve()
    if root not in p.parents:raise ValueError('Path outside root: '+str(relative))
    return p


def ro(path): return sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True)


def load_config(file):
    file=Path(file).resolve(); c=read(file)
    if c.get('version')!=VERSION:raise ValueError('Expected config '+VERSION)
    root=Path(os.path.expandvars(c.get('project_root','..')))
    if not root.is_absolute(): root=file.parent/root
    root=root.resolve()
    for key in PATHS:
        value=c.get(key)
        if value:
            expanded=os.path.expandvars(value)
            if '$' in expanded:raise ValueError('Unset environment variable in '+key)
            path=Path(expanded);c[key]=str((path if path.is_absolute() else root/path).resolve())
    c['channels']=c.get('channels',CHANNELS)
    if not c['channels'] or len(set(c['channels']))!=len(c['channels']) or any(ch not in CHANNELS for ch in c['channels']):
        raise ValueError('Channels must be unique supported ERA5 names')
    if c.get('patch_size',32)!=32 or c.get('patch_stride',8)!=8:raise ValueError('Reference build requires patch 32/stride 8')
    c.setdefault('workers',4);c.setdefault('duckdb_threads',2);c.setdefault('duckdb_memory','2GB')
    c.setdefault('start_utc',None);c.setdefault('end_utc',None)
    if c['workers']<1:raise ValueError('workers must be positive')
    inputs=[Path(c[k]) for k in ('temporal_dir','geometry_file','coverage_file','spatial_contract_file','radar_dir','pairing_dir','era5_root')]
    if c.get('reuse_14j_dir'):inputs.append(Path(c['reuse_14j_dir']))
    outputs=[Path(c[k]) for k in ('radar_cache_dir','output_dir')]
    for a in outputs:
        for b in inputs:
            if a==b or a in b.parents or b in a.parents:raise ValueError('Input/output trees must be separate: '+str(a)+' / '+str(b))
    if outputs[0]==outputs[1] or outputs[0] in outputs[1].parents or outputs[1] in outputs[0].parents:raise ValueError('Cache and dataset outputs must be separate')
    return c


def expected_split(t):
    for y in (2021,2023,2024):
        b=pd.Timestamp(f'{y}-01-01',tz='UTC')
        if b-pd.Timedelta(hours=48)<=t<b+pd.Timedelta(hours=48):return 'excluded_purge'
    return 'train' if t.year<=2020 else ('validation' if t.year<=2022 else ('test_primary' if t.year==2023 else 'test_stress_ood'))


def inputs(c):
    td=Path(c['temporal_dir']);s=read(td/'summary.json');tc=read(td/'temporal_contract.json')
    if not s.get('complete') or not s.get('temporal_rule_finalized'):raise ValueError('14p freeze not complete')
    for name in ('temporal_contract.json','selected_manifest.parquet'):
        if sha(td/name)!=s['artifact_sha256'][name]:raise ValueError('14p checksum mismatch: '+name)
    if tc['strategy'] not in ('exact','nearest2','nearest5') or tc['aggregation']!='none' or tc['no_echo_code']!=0:
        raise ValueError('Unsupported temporal contract')
    tolerance={'exact':0,'nearest2':120,'nearest5':300}[tc['strategy']]
    if tc['tolerance_seconds']!=tolerance or tc['source_timezone']!='America/Sao_Paulo':raise ValueError('Invalid temporal settings')
    df=pd.read_parquet(td/'selected_manifest.parquet')
    if len(df)!=s['selected_hours']:raise ValueError('Manifest hour count mismatch')
    for col in ('paired','era5_ready'):
        if df[col].isna().any() or not df[col].eq(True).all():raise ValueError('Noneligible manifest rows')
    df['timestamp_utc']=pd.to_datetime(df.timestamp_utc,utc=True)
    df['source_utc']=pd.to_datetime(df.source_utc,utc=True)
    if df[['timestamp_utc','source_utc']].isna().any().any():raise ValueError('Null timestamps')
    if df.timestamp_utc.duplicated().any() or not df.split.isin(SPLITS).all():raise ValueError('Invalid splits/duplicate hours')
    if not df.policy_status.isin(POLICIES).all() or not df.strategy.eq(tc['strategy']).all():raise ValueError('Invalid selected policy')
    for r in df.itertuples(index=False):
        if r.timestamp_utc!=r.timestamp_utc.floor('h'):raise ValueError('ERA5 reference not hourly')
        offset=(r.source_utc-r.timestamp_utc).total_seconds()
        if abs(offset)>tolerance or offset!=r.offset_seconds:raise ValueError('Invalid temporal offset')
        if r.timestamp_utc<pd.Timestamp('2011-01-17T02:00:00Z'):raise ValueError('Before archive')
        if expected_split(r.timestamp_utc)!=r.split or expected_split(r.source_utc)!=r.split:raise ValueError('Split leakage')
        expected=r.source_utc.tz_convert('America/Sao_Paulo').strftime('%Y/%m/%d/%Y_%m_%d_%H_%M.png')
        if str(r.path)!=expected:raise ValueError('Local PNG filename does not match UTC: '+str(r.path))
        if len(r.sha)!=64 or any(x not in '0123456789abcdef' for x in r.sha):raise ValueError('Invalid SHA')
    if df.groupby('path').split.nunique().gt(1).any():raise ValueError('PNG path reused across splits')
    for key,op in [('start_utc','ge'),('end_utc','lt')]:
        if c.get(key):
            t=pd.Timestamp(c[key])
            if t.tzinfo is None:raise ValueError('Subset boundaries require explicit timezone')
            df=df[getattr(df.timestamp_utc,op)(t)]
    df=df.sort_values('timestamp_utc').reset_index(drop=True)
    if df.empty:raise ValueError('Empty selected period')
    # Hash-specific map decisions cannot be mixed for identical content.
    if df.groupby('sha').policy_status.nunique().gt(1).any():raise ValueError('Conflicting policies for same content')
    g=reference_geometry()
    with np.load(c['geometry_file'],allow_pickle=False) as z:
        for key in ('lat','lon','px','py','geometric_mask'):
            if not np.array_equal(z[key],g[key]):raise ValueError('Geometry differs from adopted 14j: '+key)
    sc=read(c['spatial_contract_file'])
    if sc.get('no_echo_code')!=0 or sc.get('grid_shape')!=[70,84] or sc.get('target_normalization')!='target_code / 50' or sc.get('patch_stride')!=8:raise ValueError('Unsupported 14o numeric contract')
    if sorted(sc['source_palette_dbz'])!=[20,25,30,35,40,45,50]:raise ValueError('Unexpected palette')
    expected_cov=sc.get('policy_artifacts',{}).get('coverage.npz')
    if not expected_cov or sha(c['coverage_file'])!=expected_cov:raise ValueError('Coverage not bound to 14o contract')
    with np.load(c['coverage_file'],allow_pickle=False) as z:
        support=z['observed_region_mask'].copy()
        if support.dtype!=bool or support.shape!=(70,84) or not support.any():raise ValueError('Invalid coverage')
        if not np.array_equal(z['lat'],g['lat']) or not np.array_equal(z['lon'],g['lon']):raise ValueError('Coverage coordinates differ')
    if (support&~g['geometric_mask']).any():raise ValueError('Coverage outside geometry')
    g['support']=support
    semantic=dict(version=VERSION,temporal_contract=tc,manifest_sha256=s['artifact_sha256']['selected_manifest.parquet'],
        geometry_sha256=sha(c['geometry_file']),coverage_sha256=sha(c['coverage_file']),
        spatial_contract_sha256=sha(c['spatial_contract_file']),channels=c['channels'],start_utc=c['start_utc'],end_utc=c['end_utc'],
        patch_size=32,patch_stride=8,target_transform='divide_by_50',input_storage='physical; normalize once in loader',
        coverage_state=sc.get('coverage_state','inherited'),spatial_validity_confirmed=sc.get('spatial_validity_confirmed',False),
        code_sha256={p.name:sha(p) for p in sorted(Path(__file__).parent.glob('contract_*.py'))}|{'reference_geometry.py':sha(Path(__file__).with_name('reference_geometry.py'))})
    return df,g,semantic


@contextlib.contextmanager
def output_lock(root,resume,contract):
    root=Path(root)
    if root.exists() and not resume:raise ValueError('Output exists; use --resume or new directory: '+str(root))
    root.mkdir(parents=True,exist_ok=True)
    with (root/'build.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cp=root/'contract.json'
        if cp.exists():
            if read(cp)!=contract:raise ValueError('Contract/code/input changed; use new output directory')
        else:dump(cp,contract)
        yield root


def starts(n):
    v=list(range(0,n-32+1,8))
    if v[-1]!=n-32:v.append(n-32)
    return v


def validate_target(code,mask,support):
    if code.shape!=(70,84) or code.dtype!=np.float32 or mask.dtype!=bool or mask.shape!=code.shape:raise ValueError('Target shape/type')
    if (mask&~support).any() or not np.isnan(code[~mask]).all():raise ValueError('Unknown pixel contract')
    if not np.isin(code[mask],[0,20,25,30,35,40,45,50]).all():raise ValueError('Invalid radar code')
