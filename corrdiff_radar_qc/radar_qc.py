"""Versioned, conservative decoding of Sumaré PNGs.

The default is a diagnostic policy, not a proof of observational coverage.
Unknown/transparent pixels cannot be made dry without a documented source rule.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import numpy as np

VERSION = 'radar-qc-v2-reviewed-palette'
COLORS = np.array([(197,0,197),(227,6,5),(255,112,0),(195,230,0),
                   (4,85,4),(19,122,19),(0,167,12)], dtype=np.uint8)
VALUES = np.array([50,45,40,35,30,25,20], dtype=np.float32)
DEFAULT = dict(version=VERSION, palette=COLORS.tolist(), values=VALUES.tolist(),
               dry_rgba=[], dry_evidence='', source_contract_evidence='',
               map_background_rgb=[242,239,233], map_fraction_limit=0.01,
               map_rule_role='screening_only_quarantine_pending_review',
               interpolation=False)


def fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def load_policy(path=None):
    p=copy.deepcopy(DEFAULT) if path is None else json.loads(Path(path).read_text())
    if p.get('version')!=VERSION: raise ValueError('Unsupported policy version')
    if p.get('palette')!=COLORS.tolist() or p.get('values')!=VALUES.tolist():
        raise ValueError('Palette changes require a new decoder version and tests')
    if p.get('interpolation') is not False: raise ValueError('Unbounded RGB interpolation is disabled')
    if not 0 < p.get('map_fraction_limit',0) <=1: raise ValueError('Invalid map threshold')
    for rgba in p.get('dry_rgba',[]):
        if len(rgba)!=4 or any(not isinstance(x,int) or not 0<=x<=255 for x in rgba):
            raise ValueError('Dry colors must be exact RGBA integer tuples')
        if rgba[3]==255 and rgba[:3] in p['palette']:raise ValueError('Echo color cannot be declared dry')
    if p.get('dry_rgba') and not p.get('dry_evidence','').strip():
        raise ValueError('A dry-color rule requires documented source evidence')
    return p


def decode(rgba, policy=None):
    p=DEFAULT if policy is None else policy
    a=np.asarray(rgba)
    if a.ndim!=3 or a.shape[-1] not in (3,4):raise ValueError('Expected H x W x RGB(A)')
    if a.shape[-1]==3:
        a=np.concatenate([a,np.full((*a.shape[:2],1),255,dtype=a.dtype)],axis=-1)
    out=np.full(a.shape[:2],np.nan,np.float32)
    for color,value in zip(COLORS,VALUES):
        out[(a[...,3]==255)&(a[...,:3]==color).all(axis=-1)]=value
    for dry in p.get('dry_rgba',[]):out[(a==dry).all(axis=-1)]=0.
    return out


def classify(rgba, policy=None, review_action=''):
    p=DEFAULT if policy is None else policy
    a=np.asarray(rgba);grid=decode(a,p)
    map_fraction=float(np.mean((a[...,:3]==p['map_background_rgb']).all(axis=-1)))
    info=dict(map_fraction=map_fraction,decoded_fraction_before_quarantine=float(np.isfinite(grid).mean()),
              dry_pixels_before_quarantine=int(np.sum(grid==0)),
              unknown_pixels_before_quarantine=int(np.sum(~np.isfinite(grid))),
              extreme_pixels_before_quarantine=int(np.sum(grid>=45)))
    if review_action not in ('','accept','exclude'):raise ValueError('Invalid review action')
    if review_action=='exclude':status='excluded_source_review'
    elif map_fraction>=p['map_fraction_limit'] and review_action!='accept':status='quarantine_map_review'
    elif not np.isfinite(grid).any():status='quarantine_no_decodable_observation'
    # A fully dry image needs confirmation that acquisition actually occurred.
    elif not np.any(grid>0) and review_action!='accept':status='quarantine_empty_echo_review'
    else:status='accepted_partial' if not np.isfinite(grid).all() else 'accepted_full'
    if not status.startswith('accepted_'):grid[:]=np.nan
    info.update(status=status,valid_fraction=float(np.isfinite(grid).mean()),
                extreme_pixels=int(np.sum(grid>=45)),dry_pixels=int(np.sum(grid==0)))
    return grid,info


def event_mask(stored_log1p, valid_mask, threshold_dbz):
    x=np.asarray(stored_log1p,dtype=np.float32)
    # Same dtype and transform as the builder, avoiding expm1( log1p(30) ) < 30.
    threshold=np.log1p(np.float32(threshold_dbz))
    return np.asarray(valid_mask,bool)&np.isfinite(x)&(x>=threshold)
