"""Time and geometry contract for a read-only Sumaré diagnostic pilot.
Palette matches are NOT an observation-validity mask.
"""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import hashlib
import json
from pathlib import Path
import numpy as np

VERSION = '14i-alignment-pilot-v1'
UTC = timezone.utc
LOCAL = ZoneInfo('America/Sao_Paulo')
CONTRACT = {
    'version': VERSION,
    'filename_timezone': 'America/Sao_Paulo',
    'source_width': 656, 'source_height': 654,
    'radar_latitude': -22.955139, 'radar_longitude': -43.248278,
    'display_range_km': 138.9, 'earth_radius_m': 6371008.8,
    'projection': 'spherical AEQD, radar centred, north up, east right',
    'lat_range': [-23.5, -22.25], 'lon_range': [-44., -42.5],
    'grid_degree_step': 2.0/111.0,
    'sampling': 'nearest native source pixel; numpy.rint; no spatial/temporal averaging',
    'reference_repo': 'AILAB-CEFET-RJ/rionowcast-radar-stations',
    'reference_commit': '65201b6787bb26997da5bc2145807d1a8fd4be33',
    'reference_config': 'configs/radar_sumare_historical_audited_v1.json',
    'legacy_repo_commit': '554c019356481832a62d7d0d46d3d07df1bc29fe',
    'palette_rgb': [[197,0,197],[227,6,5],[255,112,0],[195,230,0],
                    [4,85,4],[19,122,19],[0,167,12]],
    'palette_dbz': [50,45,40,35,30,25,20],
    'alpha_for_palette': 255,
    'dry_rule': None,
    'map_screen_rgb': [242,239,233], 'map_screen_fraction': 0.01,
    'map_screen_is_proof': False,
    'geometric_circle_is_observation_mask': False,
    'production_ready': False,
}

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()

def utc_candidates(local_naive):
    """Round-trip rejects nonexistent times; return both folds for repeated hours."""
    if local_naive.tzinfo is not None:
        raise ValueError('Expected a naive local filename time')
    candidates=set()
    for fold in (0,1):
        u=local_naive.replace(tzinfo=LOCAL,fold=fold).astimezone(UTC)
        if u.astimezone(LOCAL).replace(tzinfo=None)==local_naive:
            candidates.add(u)
    return sorted(candidates)

def local_to_utc(name):
    dt=datetime.strptime(name,'%Y_%m_%d_%H_%M')
    choices=utc_candidates(dt)
    if len(choices)!=1:
        raise ValueError(f'Local timestamp {name}: {len(choices)} possible UTC instants; provide an explicit UTC anchor. A repeated filename still requires source review.')
    return choices[0]

def parse_utc(value):
    dt=datetime.fromisoformat(value.replace('Z','+00:00'))
    if dt.tzinfo is None or dt.utcoffset().total_seconds()!=0:
        raise ValueError('UTC anchor must include Z or +00:00')
    if dt.second or dt.microsecond: raise ValueError('Anchor must have minute precision')
    return dt.astimezone(UTC)

def png_path(root, utc):
    if utc.tzinfo is None: raise ValueError('UTC timestamp must be timezone-aware')
    local=utc.astimezone(LOCAL)
    return Path(root)/local.strftime('%Y/%m/%d/%Y_%m_%d_%H_%M.png')

def project_pixels(lat,lon):
    """Matches spherical AEQD candidate adopted by the reference repository."""
    p=np.deg2rad(lat); l=np.deg2rad(lon)
    p0=np.deg2rad(CONTRACT['radar_latitude']); l0=np.deg2rad(CONTRACT['radar_longitude'])
    dl=l-l0
    c=np.arccos(np.clip(np.sin(p0)*np.sin(p)+np.cos(p0)*np.cos(p)*np.cos(dl),-1.,1.))
    k=np.ones_like(c); np.divide(c,np.sin(c),out=k,where=c>1e-12)
    east=CONTRACT['earth_radius_m']*k*np.cos(p)*np.sin(dl)
    north=CONTRACT['earth_radius_m']*k*(np.cos(p0)*np.sin(p)-np.sin(p0)*np.cos(p)*np.cos(dl))
    radius=CONTRACT['display_range_km']*1000.
    col=327.5+east/(radius/327.5)
    row=326.5-north/(radius/326.5)
    return row,col,np.hypot(east,north)<=radius

def geometry():
    lat=np.arange(*CONTRACT['lat_range'],CONTRACT['grid_degree_step'])
    lon=np.arange(*CONTRACT['lon_range'],CONTRACT['grid_degree_step'])
    xx,yy=np.meshgrid(lon,lat)
    row,col,inside=project_pixels(yy,xx)
    py=np.rint(row).astype(int);px=np.rint(col).astype(int)
    bounds=(py>=0)&(py<654)&(px>=0)&(px<656)
    valid=inside&bounds
    oldx=((xx+44.)/1.5*655).astype(int)
    oldy=((-22.25-yy)/1.25*653).astype(int)
    # Inverse spherical AEQD for native pixels: exact native ROI diagnostics.
    y,x=np.indices((654,656))
    east=(x-327.5)*(138900./327.5);north=(326.5-y)*(138900./326.5)
    rho=np.hypot(east,north);c=rho/CONTRACT['earth_radius_m']
    p0=np.deg2rad(CONTRACT['radar_latitude']);l0=np.deg2rad(CONTRACT['radar_longitude'])
    ratio=np.divide(north*np.sin(c),rho,out=np.zeros_like(rho),where=rho!=0)
    native_lat=np.rad2deg(np.arcsin(np.cos(c)*np.sin(p0)+ratio*np.cos(p0)))
    native_lon=np.rad2deg(l0+np.arctan2(east*np.sin(c),rho*np.cos(p0)*np.cos(c)-north*np.sin(p0)*np.sin(c)))
    roi=(rho<=138900.)&(native_lat>=-23.5)&(native_lat< -22.25)&(native_lon>=-44.)&(native_lon< -42.5)
    return dict(lat=yy,lon=xx,px=px,py=py,geometric_mask=valid,
                legacy_px=oldx,legacy_py=oldy,native_roi=roi)

def decode_palette(rgba):
    if rgba.shape!=(654,656,4): raise ValueError(f'Unexpected RGBA shape {rgba.shape}')
    z=np.full(rgba.shape[:2],np.nan,dtype=np.float32)
    for rgb,value in zip(CONTRACT['palette_rgb'],CONTRACT['palette_dbz']):
        z[(rgba[:,:,3]==255)&np.all(rgba[:,:,:3]==rgb,axis=2)]=value
    return z

def analyze_array(rgba,g):
    native=decode_palette(rgba);echo=np.isfinite(native)
    mask=g['geometric_mask'];px=g['px'];py=g['py']
    corrected=np.full(mask.shape,np.nan,np.float32)
    corrected[mask]=native[py[mask],px[mask]]
    legacy=native[g['legacy_py'],g['legacy_px']]
    sampled=np.zeros((654,656),bool);sampled[py[mask],px[mask]]=True
    map_fraction=float(np.mean((rgba[:,:,3]>0)&np.all(rgba[:,:,:3]==CONTRACT['map_screen_rgb'],axis=2)))
    if np.all(rgba[:,:,3]==0): status='transparent_ambiguous'
    elif map_fraction>=CONTRACT['map_screen_fraction']: status='map_screen_review'
    elif not echo.any(): status='no_exact_palette_review'
    elif not np.isfinite(corrected).any(): status='palette_in_png_not_sampled_in_corrected_grid'
    else: status='palette_sampled_observation_validity_unconfirmed'
    review=corrected.copy()
    if status=='map_screen_review':review[:]=np.nan
    counts=dict(status=status,map_fraction=map_fraction,
        transparent_pixels=int((rgba[:,:,3]==0).sum()),
        native_palette_pixels=int(echo.sum()),native_roi_palette_pixels=int((echo&g['native_roi']).sum()),
        native_palette_pixels_outside_roi=int((echo&~g['native_roi']).sum()),
        corrected_grid_palette_points=int(np.isfinite(corrected).sum()),
        legacy_grid_palette_points=int(np.isfinite(legacy).sum()),
        native_roi_palette_pixels_not_sampled=int((echo&g['native_roi']&~sampled).sum()),
        geometric_grid_points=int(mask.sum()))
    return native,corrected,legacy,review,counts
