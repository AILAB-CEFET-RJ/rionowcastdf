"""Read-only interpolation and target validation retained from the 14n delivery."""
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from pair_contract import digest,qi,literal,choose

def strict_bool(series):
    if series.isna().any() or not series.isin([True,False]).all():
        raise ValueError('Expected non-null boolean column: '+series.name)
    return series.astype(bool)

def regrid(rows, lat, lon):
    if rows.empty: raise ValueError('No ERA5 rows at exact UTC timestamp')
    if rows[['lat','lon']].duplicated().any(): raise ValueError('Duplicate ERA5 coordinates')
    if not np.isfinite(rows[['lat','lon','value']].to_numpy(dtype=float)).all():
        raise ValueError('ERA5 contains nonfinite values/coordinates')
    ys=np.sort(rows.lat.unique());xs=np.sort(rows.lon.unique())
    if len(ys)<2 or len(xs)<2 or len(rows)!=len(ys)*len(xs): raise ValueError('Incomplete ERA5 rectangular grid')
    values=rows.pivot(index='lat',columns='lon',values='value').reindex(index=ys,columns=xs).to_numpy()
    interp=RegularGridInterpolator((ys,xs),values,method='linear',bounds_error=True)
    target=interp(np.column_stack([lat.ravel(),lon.ravel()])).reshape(lat.shape).astype(np.float32)
    if not np.isfinite(target).all(): raise ValueError('Nonfinite ERA5 interpolation')
    return target

def read_channel(engine, db, stamp, ch, config, lat, lon, checked, provenance):
    sources=db.execute('SELECT DISTINCT path FROM metrics WHERE timestamp_utc=? AND channel=?',(stamp.isoformat(),ch)).fetchall()
    if len(sources)!=1: raise ValueError(f'{stamp} {ch}: expected exactly one source file, got {len(sources)}')
    path=Path(sources[0][0])
    stored=db.execute('SELECT fingerprint,status,details FROM files WHERE path=?',(str(path),)).fetchone()
    if not stored or stored[1]!='ok': raise ValueError('Source was not successfully inventoried')
    stat=path.stat(); fp=f'{stat.st_size}:{stat.st_mtime_ns}'
    if fp!=stored[0]: raise ValueError('ERA5 changed since inventory: '+str(path))
    details=json.loads(stored[2])
    if str(path) not in checked:
        checked.add(str(path));provenance.append({'path':str(path),'fingerprint':fp,'metadata':details.get('metadata',{})})
    kind='pressure' if '_' in ch else 'single'
    var,level=ch.rsplit('_',1) if kind=='pressure' else (ch,None)
    col=details['variable_columns'][var];t=details['time_column'];names=details['columns']
    la=choose(names,['latitude','lat']);lo=choose(names,['longitude','lon'])
    where=f'try_cast({qi(t)} AS TIMESTAMPTZ)={literal(stamp.isoformat())}::TIMESTAMPTZ'
    if level:
        lv=details['pressure_column'];div='/100.0' if config['pressure_unit']=='Pa' else ''
        where+=f' AND try_cast({qi(lv)} AS DOUBLE){div}={int(level)}'
    sql=f'''SELECT try_cast({qi(la)} AS DOUBLE) AS lat,
      ((try_cast({qi(lo)} AS DOUBLE)+180)%360+360)%360-180 AS lon,
      try_cast({qi(col)} AS DOUBLE) AS value
      FROM read_parquet({literal(path)},hive_partitioning=false) WHERE {where}'''
    rows=engine.execute(sql).fetchdf()
    after=path.stat()
    if fp!=f'{after.st_size}:{after.st_mtime_ns}': raise ValueError('ERA5 changed during read')
    return regrid(rows,lat,lon)

def inside(root,relative):
    path=(root/relative).resolve()
    if root not in path.parents: raise ValueError('Cache path escapes root')
    return path

def read_radar(row,policy,shape,geometry_mask,palette):
    path=inside(policy,row.target_path)
    if digest(path)!=row.target_sha256: raise ValueError('Policy target checksum mismatch')
    with np.load(path,allow_pickle=False) as z:
        raw=z['target_code'].copy();mask=z['observation_mask'].copy();classes=z['observation_class']
        if raw.shape!=shape or mask.shape!=shape or mask.dtype!=bool:raise ValueError('Target shape/dtype mismatch')
        if classes.shape!=shape or not np.isin(classes,[0,1,2]).all():raise ValueError('Unknown observation class')
        if (mask & ~geometry_mask).any():raise ValueError('Target exceeds support region')
        if not np.array_equal(mask,classes!=0):raise ValueError('Target classes disagree with support')
        if not np.isfinite(raw[mask]).all() or not np.isnan(raw[~mask]).all():raise ValueError('Target finite/unknown contract broken')
        if not np.all(raw[classes==1]==0) or not np.isin(raw[classes==2],palette).all():raise ValueError('Target encoding broken')
    return raw,mask

