"""Exact-hour ERA5 reads with portable paths and monthly materialization."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from .contract_common import ro,read,sha,inside,dump


def qi(v):return '"'+str(v).replace('"','""')+'"'
def literal(v):return "'"+str(v).replace("'","''")+"'"
def choose(names,candidates):
    for n in candidates:
        if n in names:return n
    raise ValueError('Missing coordinate '+str(candidates))


def regrid(rows,lat,lon):
    if rows.empty or rows[['lat','lon']].duplicated().any():raise ValueError('Empty/duplicate ERA5 grid')
    if not np.isfinite(rows[['lat','lon','value']].to_numpy(dtype=float)).all():raise ValueError('Nonfinite ERA5')
    ys=np.sort(rows.lat.unique());xs=np.sort(rows.lon.unique())
    if len(ys)<2 or len(xs)<2 or len(rows)!=len(ys)*len(xs):raise ValueError('Incomplete ERA5 rectangle')
    values=rows.pivot(index='lat',columns='lon',values='value').reindex(index=ys,columns=xs).to_numpy()
    interp=RegularGridInterpolator((ys,xs),values,bounds_error=True)
    result=interp(np.column_stack([lat.ravel(),lon.ravel()])).reshape(lat.shape).astype(np.float32)
    if not np.isfinite(result).all():raise ValueError('Nonfinite interpolation')
    return result


class Inventory:
    def __init__(self,c,out):
        self.c=c;self.out=Path(out);base=Path(c['pairing_dir'])
        run=read(base/'run_config.json');self.config=run['inventory_config']
        if self.config.get('pressure_unit') not in ('Pa','hPa'):raise ValueError('Unsupported pressure unit')
        dbpath=base/'era5_inventory_snapshot.sqlite'
        if sha(dbpath)!=run['era5_snapshot_sha256']:raise ValueError('ERA5 snapshot checksum mismatch')
        self.db=ro(dbpath);self.digest=sha(dbpath)
        self.records={p:(fp,status,json.loads(details)) for p,fp,status,details in self.db.execute('SELECT path,fingerprint,status,details FROM files')}
        self.identity=read(c['era5_identity_file']) if c.get('era5_identity_file') else {}
        saved=self.out/'era5_identity.json'
        self.provenance=read(saved) if saved.exists() else {}
        self.checked={}
    def location(self,original):
        parts=Path(original).parts
        # Locate the explicit single/pressure partition root; no filename-only matching.
        indices=[i for i,v in enumerate(parts) if v in ('single','pressure')]
        if len(indices)!=1:raise ValueError('Cannot remap ERA5 path: '+original)
        return inside(self.c['era5_root'],Path(*parts[indices[0]:]))
    def verify(self,original):
        p=self.location(original);fp,status,d=self.records[original];s=p.stat();now=f'{s.st_size}:{s.st_mtime_ns}'
        if status!='ok':raise ValueError('ERA5 source not inventoried successfully')
        if original not in self.checked:
            digest=sha(p)
            expected=self.provenance.get(original,self.identity.get(original,{})).get('sha256')
            if expected:
                if digest!=expected:raise ValueError('ERA5 content changed: '+str(p))
            elif fp!=now:raise ValueError('ERA5 metadata changed; use rsync -a or trusted era5_identity_file: '+str(p))
            self.checked[original]=now
            self.provenance[original]={'sha256':digest,'inventory_fingerprint':fp,'source_metadata':d.get('metadata',{}),'variables':d['variable_columns']}
            dump(self.out/'era5_identity.json',self.provenance)
        elif now!=self.checked[original]:raise ValueError('ERA5 changed during execution')
        return p,d
    def close(self):self.db.close()


class Month:
    def __init__(self,engine,inventory,stamps,channels):
        self.e=engine;self.inv=inventory;self.tables={};self.sources={}
        for t in stamps:
            for ch in channels:
                paths=inventory.db.execute('SELECT DISTINCT path FROM metrics WHERE timestamp_utc=? AND channel=?',(t.isoformat(),ch)).fetchall()
                if len(paths)!=1:raise ValueError(f'{t} {ch}: expected one ERA5 source, got {len(paths)}')
                self.sources[(t.isoformat(),ch)]=paths[0][0]
        engine.register('wanted_times',pd.DataFrame({'wanted_time':stamps}))
        try:
            for original in sorted(set(self.sources.values())):
                p,d=inventory.verify(original);t=d['time_column'];names=d['columns'];pressure=d.get('pressure_column')
                cols=[choose(names,['latitude','lat']),choose(names,['longitude','lon'])]+list(d['variable_columns'].values())
                if pressure in names:cols.append(pressure)
                cols=', '.join(qi(x) for x in dict.fromkeys(cols) if x!=t)
                where=''
                if pressure in names:
                    factor='/100.0' if inventory.config['pressure_unit']=='Pa' else ''
                    where=f' AND try_cast(p.{qi(pressure)} AS DOUBLE){factor} IN (500,850)'
                table=f'month_{len(self.tables)}'
                engine.execute(f'''CREATE TEMP TABLE {table} AS SELECT {cols},try_cast(p.{qi(t)} AS TIMESTAMPTZ) AS {qi(t)}
                  FROM read_parquet({literal(p)},hive_partitioning=false) p
                  WHERE try_cast(p.{qi(t)} AS TIMESTAMPTZ) IN (SELECT wanted_time FROM wanted_times){where}''')
                engine.execute(f'CREATE INDEX idx_{table} ON {table} ({qi(t)})');self.tables[original]=(table,d)
        finally:engine.unregister('wanted_times')
    def field(self,t,ch,lat,lon):
        original=self.sources[(t.isoformat(),ch)];table,d=self.tables[original]
        var,level=ch.rsplit('_',1) if '_' in ch else (ch,None)
        la=choose(d['columns'],['latitude','lat']);lo=choose(d['columns'],['longitude','lon'])
        where=f'{qi(d["time_column"])}={literal(t.isoformat())}::TIMESTAMPTZ'
        if level:
            factor='/100.0' if self.inv.config['pressure_unit']=='Pa' else ''
            where+=f' AND try_cast({qi(d["pressure_column"])} AS DOUBLE){factor}={int(level)}'
        rows=self.e.execute(f'''SELECT try_cast({qi(la)} AS DOUBLE) AS lat,
          ((try_cast({qi(lo)} AS DOUBLE)+180)%360+360)%360-180 AS lon,
          try_cast({qi(d['variable_columns'][var])} AS DOUBLE) AS value FROM {table} WHERE {where}''').fetchdf()
        return regrid(rows,lat,lon)
    def close(self):
        for p,(table,d) in self.tables.items():
            self.inv.verify(p);self.e.execute('DROP TABLE '+table)
