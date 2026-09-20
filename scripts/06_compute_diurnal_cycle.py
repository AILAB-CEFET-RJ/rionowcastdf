#!/usr/bin/env python3
"""CorrDiff Fase 6 — ciclo diurno condicionado à disponibilidade.

Radar: dBZ, armazenado pelo builder como log1p(clip(dBZ, 0, None)).
A fase mantém UTC e também hora civil local (America/Sao_Paulo por padrão),
usa regras históricas de timezone/DST e gera um perfil local padronizado por
estação para reduzir confundimento sazonal.
"""
from __future__ import annotations

import argparse, json, logging, math, shutil, sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import zarr

PHASE_VERSION = "phase6-diurnal-v1-availability-conditioned-season-standardized"
RAW = ["tcwv","t2m","u10","v10","t_850","r_850","u_850","v_850","t_500","r_500","u_500","v_500"]
DERIVED = ["wind_speed_10","wind_speed_850","wind_speed_500","delta_t_500_850","delta_t_850_surface","delta_r_500_850","bulk_wind_diff_10_850","bulk_wind_diff_850_500"]
ALL = RAW + DERIVED
UNITS = {
    "tcwv":"kg m^-2","t2m":"K","u10":"m s^-1","v10":"m s^-1",
    "t_850":"K","r_850":"%","u_850":"m s^-1","v_850":"m s^-1",
    "t_500":"K","r_500":"%","u_500":"m s^-1","v_500":"m s^-1",
    "wind_speed_10":"m s^-1","wind_speed_850":"m s^-1","wind_speed_500":"m s^-1",
    "delta_t_500_850":"K","delta_t_850_surface":"K","delta_r_500_850":"percentage points",
    "bulk_wind_diff_10_850":"m s^-1","bulk_wind_diff_850_500":"m s^-1",
}
THRESHOLDS = [20,25,30,35,40,45]
REPORT_EVENTS = ["gt_0","ge_20","ge_30","ge_40","ge_45"]
SEASONS = {"DJF":"DJF_verao","MAM":"MAM_outono","JJA":"JJA_inverno","SON":"SON_primavera"}


def args():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset-dir",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,default=Path("analysis_outputs/06_diurnal"))
    p.add_argument("--local-timezone",default="America/Sao_Paulo")
    p.add_argument("--block-size",type=int,default=0)
    p.add_argument("--skip-predictors",action="store_true")
    p.add_argument("--overwrite",action="store_true")
    return p.parse_args()


def setup_output(path:Path, overwrite:bool):
    if path.exists() and any(path.iterdir()):
        if not overwrite: raise FileExistsError(f"{path} is not empty; use --overwrite")
        shutil.rmtree(path)
    path.mkdir(parents=True,exist_ok=True)


def logger_for(path:Path):
    log=logging.getLogger("corrdiff.phase6"); log.setLevel(logging.INFO); log.handlers.clear()
    fmt=logging.Formatter("%(asctime)s | %(levelname)s | %(message)s","%Y-%m-%d %H:%M:%S")
    sh=logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); log.addHandler(sh)
    fh=logging.FileHandler(path/"phase6.log",encoding="utf-8"); fh.setFormatter(fmt); log.addHandler(fh)
    return log


def open_group(path:Path)->Any:
    try:return zarr.open_group(str(path),mode="r")
    except Exception:return zarr.open(str(path),mode="r")


def infer_unit(v):
    f=v[np.isfinite(v)]
    if not len(f): return "ns"
    m=float(np.nanmedian(np.abs(f.astype(np.float64))))
    return "ns" if m>=1e17 else "us" if m>=1e14 else "ms" if m>=1e11 else "s"


def to_utc(v): return pd.DatetimeIndex(pd.to_datetime(v,unit=infer_unit(v),utc=True))


def season_code(m):
    if m in (12,1,2): return "DJF"
    if m in (3,4,5): return "MAM"
    if m in (6,7,8): return "JJA"
    return "SON"


def add_clock(df,col,tz):
    out=df.copy(); utc=pd.DatetimeIndex(pd.to_datetime(out[col],utc=True)); local=utc.tz_convert(tz)
    out["timestamp_utc"]=utc; out["timestamp_local"]=local
    out["hour_utc"]=utc.hour.astype(np.int8); out["hour_local"]=local.hour.astype(np.int8)
    out["year_utc"]=utc.year.astype(np.int16); out["year_local"]=local.year.astype(np.int16)
    out["month_local"]=local.month.astype(np.int8)
    out["utc_offset_hours"]=np.asarray([x.utcoffset().total_seconds()/3600 for x in local],dtype=np.float32)
    out["season_code"]=out["month_local"].map(season_code); out["season"]=out["season_code"].map(SEASONS)
    out["season_year"]=(out["year_local"]+(out["month_local"]==12).astype(np.int16))
    return out


def stored_threshold(x): return np.float32(np.log1p(np.float32(x)))


def derive(raw):
    u10,v10=raw["u10"],raw["v10"]; u85,v85=raw["u_850"],raw["v_850"]; u50,v50=raw["u_500"],raw["v_500"]
    return {
        "wind_speed_10":np.hypot(u10,v10),"wind_speed_850":np.hypot(u85,v85),"wind_speed_500":np.hypot(u50,v50),
        "delta_t_500_850":raw["t_500"]-raw["t_850"],"delta_t_850_surface":raw["t_850"]-raw["t2m"],
        "delta_r_500_850":raw["r_500"]-raw["r_850"],
        "bulk_wind_diff_10_850":np.hypot(u85-u10,v85-v10),"bulk_wind_diff_850_500":np.hypot(u50-u85,v50-v85),
    }


def metadata(dataset_dir):
    p=dataset_dir/"metadata.json"; return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def utc_ts(x):
    t=pd.Timestamp(x); return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def period(meta,observed):
    a=meta.get("start_date") or meta.get("begin"); b=meta.get("end_date") or meta.get("end"); freq=meta.get("time_frequency") or "1h"
    return (utc_ts(a) if a else observed.min(), utc_ts(b) if b else observed.max(), str(freq))


def expected_clock(start,end,freq,tz):
    return add_clock(pd.DataFrame({"timestamp":pd.date_range(start=start,end=end,freq=freq)}),"timestamp",tz)


def coverage(expected,available,cols):
    e=expected.groupby(cols,as_index=False).size().rename(columns={"size":"expected_timestamps"})
    a=available.groupby(cols,as_index=False).size().rename(columns={"size":"available_timestamps"})
    o=e.merge(a,on=cols,how="left"); o["available_timestamps"]=o["available_timestamps"].fillna(0).astype(int)
    o["missing_timestamps"]=o["expected_timestamps"]-o["available_timestamps"]
    o["coverage_ratio"]=o["available_timestamps"]/o["expected_timestamps"]
    return o


def complete_season_years(expected):
    g=expected.groupby(["season_code","season","season_year"],as_index=False).agg(months_present=("month_local","nunique"))
    return g[g.months_present.eq(3)][["season_code","season","season_year"]]


def aggregate_events(frame,cols):
    rows=[]
    for keys,g in frame.groupby(cols,sort=True):
        if not isinstance(keys,tuple): keys=(keys,)
        base=dict(zip(cols,keys)); vp=float(g.valid_pixels.sum()); patches=float(g.patch_count.sum()); pos=float(g.positive_pixels.sum()); ps=float(g.positive_dbz_sum.sum())
        specs=[("gt_0",0.0)]+[(f"ge_{t}",float(t)) for t in THRESHOLDS]
        for eid,th in specs:
            epcol="positive_pixels" if eid=="gt_0" else f"event_pixels_{eid}"
            ppcol="event_patches_gt_0" if eid=="gt_0" else f"event_patches_{eid}"
            fraccol="event_pixel_fraction_gt_0" if eid=="gt_0" else f"event_pixel_fraction_{eid}"
            evpix=float(g[epcol].sum()); present=g[epcol].to_numpy()>0; vals=g.loc[present,fraccol].to_numpy(dtype=float)
            rows.append({**base,"event_id":eid,"threshold_dbz":th,"available_timestamps":len(g),"valid_pixels":int(vp),"event_pixels":int(evpix),
                "pixel_event_rate":evpix/vp if vp else np.nan,"timestamp_any_event_rate":float(present.mean()),
                "patch_any_event_rate":float(g[ppcol].sum()/patches) if patches else np.nan,
                "mean_event_pixel_fraction_given_event_timestamp":float(np.nanmean(vals)) if len(vals) else np.nan,
                "median_event_pixel_fraction_given_event_timestamp":float(np.nanmedian(vals)) if len(vals) else np.nan,
                "mean_positive_dbz":ps/pos if pos else np.nan,"mean_timestamp_max_dbz":float(g.max_dbz.mean()),"median_timestamp_max_dbz":float(g.max_dbz.median())})
    return pd.DataFrame(rows)


def attach_coverage(events,cov,keys):
    return events.merge(cov[keys+["expected_timestamps","available_timestamps","missing_timestamps","coverage_ratio"]],on=keys+["available_timestamps"],how="left")


def standardize_by_season(df):
    metrics=["pixel_event_rate","timestamp_any_event_rate","patch_any_event_rate","mean_event_pixel_fraction_given_event_timestamp","mean_positive_dbz","mean_timestamp_max_dbz"]
    rows=[]
    for (eid,th,h),g in df.groupby(["event_id","threshold_dbz","hour_local"],sort=True):
        row={"event_id":eid,"threshold_dbz":th,"hour_local":int(h),"n_seasons":int(g.season_code.nunique()),"standardization":"equal-weight DJF/MAM/JJA/SON; complete season-years only"}
        for m in metrics: row[m]=float(g[m].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def predictor_stats(df,cols):
    rows=[]
    for keys,g in df.groupby(cols,sort=True):
        if not isinstance(keys,tuple): keys=(keys,)
        base=dict(zip(cols,keys))
        for name in ALL:
            v=g[name].dropna().to_numpy(float)
            if not len(v): continue
            rows.append({**base,"predictor":name,"source":"raw" if name in RAW else "derived","unit":UNITS[name],"available_timestamps":len(v),
                "mean":float(v.mean()),"std":float(v.std(ddof=1)) if len(v)>1 else np.nan,"p10":float(np.quantile(v,.1)),"median":float(np.quantile(v,.5)),"p90":float(np.quantile(v,.9))})
    return pd.DataFrame(rows)


def peak_trough(df,hcol,metric):
    out=[]
    for eid in REPORT_EVENTS:
        t=df[df.event_id.eq(eid)].dropna(subset=[metric])
        if t.empty: continue
        p=t.loc[t[metric].idxmax()]; q=t.loc[t[metric].idxmin()]
        out.append({"event_id":eid,"metric":metric,"peak_hour":int(p[hcol]),"peak_value":float(p[metric]),"trough_hour":int(q[hcol]),"trough_value":float(q[metric]),"absolute_amplitude":float(p[metric]-q[metric]),"peak_to_trough_ratio":float(p[metric]/q[metric]) if q[metric]>0 else None})
    return out


def main():
    a=args(); setup_output(a.output_dir,a.overwrite); log=logger_for(a.output_dir)
    root=open_group(a.dataset_dir/"train.zarr"); xds,yds,mds,tds=root["input"],root["target"],root["mask"],root["timestamps"]
    n=int(yds.shape[0]); traw=np.asarray(tds[:]); uraw,inv=np.unique(traw,return_inverse=True); utc=to_utc(uraw); nt=len(uraw); patch_count=np.bincount(inv,minlength=nt).astype(np.int32)
    meta=metadata(a.dataset_dir); start,end,freq=period(meta,utc); exp=expected_clock(start,end,freq,a.local_timezone); av=add_clock(pd.DataFrame({"timestamp":utc}),"timestamp",a.local_timezone)

    cov_u=coverage(exp,av,["hour_utc"]); cov_l=coverage(exp,av,["hour_local"]); ycov_u=coverage(exp,av,["year_utc","hour_utc"]); ycov_l=coverage(exp,av,["year_local","hour_local"])
    complete=complete_season_years(exp); expc=exp.merge(complete,on=["season_code","season","season_year"],how="inner"); avc=av.merge(complete,on=["season_code","season","season_year"],how="inner")
    scov=coverage(expc,avc,["season_code","season","hour_local"])
    for name,df in [("hourly_coverage_utc",cov_u),("hourly_coverage_local",cov_l),("year_hour_coverage_utc",ycov_u),("year_hour_coverage_local",ycov_l),("season_hour_coverage_local",scov)]: df.to_parquet(a.output_dir/f"{name}.parquet",index=False)

    bs=a.block_size or int(getattr(yds,"chunks",(256,))[0]); blocks=math.ceil(n/bs)
    valid=np.zeros(nt,np.int64); pos=np.zeros(nt,np.int64); possum=np.zeros(nt,np.float64); mx=np.zeros(nt,np.float32)
    ep={t:np.zeros(nt,np.int64) for t in THRESHOLDS}; pp={"gt_0":np.zeros(nt,np.int32),**{f"ge_{t}":np.zeros(nt,np.int32) for t in THRESHOLDS}}
    log.info("="*72); log.info("PHASE 6 — RADAR DIURNAL CYCLE"); log.info("patches=%d timestamps=%d timezone=%s",n,nt,a.local_timezone)
    for b,s in enumerate(range(0,n,bs),1):
        e=min(n,s+bs); gids=inv[s:e]; st=np.asarray(yds[s:e,0],np.float32); mask=np.asarray(mds[s:e,0],np.float32)>.5; ok=mask&np.isfinite(st); dbz=np.expm1(st).astype(np.float32); positive=ok&(st>0)
        np.add.at(valid,gids,ok.sum((1,2),dtype=np.int64)); np.add.at(pos,gids,positive.sum((1,2),dtype=np.int64)); np.add.at(possum,gids,np.where(positive,dbz,0).sum((1,2),dtype=np.float64))
        m=np.where(ok,dbz,-np.inf).max((1,2)); m[~np.isfinite(m)]=0; np.maximum.at(mx,gids,m)
        np.add.at(pp["gt_0"],gids,positive.reshape(len(gids),-1).any(1).astype(np.int32))
        for t in THRESHOLDS:
            ev=ok&(st>=stored_threshold(t)); c=ev.sum((1,2),dtype=np.int64); np.add.at(ep[t],gids,c); np.add.at(pp[f"ge_{t}"],gids,ev.reshape(len(gids),-1).any(1).astype(np.int32))
        if b%250==0 or b==blocks: log.info("Radar: %d/%d blocks",b,blocks)

    ts=add_clock(pd.DataFrame({"timestamp":utc,"patch_count":patch_count,"valid_pixels":valid,"positive_pixels":pos,"positive_dbz_sum":possum,"max_dbz":mx}),"timestamp",a.local_timezone)
    ts["event_pixel_fraction_gt_0"]=np.divide(pos,valid,out=np.full(nt,np.nan),where=valid>0); ts["mean_positive_dbz"]=np.divide(possum,pos,out=np.full(nt,np.nan),where=pos>0); ts["event_patches_gt_0"]=pp["gt_0"]
    for t in THRESHOLDS:
        eid=f"ge_{t}"; ts[f"event_pixels_{eid}"]=ep[t]; ts[f"event_patches_{eid}"]=pp[eid]; ts[f"event_pixel_fraction_{eid}"]=np.divide(ep[t],valid,out=np.full(nt,np.nan),where=valid>0)
    ts.to_parquet(a.output_dir/"timestamp_event_metrics.parquet",index=False)

    eu=attach_coverage(aggregate_events(ts,["hour_utc"]),cov_u,["hour_utc"]); el=attach_coverage(aggregate_events(ts,["hour_local"]),cov_l,["hour_local"])
    ss=ts.merge(complete,on=["season_code","season","season_year"],how="inner"); sel=attach_coverage(aggregate_events(ss,["season_code","season","hour_local"]),scov,["season_code","season","hour_local"]); std=standardize_by_season(sel)
    eu.to_parquet(a.output_dir/"hourly_event_rates_utc.parquet",index=False); el.to_parquet(a.output_dir/"hourly_event_rates_local.parquet",index=False); sel.to_parquet(a.output_dir/"season_hour_event_rates_local.parquet",index=False); std.to_parquet(a.output_dir/"standardized_hourly_event_rates_local.parquet",index=False)

    pred_done=False
    if not a.skip_predictors:
        channels=list(root.attrs.get("channels",[])) or meta.get("channels") or meta.get("input_channels") or []
        missing=[x for x in RAW if x not in channels]
        if missing: raise RuntimeError(f"Missing channels: {missing}")
        idx={x:channels.index(x) for x in RAW}; sums=np.zeros((nt,len(ALL)),np.float64); counts=np.zeros((nt,len(ALL)),np.int64); ibs=a.block_size or int(getattr(xds,"chunks",(256,))[0]); iblocks=math.ceil(n/ibs)
        log.info("="*72); log.info("PHASE 6 — PREDICTOR DIURNAL CYCLE")
        for b,s in enumerate(range(0,n,ibs),1):
            e=min(n,s+ibs); gids=inv[s:e]; block=np.asarray(xds[s:e],np.float32); raw={x:block[:,idx[x]] for x in RAW}; der=derive(raw)
            for j,name in enumerate(ALL):
                arr=raw[name] if name in raw else der[name]; fin=np.isfinite(arr); ps=np.where(fin,arr,0).sum((1,2),dtype=np.float64); pc=fin.sum((1,2),dtype=np.int64); np.add.at(sums[:,j],gids,ps); np.add.at(counts[:,j],gids,pc)
            if b%250==0 or b==iblocks: log.info("Input: %d/%d blocks",b,iblocks)
        means=np.divide(sums,counts,out=np.full_like(sums,np.nan),where=counts>0); pred=add_clock(pd.DataFrame({"timestamp":utc}),"timestamp",a.local_timezone)
        for j,name in enumerate(ALL): pred[name]=means[:,j]
        pred.to_parquet(a.output_dir/"predictor_timestamp_means.parquet",index=False)
        predictor_stats(pred,["hour_utc"]).to_parquet(a.output_dir/"hourly_predictor_statistics_utc.parquet",index=False); predictor_stats(pred,["hour_local"]).to_parquet(a.output_dir/"hourly_predictor_statistics_local.parquet",index=False)
        predictor_stats(pred.merge(complete,on=["season_code","season","season_year"],how="inner"),["season_code","season","hour_local"]).to_parquet(a.output_dir/"season_hour_predictor_statistics_local.parquet",index=False); pred_done=True

    warnings=[]
    if cov_u.coverage_ratio.min()<.5: warnings.append("At least one UTC hour has <50% temporal coverage.")
    if cov_l.coverage_ratio.min()<.5: warnings.append("At least one local hour has <50% temporal coverage.")
    if ycov_l.coverage_ratio.min()<.25: warnings.append("At least one year-local-hour cell has <25% coverage; inspect year_hour_coverage_local.parquet before interpreting individual years.")
    offsets=av.groupby("utc_offset_hours").size().sort_index().to_dict()
    summary={"phase_version":PHASE_VERSION,"dataset_patches":n,"available_unique_timestamps":nt,"expected_timestamps":len(exp),"temporal_coverage_ratio":nt/len(exp),"period_start_utc":str(start),"period_end_utc":str(end),"time_frequency":freq,"local_timezone":a.local_timezone,
        "observed_utc_offset_hours_counts":{str(k):int(v) for k,v in offsets.items()},"fixed_thresholds_dbz":THRESHOLDS,"fixed_threshold_comparison_domain":"stored_log1p_float32","radar_target_semantics":"dBZ clipped at zero by builder, then stored as log1p; continuous diagnostics use expm1(stored_target)","negative_dbz_preserved":False,
        "season_standardization":"equal-weight DJF/MAM/JJA/SON using complete season-years only","predictor_diurnal_cycle_computed":pred_done,"raw_local_peak_trough_timestamp_event":peak_trough(el,"hour_local","timestamp_any_event_rate"),"season_standardized_local_peak_trough_timestamp_event":peak_trough(std,"hour_local","timestamp_any_event_rate"),"warnings":warnings,
        "methodological_notes":["Primary occurrence metric: P(event | available timestamp, hour).","Season-standardized local profile helps separate diurnal structure from unequal seasonal composition.","America/Sao_Paulo historical timezone rules handle past DST transitions.","Pixel rates describe the stored overlapping-patch distribution, not a de-duplicated radar field climatology.","mean_event_pixel_fraction_given_event_timestamp separates spatial extent from occurrence frequency."]}
    (a.output_dir/"analysis_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    log.info("="*72); log.info("PHASE 6 COMPLETE — output=%s",a.output_dir)

if __name__=="__main__": main()
