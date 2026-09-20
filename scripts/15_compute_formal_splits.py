#!/usr/bin/env python3
"""CorrDiff - Fase 15: split temporal formal para treino/val/test/stress-OOD."""
from __future__ import annotations
import argparse, json, logging, shutil, sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
try:
    import zarr
except ImportError as exc:
    raise SystemExit('Fase 15 requer zarr: pip install zarr') from exc

PHASE_VERSION='phase15-formal-splits-v1-purged-temporal-ood2024'
SPLITS=['train','validation','test_primary','test_stress_ood','excluded_purge']
CODES={'excluded_purge':0,'train':1,'validation':2,'test_primary':3,'test_stress_ood':4}
EVENTS={'gt_0':0.0,'ge_20':20.0,'ge_30':30.0,'ge_40':40.0,'ge_45':45.0}
SEASONS=['DJF','MAM','JJA','SON']
RAW=['tcwv','t2m','u10','v10','t_850','r_850','u_850','v_850','t_500','r_500','u_500','v_500']
DERIVED=['wind_speed_10','wind_speed_850','wind_speed_500','delta_t_500_850','delta_t_850_surface','delta_r_500_850','bulk_wind_diff_10_850','bulk_wind_diff_850_500']
ALL=RAW+DERIVED
ALIASES={x:[x] for x in ALL}
ALIASES.update({'t_850':['t_850','t850'],'r_850':['r_850','r850'],'u_850':['u_850','u850'],'v_850':['v_850','v850'],'t_500':['t_500','t500'],'r_500':['r_500','r500'],'u_500':['u_500','u500'],'v_500':['v_500','v500']})

def parse_args():
    p=argparse.ArgumentParser(description='CorrDiff Fase 15: split temporal formal.')
    p.add_argument('--dataset-dir',type=Path,default=Path('datasets/corrdiff_2011_2024'))
    p.add_argument('--phase6-dir',type=Path,default=Path('analysis_outputs/06_diurnal'))
    p.add_argument('--phase13-dir',type=Path,default=Path('analysis_outputs/13_regimes'))
    p.add_argument('--phase14-dir',type=Path,default=Path('analysis_outputs/14_distribution_shift'))
    p.add_argument('--output-dir',type=Path,default=Path('analysis_outputs/15_formal_splits'))
    p.add_argument('--train-end-year',type=int,default=2020)
    p.add_argument('--validation-start-year',type=int,default=2021)
    p.add_argument('--validation-end-year',type=int,default=2022)
    p.add_argument('--primary-test-year',type=int,default=2023)
    p.add_argument('--stress-test-year',type=int,default=2024)
    p.add_argument('--purge-hours',type=int,default=48,help='Horas excluídas em cada lado de cada fronteira.')
    p.add_argument('--local-timezone',default='America/Sao_Paulo')
    p.add_argument('--expected-patches-per-timestamp',type=int,default=12)
    p.add_argument('--allow-variable-patches',action='store_true')
    p.add_argument('--overwrite',action='store_true')
    return p.parse_args()

def setup_output(path,overwrite):
    if path.exists() and any(path.iterdir()):
        if not overwrite: raise FileExistsError(f'{path} não vazio; use --overwrite')
        shutil.rmtree(path)
    path.mkdir(parents=True,exist_ok=True)

def setup_logger(path):
    lg=logging.getLogger('corrdiff.phase15'); lg.setLevel(logging.INFO); lg.handlers.clear()
    fmt=logging.Formatter('%(asctime)s | %(levelname)s | %(message)s','%Y-%m-%d %H:%M:%S')
    for h in [logging.StreamHandler(sys.stdout),logging.FileHandler(path/'phase15.log',encoding='utf-8')]: h.setFormatter(fmt); lg.addHandler(h)
    return lg

def open_zarr(path):
    try: return zarr.open_group(str(path),mode='r')
    except Exception: return zarr.open(str(path),mode='r')

def infer_unit(a):
    f=a[np.isfinite(a)]
    if not len(f): return 'ns'
    m=float(np.nanmedian(np.abs(f.astype(np.float64))))
    return 'ns' if m>=1e17 else 'us' if m>=1e14 else 'ms' if m>=1e11 else 's'

def to_utc(a):
    a=np.asarray(a)
    if np.issubdtype(a.dtype,np.datetime64): return pd.DatetimeIndex(pd.to_datetime(a,utc=True))
    return pd.DatetimeIndex(pd.to_datetime(a,unit=infer_unit(a),utc=True))

def season(m):
    return 'DJF' if m in (12,1,2) else 'MAM' if m in (3,4,5) else 'JJA' if m in (6,7,8) else 'SON'

def ts_col(df):
    for c in ['timestamp_utc','timestamp','time']:
        if c in df.columns:return c
    raise RuntimeError('timestamp_utc/timestamp/time não encontrado')

def normalize_time(df,tz):
    out=df.copy(); ts=pd.to_datetime(out[ts_col(out)],utc=True); out['timestamp_utc']=ts
    local=ts.dt.tz_convert(tz); out['year_utc']=ts.dt.year.astype(np.int16); out['month_local']=local.dt.month.astype(np.int8); out['hour_local']=local.dt.hour.astype(np.int8)
    out['season_code']=pd.Categorical([season(int(x)) for x in out['month_local']],categories=SEASONS,ordered=True)
    return out

def prep_radar(df,tz):
    out=normalize_time(df,tz)
    if 'max_dbz' not in out: raise RuntimeError('max_dbz ausente em timestamp_event_metrics')
    x=out['max_dbz'].to_numpy(float)
    for eid,thr in EVENTS.items(): out[f'event_{eid}']=((x>0) if eid=='gt_0' else (x>=thr)).astype(np.int8)
    if 'positive_pixel_fraction' not in out and 'event_pixel_fraction_gt_0' in out: out['positive_pixel_fraction']=out['event_pixel_fraction_gt_0']
    return out

def canonicalize(df):
    out=df.copy(); available=[]
    for name in ALL:
        found=next((a for a in ALIASES[name] if a in out.columns),None)
        if found is not None:
            if found!=name: out[name]=out[found]
            available.append(name)
    missing=sorted(set(RAW)-set(available))
    if missing: raise RuntimeError(f'predictors raw ausentes: {missing}')
    return out,available

def split_def(args):
    if not(args.train_end_year<args.validation_start_year<=args.validation_end_year<args.primary_test_year<args.stress_test_year): raise ValueError('anos dos splits inválidos')
    rows=[
      ('train',2011,args.train_end_year,True,False,True,'parameter fitting'),
      ('validation',args.validation_start_year,args.validation_end_year,True,False,True,'hyperparameter/model selection'),
      ('test_primary',args.primary_test_year,args.primary_test_year,False,True,False,'primary chronological generalization test'),
      ('test_stress_ood',args.stress_test_year,args.stress_test_year,False,True,False,'stress/OOD test'),]
    return pd.DataFrame([dict(split=s,start_utc=pd.Timestamp(f'{a}-01-01 00:00:00',tz='UTC'),end_utc=pd.Timestamp(f'{b}-12-31 23:00:00',tz='UTC'),model_selection_access=acc,locked=lock,final_refit_eligible_pre_primary_test=refit,role=role) for s,a,b,acc,lock,refit,role in rows])

def boundaries(sd):
    sd=sd.sort_values('start_utc').reset_index(drop=True); rows=[]
    for i in range(len(sd)-1):
        l,r=sd.iloc[i],sd.iloc[i+1]; rows.append(dict(boundary_id=f"{l['split']}__{r['split']}",left_split=l['split'],right_split=r['split'],boundary_utc=r['start_utc']))
    return pd.DataFrame(rows)

def assign_nominal(ts,sd):
    out=np.full(len(ts),'outside_defined_period',dtype=object)
    for _,r in sd.iterrows(): out[(ts>=r.start_utc)&(ts<=r.end_utc)]=r.split
    return out

def purge(ts,nominal,bounds,hours):
    final=nominal.copy(); reason=np.full(len(ts),'',dtype=object)
    if hours<=0:return final,reason
    d=pd.Timedelta(hours=hours)
    for _,b in bounds.iterrows():
        m=(ts>=b.boundary_utc-d)&(ts<b.boundary_utc+d)&np.isin(nominal,[b.left_split,b.right_split]); final[m]='excluded_purge'; reason[m]=b.boundary_id
    return final,reason

def split_coverage(manifest,sd,purge_hours):
    rows=[]
    for _,r in sd.iterrows():
        expected=len(pd.date_range(r.start_utc,r.end_utc,freq='h',tz='UTC')); nominal=int((manifest.nominal_split==r.split).sum()); retained=int((manifest.split==r.split).sum())
        rows.append(dict(split=r.split,nominal_start_utc=r.start_utc,nominal_end_utc=r.end_utc,expected_hours_nominal=expected,available_hours_before_purge=nominal,purged_available_hours=nominal-retained,retained_available_hours=retained,availability_ratio_before_purge=nominal/expected,retained_ratio_vs_nominal_expected=retained/expected,purge_hours_each_side_of_boundary=purge_hours))
    return pd.DataFrame(rows)

def event_balance(df):
    cont=[c for c in ['max_dbz','positive_pixel_fraction','event_pixel_fraction_ge_30','event_pixel_fraction_ge_40','event_pixel_fraction_ge_45'] if c in df]
    rows=[]
    for sp in SPLITS:
        s=df[df.split==sp]
        if s.empty: continue
        row={'split':sp,'n_timestamps':len(s)}
        for eid in EVENTS: row[f'timestamp_event_rate_{eid}']=float(s[f'event_{eid}'].mean())
        for c in cont: row[f'mean_{c}']=float(s[c].mean())
        row.update(max_dbz_p50=float(s.max_dbz.quantile(.5)),max_dbz_p90=float(s.max_dbz.quantile(.9)),max_dbz_p99=float(s.max_dbz.quantile(.99)))
        rows.append(row)
    return pd.DataFrame(rows)

def event_season(df):
    rows=[]
    for sp in SPLITS:
        s=df[df.split==sp]
        if s.empty: continue
        for se in SEASONS:
            q=s[s.season_code.astype(str)==se]
            if q.empty: continue
            row={'split':sp,'season_code':se,'n_timestamps':len(q),'fraction_of_split':len(q)/len(s)}
            for eid in EVENTS: row[f'timestamp_event_rate_{eid}']=float(q[f'event_{eid}'].mean())
            rows.append(row)
    return pd.DataFrame(rows)

def js(p,q):
    p=np.clip(np.asarray(p,float),1e-12,None);q=np.clip(np.asarray(q,float),1e-12,None);p/=p.sum();q/=q.sum();m=.5*(p+q)
    return float(.5*np.sum(p*np.log(p/m))+.5*np.sum(q*np.log(q/m)))

def regime_tables(manifest,assign):
    m=assign[['timestamp_utc','regime_space','regime_id']].merge(manifest[['timestamp_utc','split']],on='timestamp_utc',how='inner',validate='many_to_one')
    rows=[];shifts=[]
    for space in sorted(m.regime_space.unique()):
        ss=m[m.regime_space==space]; ids=sorted(ss.regime_id.unique()); vec={}
        for sp in SPLITS:
            s=ss[ss.split==sp]
            if s.empty:continue
            v=[]
            for rid in ids:
                n=int((s.regime_id==rid).sum()); p=n/len(s);v.append(p);rows.append(dict(regime_space=space,split=sp,regime_id=int(rid),regime=f'R{int(rid)}',n_timestamps=n,split_total_timestamps=len(s),prevalence=p))
            vec[sp]=np.asarray(v)
        if 'train' in vec:
            for sp in ['validation','test_primary','test_stress_ood']:
                if sp not in vec: continue
                d=vec[sp]-vec['train']; shifts.append(dict(regime_space=space,reference_split='train',evaluation_split=sp,jensen_shannon_divergence=js(vec['train'],vec[sp]),total_variation_distance=float(.5*np.abs(d).sum()),max_absolute_prevalence_delta=float(np.abs(d).max())))
    return pd.DataFrame(rows),pd.DataFrame(shifts)

def ks(x,y):
    x=np.sort(np.asarray(x,float));y=np.sort(np.asarray(y,float));x=x[np.isfinite(x)];y=y[np.isfinite(y)]
    if not len(x) or not len(y):return np.nan
    v=np.sort(np.unique(np.concatenate([x,y])));return float(np.max(np.abs(np.searchsorted(x,v,side='right')/len(x)-np.searchsorted(y,v,side='right')/len(y))))

def predictor_shift(pred,manifest,names):
    m=pred[['timestamp_utc']+names].merge(manifest[['timestamp_utc','split']],on='timestamp_utc',how='inner',validate='one_to_one'); tr=m[m.split=='train']; rows=[]
    for sp in ['validation','test_primary','test_stress_ood']:
        ev=m[m.split==sp]
        for name in names:
            x0=tr[name].to_numpy(float);x1=ev[name].to_numpy(float);x0=x0[np.isfinite(x0)];x1=x1[np.isfinite(x1)]
            if not len(x0) or not len(x1):continue
            mu0,mu1=float(x0.mean()),float(x1.mean());sd=float(x0.std())
            rows.append(dict(reference_split='train',evaluation_split=sp,predictor=name,n_reference=len(x0),n_evaluation=len(x1),reference_mean=mu0,evaluation_mean=mu1,standardized_mean_difference=(mu1-mu0)/sd if sd>1e-12 else np.nan,ks_statistic=ks(x0,x1)))
    return pd.DataFrame(rows)

def boundary_audit(manifest,bounds,hours):
    rows=[]
    for _,b in bounds.iterrows():
        l=manifest[manifest.split==b.left_split];r=manifest[manifest.split==b.right_split];last=l.timestamp_utc.max() if len(l) else pd.NaT;first=r.timestamp_utc.min() if len(r) else pd.NaT
        sep=(first-last).total_seconds()/3600 if pd.notna(last) and pd.notna(first) else np.nan
        rows.append(dict(boundary_id=b.boundary_id,left_split=b.left_split,right_split=b.right_split,boundary_utc=b.boundary_utc,purge_hours_each_side=hours,last_retained_left_timestamp=last,first_retained_right_timestamp=first,retained_separation_hours=sep,purged_available_timestamps=int(((manifest.split=='excluded_purge')&(manifest.purge_boundary==b.boundary_id)).sum()),passes_temporal_order=bool(pd.isna(last) or pd.isna(first) or last<first),passes_at_least_requested_radius_gap=bool(np.isnan(sep) or sep>=hours)))
    return pd.DataFrame(rows)

def phase14_snapshot(path):
    out={'source_phase14_dir':str(path),'available':False}; req=[path/'coverage_by_year.parquet',path/'regime_composition_shift.parquet',path/'predictor_shift_anomaly_metrics.parquet',path/'event_rate_decomposition.parquet']
    if not all(x.exists() for x in req): return out
    cov,reg,pred,dec=[pd.read_parquet(x) for x in req];out['available']=True;recent='RECENT_2023_2024_vs_pre2023'
    out['coverage_ratio']={str(y):float(cov.loc[cov.year_utc==y,'availability_ratio'].iloc[0]) for y in [2023,2024] if (cov.year_utc==y).any()}
    out['recent_regime_shift']=reg[reg.comparison_id==recent][['regime_space','jensen_shannon_divergence','total_variation_distance','max_absolute_prevalence_delta']].to_dict('records')
    t=pred[pred.comparison_id==recent].copy();t['abs_smd']=t.standardized_mean_difference.abs();out['top_recent_anomaly_predictor_shifts']=t.sort_values('abs_smd',ascending=False).head(8)[['predictor','standardized_mean_difference','ks_statistic','psi']].to_dict('records')
    out['recent_ge45_decomposition']=dec[(dec.comparison_id==recent)&(dec.event_id=='ge_45')][['regime_space','total_event_rate_change','composition_component','within_regime_component','composition_fraction_of_abs_components','within_regime_fraction_of_abs_components']].to_dict('records')
    return out

def patch_outputs(root,manifest,outdir,expected,allow,logger):
    inp,tgt,tsa=root['input'],root['target'],root['timestamps'];n=int(inp.shape[0])
    if int(tgt.shape[0])!=n:raise RuntimeError('input/target patch count mismatch')
    pts=to_utc(np.asarray(tsa[:]));
    if len(pts)!=n:raise RuntimeError('timestamps length mismatch')
    lookup=manifest.set_index('timestamp_utc').split; ps=lookup.reindex(pts).to_numpy()
    if pd.isna(ps).any():raise RuntimeError('há timestamps do Zarr ausentes no manifest da Fase 6')
    ps=np.asarray(ps,object); codes=np.array([CODES[str(x)] for x in ps],dtype=np.uint8);np.save(outdir/'patch_split_codes.npy',codes)
    rows=[];allidx=[]
    for sp in SPLITS:
        idx=np.flatnonzero(ps==sp).astype(np.int64);allidx.append(idx);np.save(outdir/('excluded_purge_patch_indices.npy' if sp=='excluded_purge' else f'{sp}_patch_indices.npy'),idx);nu=pd.DatetimeIndex(pts[idx]).nunique()
        rows.append(dict(split=sp,split_code=CODES[sp],n_patches=len(idx),n_unique_timestamps=int(nu),mean_patches_per_timestamp=(len(idx)/nu if nu else np.nan)))
    cat=np.concatenate(allidx)
    if len(cat)!=n or len(np.unique(cat))!=n:raise RuntimeError('partição de patches não é exata')
    counts=pd.Series(pts).value_counts();info={'n_patch_rows':n,'n_unique_patch_timestamps':int(counts.size),'patches_per_timestamp_min':int(counts.min()),'patches_per_timestamp_median':float(counts.median()),'patches_per_timestamp_max':int(counts.max()),'patches_per_timestamp_mean':float(counts.mean()),'expected_patches_per_timestamp':int(expected),'all_timestamps_match_expected_patch_count':bool((counts==expected).all())}
    if expected>0 and not info['all_timestamps_match_expected_patch_count'] and not allow: raise RuntimeError(f'patches/timestamp diferente de {expected}; use --allow-variable-patches somente se intencional')
    logger.info('Patch partition: %d patches | %d timestamps | patches/timestamp %d..%d',n,counts.size,counts.min(),counts.max())
    return pd.DataFrame(rows),info

def main():
    args=parse_args();setup_output(args.output_dir,args.overwrite);lg=setup_logger(args.output_dir)
    zp=args.dataset_dir/'train.zarr';rp=args.phase6_dir/'timestamp_event_metrics.parquet';pp=args.phase6_dir/'predictor_timestamp_means.parquet';gp=args.phase13_dir/'regime_assignments.parquet'
    for p in [zp,rp,pp,gp]:
        if not p.exists():raise FileNotFoundError(p)
    root=open_zarr(zp);radar=prep_radar(pd.read_parquet(rp),args.local_timezone);pred=normalize_time(pd.read_parquet(pp),args.local_timezone);pred,pnames=canonicalize(pred);reg=normalize_time(pd.read_parquet(gp),args.local_timezone)
    sd=split_def(args);bd=boundaries(sd);ts=pd.DatetimeIndex(radar.timestamp_utc);nom=assign_nominal(ts,sd);final,reason=purge(ts,nom,bd,args.purge_hours)
    if np.any(final=='outside_defined_period'):raise RuntimeError('timestamps fora do período configurado')
    roles=sd.set_index('split').role.to_dict();locks=sd.set_index('split').locked.to_dict();access=sd.set_index('split').model_selection_access.to_dict();refit=sd.set_index('split').final_refit_eligible_pre_primary_test.to_dict()
    manifest=pd.DataFrame(dict(timestamp_utc=ts,year_utc=radar.year_utc.to_numpy(),month_local=radar.month_local.to_numpy(),hour_local=radar.hour_local.to_numpy(),season_code=radar.season_code.astype(str).to_numpy(),nominal_split=nom,split=final,is_purged=(final=='excluded_purge'),purge_boundary=reason)).sort_values('timestamp_utc').reset_index(drop=True)
    manifest['role']=manifest.split.map(lambda x:'excluded temporal purge' if x=='excluded_purge' else roles[x]);manifest['locked']=manifest.split.map(lambda x:True if x=='excluded_purge' else bool(locks[x]));manifest['model_selection_access']=manifest.split.map(lambda x:False if x=='excluded_purge' else bool(access[x]));manifest['final_refit_eligible_pre_primary_test']=manifest.split.map(lambda x:False if x=='excluded_purge' else bool(refit[x]))
    manifest.to_parquet(args.output_dir/'timestamp_split_manifest.parquet',index=False)
    lg.info('Timestamp split counts:\n%s',manifest.split.value_counts().to_string())
    cov=split_coverage(manifest,sd,args.purge_hours);cov.to_parquet(args.output_dir/'split_coverage.parquet',index=False)
    rm=radar.merge(manifest[['timestamp_utc','split','season_code']],on='timestamp_utc',how='inner',validate='one_to_one',suffixes=('', '_split'));rm['season_code']=rm['season_code_split']
    eb=event_balance(rm);eb.to_parquet(args.output_dir/'split_event_balance.parquet',index=False);es=event_season(rm);es.to_parquet(args.output_dir/'split_event_by_season.parquet',index=False)
    rb,rs=regime_tables(manifest,reg);rb.to_parquet(args.output_dir/'split_regime_balance.parquet',index=False);rs.to_parquet(args.output_dir/'split_regime_shift_vs_train.parquet',index=False)
    ps=predictor_shift(pred,manifest,pnames);ps.to_parquet(args.output_dir/'split_predictor_shift_vs_train.parquet',index=False)
    ba=boundary_audit(manifest,bd,args.purge_hours);ba.to_parquet(args.output_dir/'split_boundary_audit.parquet',index=False)
    patchsum,patchinfo=patch_outputs(root,manifest,args.output_dir,args.expected_patches_per_timestamp,args.allow_variable_patches,lg);patchsum.to_parquet(args.output_dir/'split_patch_summary.parquet',index=False)
    codebook={'phase_version':PHASE_VERSION,'codes':{str(v):k for k,v in CODES.items()},'array':'patch_split_codes.npy','dtype':'uint8','semantics':'one code per train.zarr patch row'};(args.output_dir/'split_codebook.json').write_text(json.dumps(codebook,indent=2,ensure_ascii=False),encoding='utf-8')
    rationale=phase14_snapshot(args.phase14_dir);(args.output_dir/'phase14_rationale_snapshot.json').write_text(json.dumps(rationale,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    ss=[]
    for sp in SPLITS:
        t=manifest[manifest.split==sp];ss.append(dict(split=sp,split_code=CODES[sp],n_timestamps=len(t),first_timestamp_utc=(t.timestamp_utc.min() if len(t) else pd.NaT),last_timestamp_utc=(t.timestamp_utc.max() if len(t) else pd.NaT),n_years_present=int(t.year_utc.nunique()),model_selection_access=(False if sp=='excluded_purge' else bool(access[sp])),locked=(True if sp=='excluded_purge' else bool(locks[sp])),final_refit_eligible_pre_primary_test=(False if sp=='excluded_purge' else bool(refit[sp]))))
    ss=pd.DataFrame(ss).merge(patchsum[['split','n_patches','n_unique_timestamps','mean_patches_per_timestamp']],on='split',how='left');ss.to_parquet(args.output_dir/'split_summary.parquet',index=False)
    cfg={'phase_version':PHASE_VERSION,'dataset_dir':str(args.dataset_dir),'split_definition':[{k:(str(v) if isinstance(v,pd.Timestamp) else v) for k,v in r.items()} for r in sd.to_dict('records')],'purge_hours_each_side_of_boundary':args.purge_hours,'local_timezone':args.local_timezone,'policy':{'train':'fit parameters and data-dependent preprocessing','validation':'hyperparameter/model selection only','test_primary':'locked 2023 chronological test; no model selection','test_stress_ood':'locked 2024 stress/OOD test; report coverage explicitly','final_refit_before_primary_test':'after freezing design, train+validation (<=2022) may be combined for one final refit evaluated once on 2023','stress_test_policy':'2024 remains locked and separate','event_balance_policy':'do not rebalance timestamps across temporal splits'}};(args.output_dir/'split_config.json').write_text(json.dumps(cfg,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    integrity={'unique_timestamp_rows':int(manifest.timestamp_utc.nunique()),'manifest_rows':len(manifest),'timestamps_unique':bool(manifest.timestamp_utc.is_unique),'all_timestamps_assigned':bool(manifest.split.isin(SPLITS).all()),'boundary_audit_all_pass_temporal_order':bool(ba.passes_temporal_order.all()),'boundary_audit_all_pass_requested_radius_gap':bool(ba.passes_at_least_requested_radius_gap.all()),'patch_count_summary':patchinfo}
    warnings=['Temporal split is intentionally not stratified by target; natural event-rate differences are part of the generalization test.','Symmetric purge removes available timestamps on both sides of each boundary.','2024 is a stress/OOD test because Phase 14 identified severe radar coverage loss and material covariate shift.','Regime/predictor diagnostics do not reshuffle timestamps across chronological splits.','All 32x32 patches sharing a timestamp inherit the same split.','Rare-event sampling/class weighting must be applied only inside training.','Normalization or any data-dependent transform must be fit on train only, except for an explicitly declared final-refit protocol after design freeze.']
    summary={'phase_version':PHASE_VERSION,'n_available_timestamps':len(manifest),'n_total_patches':int(root['input'].shape[0]),'purge_hours_each_side_of_boundary':args.purge_hours,'split_counts_timestamps':{s:int((manifest.split==s).sum()) for s in SPLITS},'split_counts_patches':{r['split']:int(r['n_patches']) for r in patchsum.to_dict('records')},'integrity':integrity,'phase14_rationale_available':bool(rationale.get('available')),'scope':'formal chronological split protocol for CorrDiff training/model-selection/primary-test/stress-OOD','warnings':warnings,'methodological_notes':['train=2011-2020; validation=2021-2022; test_primary=2023; test_stress_ood=2024.','test_primary and test_stress_ood are locked.','After design freeze, an explicit final-refit may combine train+validation through 2022 and evaluate once on 2023.','2024 remains separate after final refit.','Split balance diagnostics are descriptive and never override chronological ordering.']}
    (args.output_dir/'analysis_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    lg.info('PHASE 15 COMPLETE | output=%s',args.output_dir)

if __name__=='__main__': main()
