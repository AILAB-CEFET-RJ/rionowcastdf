#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, logging, math, shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd

PHASE_VERSION = 'phase7-temporal-lags-v1-season-hour-adjusted'
RAW = ['tcwv','t2m','u10','v10','t_850','r_850','u_850','v_850','t_500','r_500','u_500','v_500']
DERIVED = ['wind_speed_10','wind_speed_850','wind_speed_500','delta_t_500_850','delta_t_850_surface','delta_r_500_850','bulk_wind_diff_10_850','bulk_wind_diff_850_500']
PREDICTORS = RAW + DERIVED
EVENTS = {
    'gt_0': ('positive_pixels', 0.0),
    'ge_20': ('event_pixels_ge_20', 20.0),
    'ge_30': ('event_pixels_ge_30', 30.0),
    'ge_40': ('event_pixels_ge_40', 40.0),
    'ge_45': ('event_pixels_ge_45', 45.0),
}
RADAR_METRICS = ['positive_pixel_fraction','max_dbz','mean_positive_dbz','event_pixel_fraction_ge_30','event_pixel_fraction_ge_40','event_pixel_fraction_ge_45']
SEASONS = ['DJF','MAM','JJA','SON']


def args_parser():
    p = argparse.ArgumentParser(description='CorrDiff Fase 7: dependência temporal e análise de lags.')
    p.add_argument('--phase6-dir', type=Path, default=Path('analysis_outputs/06_diurnal'))
    p.add_argument('--output-dir', type=Path, default=Path('analysis_outputs/07_lags'))
    p.add_argument('--lags', default='0,1,2,3,6,12,24')
    p.add_argument('--min-season-pairs', type=int, default=100)
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()


def setup_output(path: Path, overwrite: bool):
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f'{path} is not empty; use --overwrite')
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def setup_logger(path: Path):
    logger = logging.getLogger('corrdiff.phase7')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', '%Y-%m-%d %H:%M:%S')
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    fh = logging.FileHandler(path/'phase7.log', encoding='utf-8'); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger


def parse_lags(text: str):
    vals = sorted(set(int(x.strip()) for x in text.split(',') if x.strip()))
    if not vals or any(v < 0 for v in vals):
        raise ValueError('lags must be non-negative integers')
    return vals


def safe_corr(x, y, method='pearson'):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y); n = int(m.sum())
    if n < 3: return np.nan, n
    xs, ys = pd.Series(x[m]), pd.Series(y[m])
    if xs.nunique() < 2 or ys.nunique() < 2: return np.nan, n
    return float(xs.corr(ys, method=method)), n


def phi(a, b):
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    n11 = float(np.sum(a & b)); n10 = float(np.sum(a & ~b)); n01 = float(np.sum(~a & b)); n00 = float(np.sum(~a & ~b))
    den = math.sqrt((n11+n10)*(n01+n00)*(n11+n01)*(n10+n00))
    return (n11*n00 - n10*n01)/den if den else np.nan


def load_base(phase6: Path):
    ep = phase6/'timestamp_event_metrics.parquet'
    pp = phase6/'predictor_timestamp_means.parquet'
    sp = phase6/'analysis_summary.json'
    for p in [ep, pp, sp]:
        if not p.exists(): raise FileNotFoundError(f'Required Phase 6 output missing: {p}')
    e = pd.read_parquet(ep); p = pd.read_parquet(pp)
    s = json.loads(sp.read_text(encoding='utf-8'))
    needed_e = ['timestamp_utc','month_local','hour_local','season_code','positive_pixels','positive_pixel_fraction','max_dbz','mean_positive_dbz','event_pixels_ge_20','event_pixels_ge_30','event_pixels_ge_40','event_pixels_ge_45','event_pixel_fraction_ge_30','event_pixel_fraction_ge_40','event_pixel_fraction_ge_45']
    missing = [c for c in needed_e if c not in e.columns]
    if missing: raise RuntimeError(f'Phase 6 event metrics missing: {missing}')
    missing = [c for c in ['timestamp_utc']+PREDICTORS if c not in p.columns]
    if missing: raise RuntimeError(f'Phase 6 predictor metrics missing: {missing}')
    e = e.copy(); p = p.copy()
    e['timestamp_utc'] = pd.to_datetime(e['timestamp_utc'], utc=True)
    p['timestamp_utc'] = pd.to_datetime(p['timestamp_utc'], utc=True)
    base = e.merge(p[['timestamp_utc']+PREDICTORS], on='timestamp_utc', how='inner', validate='one_to_one').sort_values('timestamp_utc').reset_index(drop=True)
    for event_id,(count_col,_) in EVENTS.items():
        base[f'event_{event_id}'] = (base[count_col].to_numpy() > 0).astype(np.int8)
    cols = RADAR_METRICS + PREDICTORS + [f'event_{x}' for x in EVENTS]
    for c in cols:
        clim = base.groupby(['month_local','hour_local'], observed=True)[c].transform('mean')
        base[c+'__anom'] = base[c] - clim
    return base, s


def make_pairs(base: pd.DataFrame, lag: int):
    lag_cols = RADAR_METRICS + [m+'__anom' for m in RADAR_METRICS] + PREDICTORS + [p+'__anom' for p in PREDICTORS] + [f'event_{e}' for e in EVENTS] + [f'event_{e}__anom' for e in EVENTS]
    prev = base[['timestamp_utc']+lag_cols].copy()
    prev['timestamp_utc'] = prev['timestamp_utc'] + pd.to_timedelta(lag, unit='h')
    prev = prev.rename(columns={c:c+'__lag' for c in lag_cols})
    return base.merge(prev, on='timestamp_utc', how='inner', validate='one_to_one')


def event_row(g, lag, event_id, extra=None):
    curr = g[f'event_{event_id}'].to_numpy(bool); prev = g[f'event_{event_id}__lag'].to_numpy(bool)
    pe = float(curr[prev].mean()) if prev.any() else np.nan
    pn = float(curr[~prev].mean()) if (~prev).any() else np.nan
    rr = pe/pn if np.isfinite(pe) and np.isfinite(pn) and pn > 0 else np.nan
    ac, n_an = safe_corr(g[f'event_{event_id}__anom'], g[f'event_{event_id}__anom__lag'])
    r = {
        'lag_hours':lag,'event_id':event_id,'threshold_dbz':EVENTS[event_id][1],'n_pairs':len(g),
        'event_rate_t':float(curr.mean()) if len(g) else np.nan,'event_rate_t_minus_lag':float(prev.mean()) if len(g) else np.nan,
        'joint_event_rate':float((curr & prev).mean()) if len(g) else np.nan,
        'p_event_t_given_event_t_minus_lag':pe,'p_event_t_given_no_event_t_minus_lag':pn,
        'persistence_excess_probability':pe-pn if np.isfinite(pe) and np.isfinite(pn) else np.nan,
        'relative_risk':rr,'phi_coefficient':phi(curr,prev),'event_anomaly_autocorrelation':ac,
        'n_pairs_event_anomaly':n_an,'adjustment':'demeaned within month_local x hour_local'
    }
    if extra: r.update(extra)
    return r


def predictor_event_rows(pairs, lag):
    rows=[]
    for pred in PREDICTORS:
        for event_id in EVENTS:
            rr,nr = safe_corr(pairs[pred+'__lag'], pairs[f'event_{event_id}'])
            ra,na = safe_corr(pairs[pred+'__anom__lag'], pairs[f'event_{event_id}__anom'])
            rows.append({'lag_hours':lag,'predictor':pred,'source':'raw' if pred in RAW else 'derived','event_id':event_id,'threshold_dbz':EVENTS[event_id][1],'n_pairs_raw':nr,'point_biserial_r_raw':rr,'n_pairs_adjusted':na,'corr_month_hour_adjusted':ra,'direction':'predictor(t-lag) associated with event(t)'})
    return rows


def predictor_radar_rows(pairs, lag):
    rows=[]
    targets=['positive_pixel_fraction','max_dbz','event_pixel_fraction_ge_30','event_pixel_fraction_ge_40','event_pixel_fraction_ge_45']
    for pred in PREDICTORS:
        for target in targets:
            pr,npr = safe_corr(pairs[pred+'__lag'], pairs[target], 'pearson')
            sr,nsr = safe_corr(pairs[pred+'__lag'], pairs[target], 'spearman')
            pa,npa = safe_corr(pairs[pred+'__anom__lag'], pairs[target+'__anom'], 'pearson')
            sa,nsa = safe_corr(pairs[pred+'__anom__lag'], pairs[target+'__anom'], 'spearman')
            rows.append({'lag_hours':lag,'predictor':pred,'source':'raw' if pred in RAW else 'derived','target_metric':target,'n_pairs_raw':npr,'pearson_r_raw':pr,'spearman_rho_raw':sr,'n_pairs_adjusted':npa,'pearson_r_month_hour_adjusted':pa,'spearman_rho_month_hour_adjusted':sa,'direction':'predictor(t-lag) associated with radar metric(t)'})
    return rows


def main():
    a = args_parser(); lags = parse_lags(a.lags); setup_output(a.output_dir, a.overwrite); log = setup_logger(a.output_dir)
    base, phase6_summary = load_base(a.phase6_dir)
    n_avail = len(base); n_expected = int(phase6_summary.get('expected_timestamps', n_avail))
    log.info('Available timestamps: %d | Expected: %d | Lags: %s', n_avail, n_expected, lags)

    cov=[]; radar=[]; ev=[]; evs=[]; pe=[]; pr=[]
    for lag in lags:
        pairs = make_pairs(base, lag); n=len(pairs); possible=max(n_expected-lag,0)
        cov.append({'lag_hours':lag,'available_timestamp_pairs':n,'expected_possible_pairs':possible,'pair_availability_ratio':n/possible if possible else np.nan,'pair_retention_vs_available_timestamps':n/n_avail if n_avail else np.nan})
        log.info('Lag %2dh -> %d exact pairs', lag, n)
        pe.extend(predictor_event_rows(pairs, lag)); pr.extend(predictor_radar_rows(pairs, lag))
        if lag == 0: continue
        for metric in RADAR_METRICS:
            rp,n1=safe_corr(pairs[metric],pairs[metric+'__lag'],'pearson'); rs,n2=safe_corr(pairs[metric],pairs[metric+'__lag'],'spearman')
            ap,n3=safe_corr(pairs[metric+'__anom'],pairs[metric+'__anom__lag'],'pearson'); ass,n4=safe_corr(pairs[metric+'__anom'],pairs[metric+'__anom__lag'],'spearman')
            radar.append({'lag_hours':lag,'metric':metric,'n_pairs_raw':n1,'pearson_r_raw':rp,'spearman_rho_raw':rs,'n_pairs_anomaly':n3,'pearson_r_month_hour_adjusted':ap,'spearman_rho_month_hour_adjusted':ass,'adjustment':'demeaned within month_local x hour_local'})
        for event_id in EVENTS:
            ev.append(event_row(pairs,lag,event_id))
        for season in SEASONS:
            g=pairs[pairs['season_code']==season]
            if len(g) < a.min_season_pairs: continue
            for event_id in EVENTS:
                evs.append(event_row(g,lag,event_id,{'season_code':season,'season_definition':'season of current timestamp t'}))

    cov_df=pd.DataFrame(cov); radar_df=pd.DataFrame(radar); ev_df=pd.DataFrame(ev); evs_df=pd.DataFrame(evs); pe_df=pd.DataFrame(pe); pr_df=pd.DataFrame(pr)
    cov_df.to_parquet(a.output_dir/'lag_pair_coverage.parquet',index=False)
    radar_df.to_parquet(a.output_dir/'radar_lag_autocorrelation.parquet',index=False)
    ev_df.to_parquet(a.output_dir/'event_lag_dependence.parquet',index=False)
    evs_df.to_parquet(a.output_dir/'event_lag_dependence_by_season.parquet',index=False)
    pe_df.to_parquet(a.output_dir/'predictor_event_lag_associations.parquet',index=False)
    pr_df.to_parquet(a.output_dir/'predictor_radar_lag_associations.parquet',index=False)

    warnings=[]
    nz=cov_df[cov_df.lag_hours>0]
    if not nz.empty and nz.pair_availability_ratio.min() < 0.5:
        warnings.append('At least one lag has <50% exact-pair availability versus the configured timeline.')
    if phase6_summary.get('warnings'):
        warnings.append('Phase 6 reported coverage warnings; lag analyses inherit non-random availability risk.')

    summary={
        'phase_version':PHASE_VERSION,
        'source_phase6_version':phase6_summary.get('phase_version'),
        'available_timestamps':n_avail,
        'expected_timestamps':n_expected,
        'lags_hours':lags,
        'timezone_local':phase6_summary.get('local_timezone','America/Sao_Paulo'),
        'radar_unit':'dBZ',
        'climatology_adjustment':'demeaning within month_local x hour_local',
        'pairing_rule':'only exact UTC timestamp pairs t and t-lag are retained; missing timestamps are never treated as contiguous',
        'spatial_alignment_assumption':'none; timestamp-level aggregates from Phase 6 only',
        'phase8_boundary':'Phase 7 characterizes lag dependence; formal persistence forecast skill is deferred to Phase 8.',
        'warnings':warnings,
        'methodological_notes':[
            'Raw lag associations may contain seasonal and diurnal structure; adjusted metrics remove month x local-hour means.',
            'Adjusted associations are descriptive residual associations, not causal estimates.',
            'Event persistence is timestamp-level: at least one stored patch pixel crosses the dBZ threshold.',
            'Pixel-fraction metrics still describe the overlapping-patch training distribution, not a de-duplicated full field.',
            'Lag 0 is retained only as a contemporaneous reference for predictor-to-radar associations.'
        ]
    }
    (a.output_dir/'analysis_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    log.info('Phase 7 complete -> %s', a.output_dir)

if __name__ == '__main__':
    main()
