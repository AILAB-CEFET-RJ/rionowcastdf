#!/usr/bin/env python3
from __future__ import annotations

import argparse, json
from pathlib import Path
import pandas as pd


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--phase7-dir',type=Path,default=Path('analysis_outputs/07_lags'))
    p.add_argument('--output',type=Path,default=Path('analysis_outputs/summary_phase7.txt'))
    return p.parse_args()


def main():
    a=parse_args(); d=a.phase7_dir
    s=json.loads((d/'analysis_summary.json').read_text(encoding='utf-8'))
    cov=pd.read_parquet(d/'lag_pair_coverage.parquet')
    rad=pd.read_parquet(d/'radar_lag_autocorrelation.parquet')
    ev=pd.read_parquet(d/'event_lag_dependence.parquet')
    evs=pd.read_parquet(d/'event_lag_dependence_by_season.parquet')
    pe=pd.read_parquet(d/'predictor_event_lag_associations.parquet')
    pr=pd.read_parquet(d/'predictor_radar_lag_associations.parquet')

    L=[]; add=L.append
    add('='*108)
    add('CORRDIFF - FASE 7 - DEPENDÊNCIA TEMPORAL E ANÁLISE DE LAGS')
    add('='*108)
    add(f"Versão: {s.get('phase_version')}")
    add(f"Unidade radar: {s.get('radar_unit')}")
    add(f"Lags: {s.get('lags_hours')}")
    add(f"Ajuste climatológico: {s.get('climatology_adjustment')}")
    add(f"Regra de pareamento: {s.get('pairing_rule')}")
    add('')

    add('COBERTURA DOS PARES TEMPORAIS'); add('-'*108); add(cov.to_string(index=False)); add('')

    add('AUTOCORRELAÇÃO DAS MÉTRICAS DO RADAR'); add('-'*108)
    keep=['positive_pixel_fraction','max_dbz','event_pixel_fraction_ge_30','event_pixel_fraction_ge_40','event_pixel_fraction_ge_45']
    t=rad[rad.metric.isin(keep)][['lag_hours','metric','n_pairs_raw','pearson_r_raw','spearman_rho_raw','pearson_r_month_hour_adjusted','spearman_rho_month_hour_adjusted']]
    add(t.to_string(index=False)); add('')

    add('DEPENDÊNCIA TEMPORAL DOS EVENTOS'); add('-'*108)
    t=ev[['lag_hours','event_id','threshold_dbz','n_pairs','event_rate_t','p_event_t_given_event_t_minus_lag','p_event_t_given_no_event_t_minus_lag','persistence_excess_probability','relative_risk','phi_coefficient','event_anomaly_autocorrelation']]
    add(t.to_string(index=False)); add('')

    add('DEPENDÊNCIA DOS EVENTOS POR ESTAÇÃO - >=30/40/45 dBZ'); add('-'*108)
    t=evs[evs.event_id.isin(['ge_30','ge_40','ge_45'])][['season_code','lag_hours','event_id','n_pairs','p_event_t_given_event_t_minus_lag','p_event_t_given_no_event_t_minus_lag','persistence_excess_probability','relative_risk','event_anomaly_autocorrelation']]
    add(t.to_string(index=False)); add('')

    add('TOP PREDITORES LAGADOS PARA EVENTOS - AJUSTADO MÊS x HORA'); add('-'*108)
    for eid in ['ge_30','ge_40','ge_45']:
        add(f'[{eid}]')
        for lag in sorted(pe.lag_hours.unique()):
            x=pe[(pe.event_id==eid)&(pe.lag_hours==lag)].copy()
            if x.empty: continue
            x['abs_corr']=x.corr_month_hour_adjusted.abs()
            top=x.nlargest(8,'abs_corr')[['lag_hours','predictor','source','point_biserial_r_raw','corr_month_hour_adjusted','n_pairs_adjusted']]
            add(top.to_string(index=False)); add('')

    add('TOP PREDITORES LAGADOS PARA MAX_DBZ - AJUSTADO MÊS x HORA'); add('-'*108)
    for lag in sorted(pr.lag_hours.unique()):
        x=pr[(pr.target_metric=='max_dbz')&(pr.lag_hours==lag)].copy()
        if x.empty: continue
        x['abs_corr']=x.spearman_rho_month_hour_adjusted.abs()
        top=x.nlargest(8,'abs_corr')[['lag_hours','predictor','source','spearman_rho_raw','spearman_rho_month_hour_adjusted','n_pairs_adjusted']]
        add(top.to_string(index=False)); add('')

    add('AVISOS / NOTAS'); add('-'*108)
    for x in s.get('warnings',[]): add(f'- WARNING: {x}')
    for x in s.get('methodological_notes',[]): add(f'- NOTE: {x}')
    add(''); add('='*108); add('FIM DA FASE 7'); add('='*108)

    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text('\n'.join(L),encoding='utf-8')
    print(a.output)

if __name__=='__main__':
    main()
