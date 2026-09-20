#!/usr/bin/env python3
"""Resumo textual da Fase 15 - split formal CorrDiff."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--phase15-dir',type=Path,default=Path('analysis_outputs/15_formal_splits'))
    p.add_argument('--output',type=Path,default=Path('analysis_outputs/summary_phase15.txt'))
    p.add_argument('--top-k',type=int,default=12)
    return p.parse_args()

def main():
    a=parse_args();d=a.phase15_dir;k=a.top_k
    summary=json.loads((d/'analysis_summary.json').read_text(encoding='utf-8'))
    config=json.loads((d/'split_config.json').read_text(encoding='utf-8'))
    rationale=json.loads((d/'phase14_rationale_snapshot.json').read_text(encoding='utf-8'))
    ss=pd.read_parquet(d/'split_summary.parquet')
    cov=pd.read_parquet(d/'split_coverage.parquet')
    ev=pd.read_parquet(d/'split_event_balance.parquet')
    es=pd.read_parquet(d/'split_event_by_season.parquet')
    rb=pd.read_parquet(d/'split_regime_balance.parquet')
    rs=pd.read_parquet(d/'split_regime_shift_vs_train.parquet')
    ps=pd.read_parquet(d/'split_predictor_shift_vs_train.parquet')
    ba=pd.read_parquet(d/'split_boundary_audit.parquet')
    patch=pd.read_parquet(d/'split_patch_summary.parquet')
    lines=[];add=lines.append
    add('='*132);add('CORRDIFF - FASE 15 - SPLIT FORMAL DE TREINO / VALIDAÇÃO / TESTE / STRESS-OOD');add('='*132)
    add(f"Versão: {summary.get('phase_version')}")
    add(f"Timestamps disponíveis: {summary.get('n_available_timestamps')}")
    add(f"Patches totais: {summary.get('n_total_patches')}")
    add(f"Purge simétrico: {summary.get('purge_hours_each_side_of_boundary')} h de cada lado")
    add(f"Escopo: {summary.get('scope')}");add('')

    add('PROTOCOLO CONGELADO');add('-'*132)
    for row in config['split_definition']:
        add(f"{row['split']:16s} | {row['start_utc']} -> {row['end_utc']} | locked={row['locked']} | model_selection={row['model_selection_access']} | role={row['role']}")
    add('')

    add('RESUMO DOS SPLITS');add('-'*132);add(ss.to_string(index=False));add('')
    add('COVERAGE POR SPLIT');add('-'*132);add(cov.to_string(index=False));add('')
    add('AUDITORIA DAS FRONTEIRAS / PURGE');add('-'*132);add(ba.to_string(index=False));add('')
    add('PARTIÇÃO DE PATCHES');add('-'*132);add(patch.to_string(index=False));add('')

    add('BALANÇO DE EVENTOS RADAR');add('-'*132)
    cols=[c for c in ['split','n_timestamps','timestamp_event_rate_gt_0','timestamp_event_rate_ge_20','timestamp_event_rate_ge_30','timestamp_event_rate_ge_40','timestamp_event_rate_ge_45','mean_max_dbz','max_dbz_p50','max_dbz_p90','max_dbz_p99','mean_positive_pixel_fraction','mean_event_pixel_fraction_ge_30','mean_event_pixel_fraction_ge_40','mean_event_pixel_fraction_ge_45'] if c in ev.columns]
    add(ev[cols].to_string(index=False));add('')

    add('EVENTOS POR ESTAÇÃO');add('-'*132)
    for sp in ['train','validation','test_primary','test_stress_ood']:
        t=es[es.split==sp]
        if t.empty:continue
        add(f'[{sp}]');add(t.to_string(index=False));add('')

    add('SHIFT DE REGIMES VS TRAIN');add('-'*132);add(rs.to_string(index=False));add('')
    add('PREVALÊNCIA DOS REGIMES POR SPLIT');add('-'*132)
    for space in rb.regime_space.unique():
        add(f'[{space}]')
        t=rb[rb.regime_space==space].pivot(index='regime',columns='split',values='prevalence')
        add(t.to_string());add('')

    add('TOP SHIFTS DE PREDICTORS VS TRAIN');add('-'*132)
    for sp in ['validation','test_primary','test_stress_ood']:
        t=ps[ps.evaluation_split==sp].copy();t['abs_smd']=t.standardized_mean_difference.abs();t=t.sort_values(['abs_smd','ks_statistic'],ascending=False).head(k)
        add(f'[{sp}]');add(t[['predictor','standardized_mean_difference','ks_statistic','reference_mean','evaluation_mean']].to_string(index=False));add('')

    add('SNAPSHOT DA JUSTIFICATIVA DA FASE 14');add('-'*132)
    if rationale.get('available'):
        add(f"Coverage recente: {rationale.get('coverage_ratio')}")
        add('Shift de composição recente:')
        add(pd.DataFrame(rationale.get('recent_regime_shift',[])).to_string(index=False))
        add('')
        add('Top shifts ERA5 em anomalias mês×hora:')
        add(pd.DataFrame(rationale.get('top_recent_anomaly_predictor_shifts',[])).to_string(index=False))
        add('')
        add('Decomposição >=45 dBZ recente:')
        add(pd.DataFrame(rationale.get('recent_ge45_decomposition',[])).to_string(index=False))
    else:
        add('Outputs esperados da Fase 14 não estavam disponíveis.')
    add('')

    add('INTEGRIDADE');add('-'*132)
    for key,val in summary.get('integrity',{}).items():
        add(f'{key}: {val}')
    add('')
    add('POLÍTICA DE USO');add('-'*132)
    for key,val in config.get('policy',{}).items():add(f'- {key}: {val}')
    add('')
    add('AVISOS / NOTAS');add('-'*132)
    for x in summary.get('warnings',[]):add(f'- WARNING: {x}')
    for x in summary.get('methodological_notes',[]):add(f'- NOTE: {x}')
    add('');add('='*132);add('FIM DA FASE 15');add('='*132)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text('\n'.join(lines),encoding='utf-8');print(a.output)

if __name__=='__main__':main()
