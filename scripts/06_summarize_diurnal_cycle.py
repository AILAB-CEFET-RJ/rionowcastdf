#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

EVENTS=["gt_0","ge_20","ge_30","ge_40","ge_45"]

def main():
    p=argparse.ArgumentParser(); p.add_argument("--phase6-dir",type=Path,default=Path("analysis_outputs/06_diurnal")); p.add_argument("--output",type=Path,default=Path("analysis_outputs/summary_phase6.txt")); a=p.parse_args(); d=a.phase6_dir
    s=json.loads((d/"analysis_summary.json").read_text()); cu=pd.read_parquet(d/"hourly_coverage_utc.parquet"); cl=pd.read_parquet(d/"hourly_coverage_local.parquet"); el=pd.read_parquet(d/"hourly_event_rates_local.parquet"); std=pd.read_parquet(d/"standardized_hourly_event_rates_local.parquet"); sea=pd.read_parquet(d/"season_hour_event_rates_local.parquet")
    L=[]; add=L.append; add("="*100); add("CORRDIFF - FASE 6 - CICLO DIURNO CONDICIONADO À DISPONIBILIDADE"); add("="*100); add(f"Versão: {s['phase_version']}"); add(f"Timezone local: {s['local_timezone']}"); add(f"Cobertura global: {100*s['temporal_coverage_ratio']:.2f}%"); add(f"Offsets UTC observados: {s['observed_utc_offset_hours_counts']}"); add("")
    add("COBERTURA POR HORA UTC"); add("-"*100); add(cu.to_string(index=False)); add(""); add("COBERTURA POR HORA LOCAL"); add("-"*100); add(cl.to_string(index=False)); add("")
    for eid in EVENTS:
        add(f"EVENTO {eid} - HORA LOCAL"); add("-"*100); t=el[el.event_id.eq(eid)][["hour_local","coverage_ratio","pixel_event_rate","timestamp_any_event_rate","patch_any_event_rate","mean_event_pixel_fraction_given_event_timestamp","mean_positive_dbz","mean_timestamp_max_dbz"]]; add(t.to_string(index=False)); add("")
    add("PERFIL LOCAL PADRONIZADO POR ESTAÇÃO"); add("-"*100); add(std[std.event_id.isin(EVENTS)][["event_id","hour_local","n_seasons","pixel_event_rate","timestamp_any_event_rate","patch_any_event_rate","mean_event_pixel_fraction_given_event_timestamp","mean_positive_dbz"]].to_string(index=False)); add("")
    add("CICLO POR ESTAÇÃO"); add("-"*100); add(sea[sea.event_id.isin(["gt_0","ge_30","ge_40","ge_45"])][["season_code","season","hour_local","event_id","coverage_ratio","timestamp_any_event_rate","pixel_event_rate","mean_event_pixel_fraction_given_event_timestamp"]].to_string(index=False)); add("")
    pred=d/"hourly_predictor_statistics_local.parquet"
    if pred.exists():
        x=pd.read_parquet(pred); key=["tcwv","r_500","t_850","t_500","v_500","r_850","wind_speed_10","delta_r_500_850","delta_t_500_850","delta_t_850_surface"]; add("PREDITORES-CHAVE - HORA LOCAL"); add("-"*100); add(x[x.predictor.isin(key)][["hour_local","predictor","mean","std","p10","median","p90"]].to_string(index=False)); add("")
    add("PICOS/MÍNIMOS - PERFIL LOCAL BRUTO"); add("-"*100); [add(f"- {x}") for x in s.get("raw_local_peak_trough_timestamp_event",[])]; add(""); add("PICOS/MÍNIMOS - PERFIL PADRONIZADO POR ESTAÇÃO"); add("-"*100); [add(f"- {x}") for x in s.get("season_standardized_local_peak_trough_timestamp_event",[])]; add("")
    add("AVISOS / NOTAS"); add("-"*100); [add(f"- WARNING: {x}") for x in s.get("warnings",[])]; [add(f"- NOTE: {x}") for x in s.get("methodological_notes",[])]; add(""); add("="*100); add("FIM DA FASE 6"); add("="*100)
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text("\n".join(L),encoding="utf-8"); print(a.output)
if __name__=="__main__": main()
