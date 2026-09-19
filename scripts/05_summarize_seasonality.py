#!/usr/bin/env python3
"""Resumo textual da Fase 5 - Sazonalidade."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--phase5-dir",
        type=Path,
        default=Path("analysis_outputs/05_seasonality"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_outputs/summary_phase5.txt"),
    )
    return p.parse_args()


def pct(x):
    return "NA" if pd.isna(x) else f"{100*x:.2f}%"


def main():
    args = parse_args()
    d = args.phase5_dir

    summary = json.loads((d / "analysis_summary.json").read_text())
    mcov = pd.read_parquet(d / "monthly_coverage.parquet")
    scov = pd.read_parquet(d / "season_year_coverage.parquet")
    me = pd.read_parquet(d / "monthly_event_rates.parquet")
    se = pd.read_parquet(d / "seasonal_event_rates.parquet")

    lines = []
    add = lines.append

    add("=" * 88)
    add("CORRDIFF - FASE 5 - SAZONALIDADE CONDICIONADA À DISPONIBILIDADE")
    add("=" * 88)
    add(f"Versão: {summary.get('phase_version')}")
    add(
        f"Cobertura temporal global: "
        f"{pct(summary.get('temporal_coverage_ratio'))}"
    )
    add(
        f"Timestamps disponíveis / esperados: "
        f"{summary.get('available_unique_timestamps')} / "
        f"{summary.get('expected_timestamps')}"
    )
    add("")

    add("COBERTURA CLIMATOLÓGICA POR MÊS")
    add("-" * 88)
    add(
        mcov[[
            "month", "expected_timestamps", "available_timestamps",
            "missing_timestamps", "coverage_ratio"
        ]].to_string(index=False)
    )
    add("")

    for event_id in ["gt_0", "ge_20", "ge_30", "ge_40", "ge_45"]:
        add(f"EVENTO {event_id} - POR MÊS")
        add("-" * 88)
        t = me[me["event_id"] == event_id][[
            "month", "coverage_ratio", "pixel_event_rate",
            "timestamp_any_event_rate", "patch_any_event_rate",
            "mean_positive_radar", "mean_timestamp_max_radar",
        ]].copy()
        add(t.to_string(index=False))
        add("")

    add("EVENTOS POR ESTAÇÃO - SOMENTE SEASON_YEARS COMPLETOS")
    add("-" * 88)
    t = se[se["event_id"].isin(["gt_0", "ge_30", "ge_40", "ge_45"])][[
        "season_code", "season", "event_id", "coverage_ratio",
        "pixel_event_rate", "timestamp_any_event_rate",
        "patch_any_event_rate", "mean_positive_radar",
    ]]
    add(t.to_string(index=False))
    add("")

    pred_path = d / "monthly_predictor_statistics.parquet"
    if pred_path.exists():
        pred = pd.read_parquet(pred_path)
        add("SAZONALIDADE DOS PREDITORES-CHAVE")
        add("-" * 88)
        key = [
            "tcwv", "r_500", "t_850", "t_500", "v_500", "r_850",
            "delta_r_500_850", "delta_t_500_850",
            "delta_t_850_surface", "wind_speed_10",
        ]
        p = pred[pred["predictor"].isin(key)][[
            "month", "predictor", "mean", "std", "p10", "median", "p90"
        ]]
        add(p.to_string(index=False))
        add("")

    add("AVISOS / NOTAS")
    add("-" * 88)
    for item in summary.get("warnings", []):
        add(f"- WARNING: {item}")
    for item in summary.get("methodological_notes", []):
        add(f"- NOTE: {item}")
    add("")
    add("=" * 88)
    add("FIM DA FASE 5")
    add("=" * 88)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
