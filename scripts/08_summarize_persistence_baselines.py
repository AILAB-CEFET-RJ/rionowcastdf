#!/usr/bin/env python3
"""Resumo textual da Fase 8 - Baselines temporais de persistência."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EVENTS = ["gt_0", "ge_20", "ge_30", "ge_40", "ge_45"]
BASELINE_ORDER = [
    "train_prevalence",
    "train_mean",
    "climatology_month_hour",
    "persistence_1h",
    "persistence_mean_lags",
    "persistence_exp_decay",
    "logistic_temporal",
    "ridge_temporal",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--phase8-dir",
        type=Path,
        default=Path("analysis_outputs/08_persistence_baselines"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_outputs/summary_phase8.txt"),
    )
    return p.parse_args()


def fmt_float(x, digits=6):
    return "NA" if pd.isna(x) else f"{x:.{digits}f}"


def main():
    args = parse_args()
    d = args.phase8_dir

    summary = json.loads(
        (d / "analysis_summary.json").read_text(encoding="utf-8")
    )
    cohort = pd.read_parquet(
        d / "evaluation_cohort_summary.parquet"
    )
    binary = pd.read_parquet(
        d / "binary_baseline_metrics.parquet"
    )
    binary_season = pd.read_parquet(
        d / "binary_baseline_metrics_by_season.parquet"
    )
    continuous = pd.read_parquet(
        d / "continuous_baseline_metrics.parquet"
    )
    condition = pd.read_parquet(
        d / "continuous_baseline_metrics_by_condition.parquet"
    )
    status = pd.read_parquet(
        d / "model_status.parquet"
    )

    lines = []
    add = lines.append

    add("=" * 112)
    add("CORRDIFF - FASE 8 - BASELINES TEMPORAIS DE PERSISTÊNCIA")
    add("=" * 112)
    add(f"Versão: {summary.get('phase_version')}")
    add(f"Unidade radar: {summary.get('radar_unit')}")
    add(f"Lags: {summary.get('lags_hours')}")
    add(
        "Split provisório: "
        f"{summary.get('provisional_split')}"
    )
    add(
        "Retenção do cohort comum: "
        f"{100*summary.get('common_lag_cohort_retention', float('nan')):.2f}%"
    )
    add("")

    add("COHORT DE AVALIAÇÃO")
    add("-" * 112)
    add(cohort.to_string(index=False))
    add("")

    add("MODELOS / STATUS")
    add("-" * 112)
    add(status.to_string(index=False))
    add("")

    add("MÉTRICAS BINÁRIAS - TESTE")
    add("-" * 112)
    btest = binary[binary["split"] == "test"].copy()
    btest["baseline"] = pd.Categorical(
        btest["baseline"],
        categories=BASELINE_ORDER,
        ordered=True,
    )
    btest = btest.sort_values(["event_id", "baseline"])
    cols = [
        "event_id",
        "threshold_dbz",
        "baseline",
        "n",
        "event_rate",
        "brier",
        "brier_skill_vs_climatology",
        "roc_auc",
        "average_precision",
        "ece_10bins",
    ]
    add(btest[cols].to_string(index=False))
    add("")

    add("SKILL BINÁRIO POR ESTAÇÃO - TESTE")
    add("-" * 112)
    bst = binary_season[
        (binary_season["split"] == "test")
        & binary_season["event_id"].isin(["ge_30", "ge_40", "ge_45"])
        & binary_season["baseline"].isin(
            [
                "climatology_month_hour",
                "persistence_1h",
                "persistence_exp_decay",
                "logistic_temporal",
            ]
        )
    ][[
        "season_code",
        "event_id",
        "baseline",
        "n",
        "event_rate",
        "brier",
        "roc_auc",
        "average_precision",
        "ece_10bins",
    ]]
    add(bst.to_string(index=False))
    add("")

    add("MÉTRICAS CONTÍNUAS - TESTE")
    add("-" * 112)
    ctest = continuous[continuous["split"] == "test"].copy()
    ctest["baseline"] = pd.Categorical(
        ctest["baseline"],
        categories=BASELINE_ORDER,
        ordered=True,
    )
    ctest = ctest.sort_values(["metric", "baseline"])
    cols = [
        "metric",
        "baseline",
        "n",
        "mae",
        "rmse",
        "bias",
        "pearson_r",
        "spearman_rho",
        "mae_skill_vs_climatology",
        "rmse_skill_vs_climatology",
    ]
    add(ctest[cols].to_string(index=False))
    add("")

    add("MÉTRICAS CONTÍNUAS CONDICIONADAS - TESTE")
    add("-" * 112)
    cond = condition[
        (condition["split"] == "test")
        & condition["condition"].isin(["all", "ge_30", "ge_40", "ge_45"])
        & condition["baseline"].isin(
            [
                "climatology_month_hour",
                "persistence_1h",
                "persistence_exp_decay",
                "ridge_temporal",
            ]
        )
    ][[
        "metric",
        "condition",
        "baseline",
        "n",
        "mae",
        "rmse",
        "mae_skill_vs_climatology",
        "rmse_skill_vs_climatology",
    ]]
    add(cond.to_string(index=False))
    add("")

    add("MENOR BRIER NO TESTE POR EVENTO")
    add("-" * 112)
    for row in summary.get("best_test_binary_by_brier", []):
        add(f"- {row}")
    add("")

    add("MENOR RMSE NO TESTE POR MÉTRICA")
    add("-" * 112)
    for row in summary.get("best_test_continuous_by_rmse", []):
        add(f"- {row}")
    add("")

    add("AVISOS / NOTAS")
    add("-" * 112)
    for item in summary.get("warnings", []):
        add(f"- WARNING: {item}")
    for item in summary.get("methodological_notes", []):
        add(f"- NOTE: {item}")

    add("")
    add("=" * 112)
    add("FIM DA FASE 8")
    add("=" * 112)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
