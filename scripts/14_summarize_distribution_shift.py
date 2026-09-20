#!/usr/bin/env python3
"""Resumo textual da Fase 14 - distribution shift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


RECENT = "RECENT_2023_2024_vs_pre2023"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--phase14-dir",
        type=Path,
        default=Path("analysis_outputs/14_distribution_shift"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_outputs/summary_phase14.txt"),
    )
    p.add_argument("--top-k", type=int, default=12)
    return p.parse_args()


def top_shift(
    df: pd.DataFrame,
    comparison_id: str,
    k: int,
) -> pd.DataFrame:
    t = df[df["comparison_id"] == comparison_id].copy()
    t["abs_smd"] = t["standardized_mean_difference"].abs()
    return (
        t.sort_values(
            ["abs_smd", "ks_statistic", "psi"],
            ascending=[False, False, False],
        )
        .head(k)
    )


def main():
    args = parse_args()
    d = args.phase14_dir
    k = args.top_k

    summary = json.loads(
        (d / "analysis_summary.json").read_text(encoding="utf-8")
    )

    windows = pd.read_parquet(d / "comparison_windows.parquet")
    cov_y = pd.read_parquet(d / "coverage_by_year.parquet")
    cov_m = pd.read_parquet(d / "coverage_by_month.parquet")
    gaps = pd.read_parquet(d / "missing_gap_runs.parquet")
    pred = pd.read_parquet(d / "predictor_shift_metrics.parquet")
    anom = pd.read_parquet(d / "predictor_shift_anomaly_metrics.parquet")
    reg_shift = pd.read_parquet(d / "regime_composition_shift.parquet")
    reg_delta = pd.read_parquet(d / "regime_prevalence_delta.parquet")
    radar_event = pd.read_parquet(d / "radar_event_shift.parquet")
    radar_cont = pd.read_parquet(d / "radar_continuous_shift.parquet")
    conditional = pd.read_parquet(
        d / "conditional_event_shift_by_regime.parquet"
    )
    decomposition = pd.read_parquet(
        d / "event_rate_decomposition.parquet"
    )

    lines = []
    add = lines.append

    add("=" * 132)
    add("CORRDIFF - FASE 14 - DISTRIBUTION SHIFT E ESTABILIDADE TEMPORAL")
    add("=" * 132)
    add(f"Versão: {summary.get('phase_version')}")
    add(f"Timestamps radar: {summary.get('n_radar_timestamps')}")
    add(f"Timestamps predictors: {summary.get('n_predictor_timestamps')}")
    add(f"Bootstrap: {summary.get('bootstrap_reps')} reps em blocos {summary.get('bootstrap_block')}")
    add(f"Escopo: {summary.get('scope')}")
    add("")

    add("JANELAS DE COMPARAÇÃO")
    add("-" * 132)
    add(windows.to_string(index=False))
    add("")

    add("COVERAGE RADAR POR ANO")
    add("-" * 132)
    add(
        cov_y[
            [
                "year_utc",
                "expected_hours",
                "available_hours",
                "missing_hours",
                "availability_ratio",
            ]
        ].to_string(index=False)
    )
    add("")

    add("COVERAGE RADAR - MESES DE MENOR DISPONIBILIDADE")
    add("-" * 132)
    add(
        cov_m.sort_values(
            ["availability_ratio", "year_utc", "month_utc"]
        ).head(24)[
            [
                "year_utc",
                "month_utc",
                "expected_hours",
                "available_hours",
                "availability_ratio",
            ]
        ].to_string(index=False)
    )
    add("")

    add("MAIORES GAPS DE RADAR")
    add("-" * 132)
    if gaps.empty:
        add("Nenhum gap encontrado.")
    else:
        add(gaps.head(20).to_string(index=False))
    add("")

    add("SHIFT DE PREDICTORS - COMPARAÇÃO F5 / 2023-2024")
    add("-" * 132)
    add("[absolute]")
    t = top_shift(pred, RECENT, k)
    add(
        t[
            [
                "predictor",
                "reference_mean",
                "evaluation_mean",
                "standardized_mean_difference",
                "std_ratio_eval_over_reference",
                "ks_statistic",
                "psi",
                "p10_shift_in_reference_sigma",
                "p50_shift_in_reference_sigma",
                "p90_shift_in_reference_sigma",
            ]
        ].to_string(index=False)
    )
    add("")
    add("[reference_month_hour_anomaly]")
    t = top_shift(anom, RECENT, k)
    add(
        t[
            [
                "predictor",
                "reference_mean",
                "evaluation_mean",
                "standardized_mean_difference",
                "std_ratio_eval_over_reference",
                "ks_statistic",
                "psi",
                "p10_shift_in_reference_sigma",
                "p50_shift_in_reference_sigma",
                "p90_shift_in_reference_sigma",
            ]
        ].to_string(index=False)
    )
    add("")

    add("TOP SHIFTS ABSOLUTOS POR ANO")
    add("-" * 132)
    for cid in windows["comparison_id"]:
        if cid == RECENT:
            continue
        add(f"[{cid}]")
        t = top_shift(pred, cid, min(k, 8))
        add(
            t[
                [
                    "predictor",
                    "standardized_mean_difference",
                    "ks_statistic",
                    "psi",
                ]
            ].to_string(index=False)
        )
        add("")

    add("SHIFT DE COMPOSIÇÃO DOS REGIMES")
    add("-" * 132)
    add(
        reg_shift.sort_values(
            ["comparison_id", "regime_space"]
        )[
            [
                "comparison_id",
                "regime_space",
                "n_reference",
                "n_evaluation",
                "jensen_shannon_divergence",
                "total_variation_distance",
                "max_absolute_prevalence_delta",
            ]
        ].to_string(index=False)
    )
    add("")

    add("DELTA DE PREVALÊNCIA DOS REGIMES - 2023/2024 vs <=2022")
    add("-" * 132)
    t = reg_delta[reg_delta["comparison_id"] == RECENT].copy()
    add(
        t.sort_values(
            ["regime_space", "prevalence_delta"],
            ascending=[True, False],
        )[
            [
                "regime_space",
                "regime",
                "reference_prevalence",
                "evaluation_prevalence",
                "prevalence_delta",
                "prevalence_ratio_eval_over_ref",
            ]
        ].to_string(index=False)
    )
    add("")

    add("SHIFT MARGINAL DO RADAR - EVENTOS")
    add("-" * 132)
    add(
        radar_event.sort_values(
            ["comparison_id", "event_id"]
        )[
            [
                "comparison_id",
                "event_id",
                "reference_event_rate",
                "evaluation_event_rate",
                "event_rate_difference",
                "event_rate_difference_ci95_low",
                "event_rate_difference_ci95_high",
                "risk_ratio_eval_over_reference",
                "risk_ratio_ci95_low",
                "risk_ratio_ci95_high",
            ]
        ].to_string(index=False)
    )
    add("")

    add("SHIFT MARGINAL DO RADAR - MÉTRICAS CONTÍNUAS")
    add("-" * 132)
    add(
        radar_cont.sort_values(
            ["comparison_id", "metric"]
        )[
            [
                "comparison_id",
                "metric",
                "reference_mean",
                "evaluation_mean",
                "mean_difference",
                "standardized_mean_difference",
                "ks_statistic",
                "p50_shift_in_reference_sigma",
                "p90_shift_in_reference_sigma",
                "p99_shift_in_reference_sigma",
            ]
        ].to_string(index=False)
    )
    add("")

    add("SHIFT CONDICIONAL >=45 dBZ DENTRO DOS REGIMES - F5")
    add("-" * 132)
    t = conditional[
        (conditional["comparison_id"] == RECENT)
        & (conditional["event_id"] == "ge_45")
    ].copy()
    add(
        t.sort_values(
            ["regime_space", "regime_id"]
        )[
            [
                "regime_space",
                "regime",
                "n_reference",
                "n_evaluation",
                "reference_event_rate",
                "evaluation_event_rate",
                "event_rate_difference",
                "event_rate_difference_ci95_low",
                "event_rate_difference_ci95_high",
                "risk_ratio_eval_over_reference",
                "risk_ratio_ci95_low",
                "risk_ratio_ci95_high",
            ]
        ].to_string(index=False)
    )
    add("")

    add("SHIFT CONDICIONAL >=45 dBZ POR ANO")
    add("-" * 132)
    t = conditional[
        (conditional["event_id"] == "ge_45")
        & (conditional["comparison_id"] != RECENT)
    ].copy()
    add(
        t.sort_values(
            ["comparison_id", "regime_space", "regime_id"]
        )[
            [
                "comparison_id",
                "regime_space",
                "regime",
                "reference_event_rate",
                "evaluation_event_rate",
                "event_rate_difference",
                "event_rate_difference_ci95_low",
                "event_rate_difference_ci95_high",
            ]
        ].to_string(index=False)
    )
    add("")

    add("DECOMPOSIÇÃO DA MUDANÇA DA TAXA DE EVENTOS")
    add("-" * 132)
    add(
        decomposition.sort_values(
            ["comparison_id", "regime_space", "event_id"]
        )[
            [
                "comparison_id",
                "regime_space",
                "event_id",
                "reference_event_rate",
                "evaluation_event_rate",
                "total_event_rate_change",
                "composition_component",
                "within_regime_component",
                "decomposition_residual",
                "composition_fraction_of_abs_components",
                "within_regime_fraction_of_abs_components",
            ]
        ].to_string(index=False)
    )
    add("")

    add("DECOMPOSIÇÃO F5 - FOCO >=45 dBZ")
    add("-" * 132)
    t = decomposition[
        (decomposition["comparison_id"] == RECENT)
        & (decomposition["event_id"] == "ge_45")
    ]
    add(t.to_string(index=False))
    add("")

    add("AVISOS / NOTAS")
    add("-" * 132)
    for item in summary.get("warnings", []):
        add(f"- WARNING: {item}")
    for item in summary.get("methodological_notes", []):
        add(f"- NOTE: {item}")

    add("")
    add("=" * 132)
    add("FIM DA FASE 14")
    add("=" * 132)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
