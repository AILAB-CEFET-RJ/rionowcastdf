#!/usr/bin/env python3
"""Resumo textual da Fase 13 - regimes meteorológicos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--phase13-dir",
        type=Path,
        default=Path("analysis_outputs/13_regimes"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_outputs/summary_phase13.txt"),
    )
    p.add_argument("--top-k", type=int, default=8)
    return p.parse_args()


def top_abs_profiles(
    profiles: pd.DataFrame,
    regime_space: str,
    regime_id: int,
    k: int,
) -> pd.DataFrame:
    t = profiles[
        (profiles["regime_space"] == regime_space)
        & (profiles["regime_id"] == regime_id)
    ].copy()
    t["abs_anom"] = t["regime_mean_anomaly_sigma"].abs()
    return (
        t.sort_values("abs_anom", ascending=False)
        .head(k)
        .drop(columns="abs_anom")
    )


def main():
    args = parse_args()
    d = args.phase13_dir
    k = args.top_k

    summary = json.loads(
        (d / "analysis_summary.json").read_text(encoding="utf-8")
    )
    diag = pd.read_parquet(d / "clustering_diagnostics.parquet")
    pca = pd.read_parquet(d / "pca_explained_variance.parquet")
    load = pd.read_parquet(d / "pca_loadings.parquet")
    profiles = pd.read_parquet(d / "regime_predictor_profiles.parquet")
    radar = pd.read_parquet(d / "regime_radar_metrics.parquet")
    enrich = pd.read_parquet(d / "regime_event_enrichment.parquet")
    season = pd.read_parquet(d / "regime_seasonality.parquet")
    yearly = pd.read_parquet(d / "regime_yearly_prevalence.parquet")
    event_year = pd.read_parquet(d / "regime_event_metrics_by_year.parquet")
    transitions = pd.read_parquet(d / "regime_transition_1h.parquet")
    crosswalk = pd.read_parquet(d / "regime_crosswalk.parquet")

    lines = []
    add = lines.append

    add("=" * 128)
    add("CORRDIFF - FASE 13 - REGIMES METEOROLÓGICOS")
    add("=" * 128)
    add(f"Versão: {summary.get('phase_version')}")
    add(f"Timestamps alinhados: {summary.get('n_aligned_timestamps')}")
    add(f"Número de regimes de referência: {summary.get('reference_n_regimes')}")
    add(f"k candidatos: {summary.get('candidate_k')}")
    add(f"Threshold PCA: {summary.get('pca_variance_threshold')}")
    add(f"Componentes PCA por espaço: {summary.get('pca_components_by_space')}")
    add(f"Features de clustering: {summary.get('clustering_features')}")
    add(f"Escopo: {summary.get('scope')}")
    add("")

    add("DIAGNÓSTICOS DE CLUSTERING")
    add("-" * 128)
    add(
        diag.sort_values(["regime_space", "k"])[
            [
                "regime_space",
                "k",
                "inertia",
                "silhouette",
                "calinski_harabasz",
                "davies_bouldin",
                "reference_k",
                "reference_k_stability_ari_mean",
                "reference_k_stability_ari_min",
            ]
        ].to_string(index=False)
    )
    add("")

    add("PCA - VARIÂNCIA EXPLICADA")
    add("-" * 128)
    for space in summary.get("regime_spaces", []):
        add(f"[{space}]")
        t = pca[pca["regime_space"] == space]
        add(
            t[
                [
                    "component",
                    "explained_variance_ratio",
                    "cumulative_explained_variance_ratio",
                ]
            ].to_string(index=False)
        )
        add("")

    add("PCA - LOADINGS DOMINANTES")
    add("-" * 128)
    for space in summary.get("regime_spaces", []):
        add(f"[{space}]")
        pcs = (
            load.loc[load["regime_space"] == space, "component_index"]
            .drop_duplicates()
            .sort_values()
            .head(5)
        )
        for pc_idx in pcs:
            t = load[
                (load["regime_space"] == space)
                & (load["component_index"] == pc_idx)
            ].copy()
            t["abs_loading"] = t["loading"].abs()
            t = t.sort_values("abs_loading", ascending=False).head(k)
            add(f"  PC{int(pc_idx)}")
            add(
                t[["feature", "loading"]].to_string(index=False)
            )
        add("")

    add("PERFIS DOS REGIMES - ANOMALIA EM SIGMAS DO CONJUNTO")
    add("-" * 128)
    for space in summary.get("regime_spaces", []):
        add(f"[{space}]")
        ids = sorted(
            profiles.loc[
                profiles["regime_space"] == space,
                "regime_id",
            ].unique()
        )
        for rid in ids:
            add(f"  R{int(rid)}")
            t = top_abs_profiles(profiles, space, int(rid), k)
            add(
                t[
                    [
                        "predictor",
                        "regime_mean",
                        "global_mean",
                        "regime_mean_anomaly_sigma",
                    ]
                ].to_string(index=False)
            )
        add("")

    add("RADAR POR REGIME")
    add("-" * 128)
    radar_cols = [
        c for c in [
            "regime_space",
            "regime",
            "n_timestamps",
            "prevalence",
            "timestamp_event_rate_gt_0",
            "timestamp_event_rate_ge_20",
            "timestamp_event_rate_ge_30",
            "timestamp_event_rate_ge_40",
            "timestamp_event_rate_ge_45",
            "mean_max_dbz",
            "mean_positive_pixel_fraction",
            "mean_event_pixel_fraction_ge_30",
            "mean_event_pixel_fraction_ge_40",
            "mean_event_pixel_fraction_ge_45",
        ]
        if c in radar.columns
    ]
    add(
        radar.sort_values(
            ["regime_space", "regime_id"]
        )[radar_cols].to_string(index=False)
    )
    add("")

    add("ENRIQUECIMENTO DE EVENTOS POR REGIME")
    add("-" * 128)
    t = enrich[
        enrich["event_id"].isin(["ge_30", "ge_40", "ge_45"])
    ].copy()
    add(
        t.sort_values(
            ["regime_space", "event_id", "risk_ratio_vs_global"],
            ascending=[True, True, False],
        )[
            [
                "regime_space",
                "regime",
                "event_id",
                "global_event_rate",
                "regime_event_rate",
                "risk_ratio_vs_global",
                "absolute_rate_difference",
            ]
        ].to_string(index=False)
    )
    add("")

    add("COMPOSIÇÃO SAZONAL DOS REGIMES")
    add("-" * 128)
    for space in summary.get("regime_spaces", []):
        add(f"[{space}]")
        t = season[season["regime_space"] == space]
        pivot = t.pivot(
            index="season_code",
            columns="regime",
            values="prevalence_regime_within_season",
        )
        add(pivot.to_string())
        add("")

    add("PREVALÊNCIA ANUAL DOS REGIMES - 2021 A 2024")
    add("-" * 128)
    t = yearly[yearly["year_utc"] >= 2021].copy()
    for space in summary.get("regime_spaces", []):
        add(f"[{space}]")
        s = t[t["regime_space"] == space]
        pivot = s.pivot(
            index="year_utc",
            columns="regime",
            values="prevalence_within_group",
        )
        add(pivot.to_string())
        add("")

    add(">=45 dBZ POR REGIME E ANO - 2021 A 2024")
    add("-" * 128)
    t = event_year[event_year["year_utc"] >= 2021].copy()
    for space in summary.get("regime_spaces", []):
        add(f"[{space}]")
        s = t[t["regime_space"] == space]
        pivot = s.pivot(
            index="year_utc",
            columns="regime",
            values="timestamp_event_rate_ge_45",
        )
        add(pivot.to_string())
        add("")

    add("PERSISTÊNCIA DOS REGIMES - TRANSIÇÃO EXATA DE 1 HORA")
    add("-" * 128)
    rows = []
    for space in summary.get("regime_spaces", []):
        t = transitions[transitions["regime_space"] == space]
        for rid in sorted(t["regime_from"].unique()):
            r = t[
                (t["regime_from"] == rid)
                & (t["regime_to"] == rid)
            ]
            rows.append({
                "regime_space": space,
                "regime": f"R{int(rid)}",
                "n_from_pairs": int(
                    t.loc[
                        t["regime_from"] == rid,
                        "n_from_pairs",
                    ].iloc[0]
                ),
                "p_same_regime_after_1h": (
                    float(r["transition_probability"].iloc[0])
                    if not r.empty
                    else 0.0
                ),
            })
    add(pd.DataFrame(rows).to_string(index=False))
    add("")

    add("MATRIZ DE TRANSIÇÃO 1H")
    add("-" * 128)
    for space in summary.get("regime_spaces", []):
        add(f"[{space}]")
        t = transitions[transitions["regime_space"] == space]
        pivot = t.pivot(
            index="from_regime",
            columns="to_regime",
            values="transition_probability",
        ).fillna(0.0)
        add(pivot.to_string())
        add("")

    add("CROSSWALK - REGIMES ABSOLUTOS × ANOMALIA MÊS×HORA")
    add("-" * 128)
    pivot = crosswalk.pivot(
        index="absolute_regime_id",
        columns="anomaly_regime_id",
        values="fraction_within_absolute_regime",
    ).fillna(0.0)
    pivot.index = [f"absolute_R{int(x)}" for x in pivot.index]
    pivot.columns = [f"anomaly_R{int(x)}" for x in pivot.columns]
    add(pivot.to_string())
    add("")

    add("AVISOS / NOTAS")
    add("-" * 128)
    for item in summary.get("warnings", []):
        add(f"- WARNING: {item}")
    for item in summary.get("methodological_notes", []):
        add(f"- NOTE: {item}")

    add("")
    add("=" * 128)
    add("FIM DA FASE 13")
    add("=" * 128)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
