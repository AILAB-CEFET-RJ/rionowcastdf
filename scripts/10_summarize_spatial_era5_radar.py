#!/usr/bin/env python3
"""Resumo textual da Fase 10 - Relações espaciais ERA5 × Radar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


RADAR_ORDER = [
    "target_log1p",
    "gt_0",
    "ge_20",
    "ge_30",
    "ge_40",
    "ge_45",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--phase10-dir",
        type=Path,
        default=Path("analysis_outputs/10_spatial_era5_radar"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_outputs/summary_phase10.txt"),
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=10,
    )
    return p.parse_args()


def top_abs(
    df: pd.DataFrame,
    value_col: str,
    k: int,
) -> pd.DataFrame:
    t = df[np.isfinite(df[value_col])].copy()
    if t.empty:
        return t
    t["_abs"] = t[value_col].abs()
    return t.nlargest(k, "_abs").drop(columns="_abs")


def main():
    args = parse_args()
    d = args.phase10_dir
    k = args.top_k

    summary = json.loads(
        (d / "analysis_summary.json").read_text(encoding="utf-8")
    )
    strata = pd.read_parquet(d / "patch_strata_counts.parquet")
    channels = pd.read_parquet(d / "input_channel_metadata.parquet")
    pixel = pd.read_parquet(d / "pixel_spatial_associations.parquet")
    pixel_season = pd.read_parquet(
        d / "pixel_spatial_associations_by_season.parquet"
    )
    offsets = pd.read_parquet(
        d / "spatial_cross_offset_associations.parquet"
    )
    contrasts = pd.read_parquet(
        d / "within_patch_event_contrasts.parquet"
    )
    contrasts_season = pd.read_parquet(
        d / "within_patch_event_contrasts_by_season.parquet"
    )
    patch = pd.read_parquet(d / "patch_level_associations.parquet")

    lines = []
    add = lines.append

    add("=" * 120)
    add("CORRDIFF - FASE 10 - RELAÇÕES ESPACIAIS ERA5 × RADAR")
    add("=" * 120)
    add(f"Versão: {summary.get('phase_version')}")
    add(f"Input shape: {summary.get('input_shape')}")
    add(f"Target shape: {summary.get('target_shape')}")
    add(
        f"Patches: {summary.get('total_patches')} | "
        f"amostrados: {summary.get('sampled_patches')} "
        f"({100*summary.get('sample_ratio', float('nan')):.2f}%)"
    )
    add(f"Unidade radar: {summary.get('radar_unit')}")
    add(f"Resolução da grade CorrDiff: {summary.get('radar_resolution_km')} km/pixel")
    add(f"Janelas de suavização: {summary.get('smoothing_windows_pixels')}")
    add(f"Offsets: {summary.get('offset_steps_pixels')}")
    add(f"Fonte dos nomes dos canais: {summary.get('channel_name_source')}")
    add(f"Escopo: {summary.get('scope')}")
    add("")

    add("ESTRATOS DE INTENSIDADE - SCAN EXATO DO TARGET")
    add("-" * 120)
    add(strata.to_string(index=False))
    add("")

    add("PREDITORES")
    add("-" * 120)
    add(channels.to_string(index=False))
    add("")

    add("ASSOCIAÇÃO PIXEL A PIXEL - SEM SUAVIZAÇÃO - RAW")
    add("-" * 120)
    base = pixel[
        (pixel["smoothing_window_pixels"] == 1)
        & (pixel["association_mode"] == "raw")
    ]
    for radar_field in RADAR_ORDER:
        t = top_abs(
            base[base["radar_field"] == radar_field],
            "pearson_r",
            k,
        )
        add(f"[{radar_field}]")
        if t.empty:
            add("sem resultados")
        else:
            add(
                t[
                    [
                        "predictor",
                        "pearson_r",
                        "predictor_mean",
                        "predictor_std",
                        "radar_mean",
                    ]
                ].to_string(index=False)
            )
        add("")

    add("ASSOCIAÇÃO LOCAL - CENTRALIZADA DENTRO DO PATCH - SEM SUAVIZAÇÃO")
    add("-" * 120)
    centered = pixel[
        (pixel["smoothing_window_pixels"] == 1)
        & (pixel["association_mode"] == "within_patch_centered")
    ]
    for radar_field in RADAR_ORDER:
        t = top_abs(
            centered[centered["radar_field"] == radar_field],
            "pearson_r",
            k,
        )
        add(f"[{radar_field}]")
        if t.empty:
            add("sem resultados")
        else:
            add(
                t[
                    ["predictor", "pearson_r"]
                ].to_string(index=False)
            )
        add("")

    add("EFEITO DA ESCALA LOCAL - TOP PREDITORES POR LIMIAR")
    add("-" * 120)
    for radar_field in ["ge_30", "ge_40", "ge_45"]:
        add(f"[{radar_field}]")
        t = pixel[
            (pixel["radar_field"] == radar_field)
            & (pixel["association_mode"] == "within_patch_centered")
        ].copy()
        if t.empty:
            add("sem resultados")
            add("")
            continue
        t = t[np.isfinite(t["pearson_r"])].copy()
        if t.empty:
            add("sem resultados")
            add("")
            continue
        t["_abs"] = t["pearson_r"].abs()
        idx = t.groupby("predictor")["_abs"].idxmax()
        best = t.loc[idx].sort_values("_abs", ascending=False).head(k)
        add(
            best[
                [
                    "predictor",
                    "smoothing_window_pixels",
                    "smoothing_half_width_km",
                    "pearson_r",
                ]
            ].to_string(index=False)
        )
        add("")

    add("CONTRASTE ESPACIAL DENTRO DO PATCH - EVENTO MENOS BACKGROUND")
    add("-" * 120)
    for event_id in ["ge_20", "ge_30", "ge_40", "ge_45"]:
        t = contrasts[contrasts["event_id"] == event_id].copy()
        t = t[np.isfinite(t["mean_delta_in_patch_sigma"])]
        t["_abs"] = t["mean_delta_in_patch_sigma"].abs()
        t = t.sort_values("_abs", ascending=False).head(k)
        add(f"[{event_id}]")
        add(
            t[
                [
                    "predictor",
                    "sample_event_patches",
                    "mean_event_minus_background",
                    "mean_delta_in_patch_sigma",
                    "weighted_fraction_positive_delta",
                ]
            ].to_string(index=False)
        )
        add("")

    add("CONTRASTE ESPACIAL POR ESTAÇÃO - >=45 dBZ")
    add("-" * 120)
    t45 = contrasts_season[
        contrasts_season["event_id"] == "ge_45"
    ].copy()
    for season in ["DJF", "MAM", "JJA", "SON"]:
        t = t45[t45["season_code"] == season].copy()
        t = t[np.isfinite(t["mean_delta_in_patch_sigma"])]
        t["_abs"] = t["mean_delta_in_patch_sigma"].abs()
        t = t.sort_values("_abs", ascending=False).head(k)
        add(f"[{season}]")
        add(
            t[
                [
                    "predictor",
                    "sample_event_patches",
                    "mean_delta_in_patch_sigma",
                    "weighted_fraction_positive_delta",
                ]
            ].to_string(index=False)
        )
        add("")

    add("CROSS-OFFSET - MELHOR DESLOCAMENTO POR PREDITOR")
    add("-" * 120)
    for radar_field in ["target_log1p", "ge_30", "ge_40", "ge_45"]:
        t = offsets[
            (offsets["radar_field"] == radar_field)
            & (
                offsets["association_mode"]
                == "within_patch_centered"
            )
        ].copy()
        t = t[np.isfinite(t["pearson_r"])].copy()
        add(f"[{radar_field}]")
        if t.empty:
            add("sem resultados")
            add("")
            continue
        t["_abs"] = t["pearson_r"].abs()
        idx = t.groupby("predictor")["_abs"].idxmax()
        best = t.loc[idx].sort_values("_abs", ascending=False).head(k)
        add(
            best[
                [
                    "predictor",
                    "offset_direction",
                    "offset_step_pixels",
                    "offset_distance_km",
                    "pearson_r",
                ]
            ].to_string(index=False)
        )
        add("")

    add("ASSOCIAÇÃO EM NÍVEL DE PATCH")
    add("-" * 120)
    for radar_metric in [
        "max_dbz",
        "positive_pixel_fraction",
        "event_pixel_fraction_ge_30",
        "event_pixel_fraction_ge_40",
        "event_pixel_fraction_ge_45",
    ]:
        t = top_abs(
            patch[patch["radar_field"] == radar_metric],
            "pearson_r",
            k,
        )
        add(f"[{radar_metric}]")
        add(
            t[
                ["predictor", "pearson_r"]
            ].to_string(index=False)
        )
        add("")

    add("TOP ASSOCIAÇÕES LOCAIS CENTRALIZADAS POR ESTAÇÃO - >=40/45 dBZ")
    add("-" * 120)
    ps = pixel_season[
        (pixel_season["smoothing_window_pixels"] == 1)
        & (
            pixel_season["association_mode"]
            == "within_patch_centered"
        )
        & pixel_season["radar_field"].isin(["ge_40", "ge_45"])
    ]
    for radar_field in ["ge_40", "ge_45"]:
        add(f"[{radar_field}]")
        for season in ["DJF", "MAM", "JJA", "SON"]:
            t = top_abs(
                ps[
                    (ps["radar_field"] == radar_field)
                    & (ps["season_code"] == season)
                ],
                "pearson_r",
                min(k, 6),
            )
            add(f"  {season}")
            if t.empty:
                add("  sem resultados")
            else:
                add(
                    t[
                        ["predictor", "pearson_r"]
                    ].to_string(index=False)
                )
        add("")

    add("AVISOS / NOTAS")
    add("-" * 120)
    for item in summary.get("warnings", []):
        add(f"- WARNING: {item}")
    for item in summary.get("methodological_notes", []):
        add(f"- NOTE: {item}")

    add("")
    add("=" * 120)
    add("FIM DA FASE 10")
    add("=" * 120)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
