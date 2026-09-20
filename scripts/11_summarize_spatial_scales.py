#!/usr/bin/env python3
"""Resumo textual da Fase 11 - análise formal de escalas espaciais."""

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

SEASONS = ["DJF", "MAM", "JJA", "SON"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--phase11-dir",
        type=Path,
        default=Path("analysis_outputs/11_spatial_scales"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_outputs/summary_phase11.txt"),
    )
    p.add_argument("--top-k", type=int, default=8)
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


def energy_profile_table(
    energy: pd.DataFrame,
    fields: list[str],
) -> pd.DataFrame:
    t = energy[
        (energy["orientation"] == "ALL_DETAIL")
        & energy["field"].isin(fields)
    ][
        [
            "field",
            "haar_level",
            "nominal_support_km",
            "detail_energy_fraction",
            "rms_coefficient",
        ]
    ].copy()
    return t.sort_values(["field", "haar_level"])


def predictor_scale_summary(
    energy: pd.DataFrame,
) -> pd.DataFrame:
    t = energy[
        (energy["orientation"] == "ALL_DETAIL")
        & (energy["field_group"] == "predictor")
    ].copy()

    rows = []
    for field, g in t.groupby("field", observed=True):
        fine = g.loc[
            g["nominal_support_km"] <= 8,
            "detail_energy_fraction",
        ].sum()
        mid = g.loc[
            g["nominal_support_km"] == 16,
            "detail_energy_fraction",
        ].sum()
        coarse = g.loc[
            g["nominal_support_km"] >= 32,
            "detail_energy_fraction",
        ].sum()
        peak = g.loc[g["detail_energy_fraction"].idxmax()]
        rows.append({
            "predictor": field,
            "fine_energy_fraction_le_8km": fine,
            "energy_fraction_16km": mid,
            "coarse_energy_fraction_ge_32km": coarse,
            "peak_nominal_support_km": peak["nominal_support_km"],
            "peak_energy_fraction": peak["detail_energy_fraction"],
        })

    return pd.DataFrame(rows).sort_values(
        "fine_energy_fraction_le_8km",
        ascending=False,
    )


def main():
    args = parse_args()
    d = args.phase11_dir
    k = args.top_k

    summary = json.loads(
        (d / "analysis_summary.json").read_text(encoding="utf-8")
    )
    strata = pd.read_parquet(d / "patch_strata_counts.parquet")
    energy = pd.read_parquet(d / "haar_scale_energy.parquet")
    energy_season = pd.read_parquet(
        d / "haar_scale_energy_by_season.parquet"
    )
    haar = pd.read_parquet(d / "haar_same_scale_associations.parquet")
    haar_season = pd.read_parquet(
        d / "haar_same_scale_associations_by_season.parquet"
    )
    cross = pd.read_parquet(
        d / "haar_patch_energy_cross_scale.parquet"
    )
    coarse = pd.read_parquet(d / "coarse_grain_associations.parquet")
    coarse_season = pd.read_parquet(
        d / "coarse_grain_associations_by_season.parquet"
    )

    lines = []
    add = lines.append

    add("=" * 124)
    add("CORRDIFF - FASE 11 - ANÁLISE FORMAL DE ESCALAS ESPACIAIS")
    add("=" * 124)
    add(f"Versão: {summary.get('phase_version')}")
    add(f"Input shape: {summary.get('input_shape')}")
    add(f"Target shape: {summary.get('target_shape')}")
    add(
        f"Patches: {summary.get('total_patches')} | "
        f"amostrados: {summary.get('sampled_patches')} "
        f"({100*summary.get('sample_ratio', float('nan')):.2f}%)"
    )
    add(f"Unidade radar: {summary.get('radar_unit')}")
    add(
        "Resolução nominal da grade CorrDiff: "
        f"{summary.get('radar_resolution_km')} km/pixel"
    )
    add(f"Níveis Haar: {summary.get('haar_levels')}")
    add(
        "Suportes Haar nominais: "
        f"{summary.get('haar_scale_supports')}"
    )
    add(
        "Blocos coarse graining: "
        f"{summary.get('coarse_block_widths_km')} km"
    )
    add(f"Escopo: {summary.get('scope')}")
    add("")

    add("ESTRATOS DE INTENSIDADE - SCAN EXATO DO TARGET")
    add("-" * 124)
    add(strata.to_string(index=False))
    add("")

    add("ENERGIA ESPACIAL HAAR DO RADAR")
    add("-" * 124)
    add(
        energy_profile_table(
            energy,
            RADAR_ORDER,
        ).to_string(index=False)
    )
    add("")

    add("DISTRIBUIÇÃO DE ENERGIA ESPACIAL DOS PREDITORES")
    add("-" * 124)
    ps = predictor_scale_summary(energy)
    add(ps.to_string(index=False))
    add("")

    add("ASSOCIAÇÃO HAAR NA MESMA ESCALA - ALL_DETAIL")
    add("-" * 124)
    hp = haar[haar["orientation"] == "ALL_DETAIL"].copy()

    for radar_field in RADAR_ORDER:
        add(f"[{radar_field}]")
        for support in sorted(hp["nominal_support_km"].unique()):
            t = top_abs(
                hp[
                    (hp["radar_field"] == radar_field)
                    & (hp["nominal_support_km"] == support)
                ],
                "pearson_r",
                k,
            )
            add(f"  suporte nominal = {support:g} km")
            if t.empty:
                add("  sem resultados")
            else:
                add(
                    t[
                        ["predictor", "pearson_r"]
                    ].to_string(index=False)
                )
        add("")

    add("ASSOCIAÇÃO HAAR ORIENTADA - >=40/45 dBZ")
    add("-" * 124)
    for radar_field in ["ge_40", "ge_45"]:
        add(f"[{radar_field}]")
        for support in sorted(haar["nominal_support_km"].unique()):
            t0 = haar[
                (haar["radar_field"] == radar_field)
                & (haar["nominal_support_km"] == support)
                & (haar["orientation"] != "ALL_DETAIL")
            ]
            t = top_abs(t0, "pearson_r", min(k, 6))
            add(f"  suporte nominal = {support:g} km")
            if t.empty:
                add("  sem resultados")
            else:
                add(
                    t[
                        ["orientation", "predictor", "pearson_r"]
                    ].to_string(index=False)
                )
        add("")

    add("COARSE GRAINING - ASSOCIAÇÃO CENTRALIZADA DENTRO DO PATCH")
    add("-" * 124)
    cg = coarse[
        coarse["association_mode"] == "within_patch_centered"
    ].copy()
    for radar_field in ["target_log1p", "ge_30", "ge_40", "ge_45"]:
        add(f"[{radar_field}]")
        for width in sorted(cg["nominal_block_width_km"].unique()):
            t = top_abs(
                cg[
                    (cg["radar_field"] == radar_field)
                    & (cg["nominal_block_width_km"] == width)
                ],
                "pearson_r",
                k,
            )
            add(f"  bloco = {width:g} km")
            if t.empty:
                add("  sem resultados")
            else:
                add(
                    t[
                        ["predictor", "pearson_r"]
                    ].to_string(index=False)
                )
        add("")

    add("RAW vs CENTRALIZADO POR ESCALA - PREDITORES DE MAIOR |r|")
    add("-" * 124)
    for radar_field in ["target_log1p", "ge_30", "ge_40", "ge_45"]:
        add(f"[{radar_field}]")
        for width in sorted(coarse["nominal_block_width_km"].unique()):
            for mode in ["raw", "within_patch_centered"]:
                t = top_abs(
                    coarse[
                        (coarse["radar_field"] == radar_field)
                        & (
                            coarse["nominal_block_width_km"]
                            == width
                        )
                        & (coarse["association_mode"] == mode)
                    ],
                    "pearson_r",
                    3,
                )
                if t.empty:
                    continue
                vals = ", ".join(
                    f"{r.predictor}={r.pearson_r:+.4f}"
                    for r in t.itertuples()
                )
                add(f"  {width:g} km | {mode}: {vals}")
        add("")

    add("CORRELAÇÃO ENTRE ENERGIA DE ESCALAS - TOP PARES")
    add("-" * 124)
    for radar_field in ["target_log1p", "ge_30", "ge_40", "ge_45"]:
        t = top_abs(
            cross[cross["radar_field"] == radar_field],
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
                        "predictor_nominal_support_km",
                        "radar_nominal_support_km",
                        "pearson_r",
                    ]
                ].to_string(index=False)
            )
        add("")

    add("SAZONALIDADE DA ENERGIA DO RADAR - >=45 dBZ")
    add("-" * 124)
    t = energy_season[
        (energy_season["field"] == "ge_45")
        & (energy_season["orientation"] == "ALL_DETAIL")
    ][
        [
            "season_code",
            "nominal_support_km",
            "detail_energy_fraction",
            "rms_coefficient",
        ]
    ].copy()
    add(t.sort_values(["season_code", "nominal_support_km"]).to_string(index=False))
    add("")

    add("SAZONALIDADE DA ASSOCIAÇÃO HAAR - >=45 dBZ")
    add("-" * 124)
    hs = haar_season[
        (haar_season["radar_field"] == "ge_45")
        & (haar_season["orientation"] == "ALL_DETAIL")
    ]
    for season in SEASONS:
        add(f"[{season}]")
        for support in sorted(hs["nominal_support_km"].unique()):
            t = top_abs(
                hs[
                    (hs["season_code"] == season)
                    & (hs["nominal_support_km"] == support)
                ],
                "pearson_r",
                min(k, 5),
            )
            if t.empty:
                continue
            vals = ", ".join(
                f"{r.predictor}={r.pearson_r:+.4f}"
                for r in t.itertuples()
            )
            add(f"  {support:g} km: {vals}")
        add("")

    add("SAZONALIDADE DO COARSE GRAINING CENTRALIZADO - >=45 dBZ")
    add("-" * 124)
    cs = coarse_season[
        (coarse_season["radar_field"] == "ge_45")
        & (
            coarse_season["association_mode"]
            == "within_patch_centered"
        )
    ]
    for season in SEASONS:
        add(f"[{season}]")
        for width in sorted(cs["nominal_block_width_km"].unique()):
            t = top_abs(
                cs[
                    (cs["season_code"] == season)
                    & (cs["nominal_block_width_km"] == width)
                ],
                "pearson_r",
                min(k, 5),
            )
            if t.empty:
                continue
            vals = ", ".join(
                f"{r.predictor}={r.pearson_r:+.4f}"
                for r in t.itertuples()
            )
            add(f"  {width:g} km: {vals}")
        add("")

    add("AVISOS / NOTAS")
    add("-" * 124)
    for item in summary.get("warnings", []):
        add(f"- WARNING: {item}")
    for item in summary.get("methodological_notes", []):
        add(f"- NOTE: {item}")

    add("")
    add("=" * 124)
    add("FIM DA FASE 11")
    add("=" * 124)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
