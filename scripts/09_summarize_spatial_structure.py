#!/usr/bin/env python3
"""Resumo textual da Fase 9 - Estrutura espacial do radar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EVENT_ORDER = ["gt_0", "ge_20", "ge_30", "ge_40", "ge_45"]
DIRECTION_ORDER = ["EW", "NS", "NW_SE", "NE_SW"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--phase9-dir",
        type=Path,
        default=Path("analysis_outputs/09_spatial_structure"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_outputs/summary_phase9.txt"),
    )
    return p.parse_args()


def coherence_length(
    table: pd.DataFrame,
    threshold: float = 0.5,
) -> float:
    t = table.sort_values("distance_km")
    below = t[t["pearson_r"] < threshold]
    if below.empty:
        return np.nan
    return float(below.iloc[0]["distance_km"])


def main():
    args = parse_args()
    d = args.phase9_dir

    summary = json.loads(
        (d / "analysis_summary.json").read_text(encoding="utf-8")
    )
    strata = pd.read_parquet(d / "patch_strata_counts.parquet")
    spatial = pd.read_parquet(d / "spatial_lag_statistics.parquet")
    morph = pd.read_parquet(d / "morphology_summary.parquet")
    season = pd.read_parquet(d / "morphology_by_season.parquet")

    lines = []
    add = lines.append

    add("=" * 116)
    add("CORRDIFF - FASE 9 - ESTRUTURA ESPACIAL DO RADAR")
    add("=" * 116)
    add(f"Versão: {summary.get('phase_version')}")
    add(f"Unidade radar: {summary.get('radar_unit')}")
    add(f"Target armazenado: {summary.get('stored_target')}")
    add(f"Target shape: {summary.get('target_shape')}")
    add(f"Patch shape: {summary.get('patch_shape')}")
    add(
        f"Patches: {summary.get('total_patches')} | "
        f"amostrados: {summary.get('sampled_patches')} "
        f"({100*summary.get('sample_ratio', float('nan')):.2f}%)"
    )
    add(f"Resolução: {summary.get('radar_resolution_km')} km/pixel")
    add(f"Steps: {summary.get('spatial_steps_pixels')}")
    add(f"Escopo: {summary.get('scope')}")
    add("")

    add("ESTRATOS DE INTENSIDADE - SCAN EXATO DE TODOS OS PATCHES")
    add("-" * 116)
    add(strata.to_string(index=False))
    add("")

    add("TAXA EXATA DE PATCHES COM EVENTO")
    add("-" * 116)
    rates = summary.get("exact_patch_any_event_rates", {})
    for event_id in EVENT_ORDER:
        if event_id in rates:
            add(f"{event_id:>8}: {100*float(rates[event_id]):9.5f}%")
    add("")

    add("CORRELAÇÃO ESPACIAL DO TARGET LOG1P")
    add("-" * 116)
    t = spatial[spatial["field"] == "target_log1p"][
        [
            "direction",
            "step_pixels",
            "distance_km",
            "pearson_r",
            "semivariance",
            "weighted_pair_mass",
        ]
    ].copy()
    t["direction"] = pd.Categorical(
        t["direction"],
        categories=DIRECTION_ORDER,
        ordered=True,
    )
    t = t.sort_values(["direction", "step_pixels"])
    add(t.to_string(index=False))
    add("")

    add("COMPRIMENTO DE COERÊNCIA APROXIMADO (primeiro lag com r < 0.5)")
    add("-" * 116)
    rows = []
    for field in ["target_log1p", "gt_0", "ge_20", "ge_30", "ge_40", "ge_45"]:
        for direction in DIRECTION_ORDER:
            g = spatial[
                (spatial["field"] == field)
                & (spatial["direction"] == direction)
            ]
            if g.empty:
                continue
            rows.append({
                "field": field,
                "direction": direction,
                "first_distance_km_r_lt_0_5": coherence_length(g, 0.5),
                "r_at_min_distance": float(
                    g.sort_values("distance_km").iloc[0]["pearson_r"]
                ),
                "r_at_max_distance": float(
                    g.sort_values("distance_km").iloc[-1]["pearson_r"]
                ),
            })
    add(pd.DataFrame(rows).to_string(index=False))
    add("")

    add("ESTRUTURA BINÁRIA POR LIMIAR - CORRELAÇÃO E MISMATCH")
    add("-" * 116)
    t = spatial[
        spatial["field"].isin(["ge_20", "ge_30", "ge_40", "ge_45"])
    ][
        [
            "field",
            "direction",
            "step_pixels",
            "distance_km",
            "pearson_r",
            "pair_mismatch_probability",
        ]
    ].copy()
    t["direction"] = pd.Categorical(
        t["direction"],
        categories=DIRECTION_ORDER,
        ordered=True,
    )
    t = t.sort_values(["field", "direction", "step_pixels"])
    add(t.to_string(index=False))
    add("")

    add("ANISOTROPIA EW vs NS")
    add("-" * 116)
    ani_rows = []
    for field in ["target_log1p", "ge_20", "ge_30", "ge_40", "ge_45"]:
        piv = spatial[
            spatial["field"] == field
        ].pivot_table(
            index=["step_pixels"],
            columns="direction",
            values="pearson_r",
            aggfunc="first",
        )
        for step, row in piv.iterrows():
            if "EW" not in row or "NS" not in row:
                continue
            ew = row.get("EW", np.nan)
            ns = row.get("NS", np.nan)
            ani_rows.append({
                "field": field,
                "step_pixels": int(step),
                "EW_r": ew,
                "NS_r": ns,
                "EW_minus_NS": ew - ns
                if np.isfinite(ew) and np.isfinite(ns)
                else np.nan,
            })
    add(pd.DataFrame(ani_rows).to_string(index=False))
    add("")

    add("MORFOLOGIA GLOBAL DOS PATCHES")
    add("-" * 116)
    cols = [
        "event_id",
        "threshold_dbz",
        "patch_any_event_rate",
        "mean_event_fraction_all_patches",
        "event_fraction_p90_all_patches",
        "event_fraction_p99_all_patches",
        "mean_event_fraction_given_event",
        "mean_component_count_given_event",
        "mean_largest_component_fraction_of_event",
        "mean_internal_edge_transition_density_given_event",
        "border_touch_rate_given_event",
        "largest_component_border_touch_rate_given_event",
        "mean_largest_component_bbox_fill",
    ]
    add(morph[cols].to_string(index=False))
    add("")

    add("MORFOLOGIA POR ESTAÇÃO - >=30/40/45 dBZ")
    add("-" * 116)
    t = season[
        season["event_id"].isin(["ge_30", "ge_40", "ge_45"])
    ][
        [
            "season_code",
            "event_id",
            "patch_any_event_rate",
            "mean_event_fraction_given_event",
            "mean_component_count_given_event",
            "mean_largest_component_fraction_of_event",
            "border_touch_rate_given_event",
            "mean_internal_edge_transition_density_given_event",
        ]
    ]
    add(t.to_string(index=False))
    add("")

    add("AVISOS / NOTAS")
    add("-" * 116)
    for item in summary.get("warnings", []):
        add(f"- WARNING: {item}")
    for item in summary.get("methodological_notes", []):
        add(f"- NOTE: {item}")

    add("")
    add("=" * 116)
    add("FIM DA FASE 9")
    add("=" * 116)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
