#!/usr/bin/env python3
"""Resumo textual da Fase 12 - análise multivariada."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--phase12-dir",
        type=Path,
        default=Path("analysis_outputs/12_multivariate"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_outputs/summary_phase12.txt"),
    )
    p.add_argument("--top-k", type=int, default=10)
    return p.parse_args()


def base_predictor(feature: str) -> str:
    if feature.endswith("__mean"):
        return feature[:-6]
    if feature.endswith("__std"):
        return feature[:-5]
    return feature


def aggregate_binary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["event_id", "model", "representation"]
    for keys, g in df.groupby(group_cols, observed=True):
        row = dict(zip(group_cols, keys))
        for col in [
            "brier",
            "brier_skill_vs_climatology",
            "roc_auc",
            "average_precision",
            "ece_10bins",
            "weighted_event_rate",
        ]:
            vals = g[col].to_numpy(dtype=float)
            row[f"{col}_mean"] = float(np.nanmean(vals))
            row[f"{col}_std"] = float(np.nanstd(vals, ddof=0))
        row["n_folds"] = int(len(g))
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_continuous(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["target", "model", "representation"]
    for keys, g in df.groupby(group_cols, observed=True):
        row = dict(zip(group_cols, keys))
        for col in [
            "rmse",
            "mae",
            "bias",
            "rmse_skill_vs_climatology",
            "mae_skill_vs_climatology",
            "weighted_target_mean",
        ]:
            vals = g[col].to_numpy(dtype=float)
            row[f"{col}_mean"] = float(np.nanmean(vals))
            row[f"{col}_std"] = float(np.nanstd(vals, ddof=0))
        row["n_folds"] = int(len(g))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    d = args.phase12_dir
    k = args.top_k

    summary = json.loads(
        (d / "analysis_summary.json").read_text(encoding="utf-8")
    )
    strata = pd.read_parquet(d / "patch_strata_counts.parquet")
    metadata = pd.read_parquet(d / "input_channel_metadata.parquet")
    folds = pd.read_parquet(d / "temporal_fold_definition.parquet")
    corr = pd.read_parquet(d / "weighted_predictor_correlation.parquet")
    pca = pd.read_parquet(d / "pca_explained_variance.parquet")
    loadings = pd.read_parquet(d / "pca_loadings.parquet")
    binary = pd.read_parquet(d / "binary_model_metrics.parquet")
    continuous = pd.read_parquet(d / "continuous_model_metrics.parquet")
    stability = pd.read_parquet(d / "coefficient_stability.parquet")
    status = pd.read_parquet(d / "model_status.parquet")

    b_agg = aggregate_binary(binary)
    c_agg = aggregate_continuous(continuous)

    family_map = dict(
        zip(metadata["predictor"], metadata["family"])
    )

    lines = []
    add = lines.append

    add("=" * 124)
    add("CORRDIFF - FASE 12 - ANÁLISE MULTIVARIADA")
    add("=" * 124)
    add(f"Versão: {summary.get('phase_version')}")
    add(f"Input shape: {summary.get('input_shape')}")
    add(f"Target shape: {summary.get('target_shape')}")
    add(
        f"Patches: {summary.get('total_patches')} | "
        f"amostrados: {summary.get('sampled_patches')} "
        f"({100*summary.get('sample_ratio', float('nan')):.2f}%)"
    )
    add(f"Representações: {summary.get('representations')}")
    add(f"Escopo: {summary.get('scope')}")
    add("")

    add("ESTRATOS DE INTENSIDADE - SCAN EXATO")
    add("-" * 124)
    add(strata.to_string(index=False))
    add("")

    add("FOLDS TEMPORAIS FORWARD-CHAINING")
    add("-" * 124)
    add(folds.to_string(index=False))
    add("")

    add("COLINEARIDADE - MAIORES |CORRELAÇÕES| ENTRE PREDITORES MÉDIOS")
    add("-" * 124)
    c = corr[
        corr["feature_i"] < corr["feature_j"]
    ].copy()
    c["abs_corr"] = c["weighted_correlation"].abs()
    c = c.sort_values("abs_corr", ascending=False).head(25)
    add(
        c[
            ["feature_i", "feature_j", "weighted_correlation"]
        ].to_string(index=False)
    )
    add("")

    add("PCA DOS 20 PREDITORES MÉDIOS")
    add("-" * 124)
    add(
        pca.head(12).to_string(index=False)
    )
    add("")
    add(
        "Resumo PCA: "
        f"{summary.get('pca_means_only')}"
    )
    add("")

    add("LOADINGS DOMINANTES DOS PRIMEIROS COMPONENTES")
    add("-" * 124)
    for pc in ["PC1", "PC2", "PC3", "PC4", "PC5"]:
        t = loadings[loadings["component"] == pc].copy()
        if t.empty:
            continue
        t["abs_loading"] = t["loading"].abs()
        t = t.sort_values("abs_loading", ascending=False).head(k)
        add(f"[{pc}]")
        add(
            t[["feature", "loading"]].to_string(index=False)
        )
        add("")

    add("MÉTRICAS BINÁRIAS - MÉDIA DOS FOLDS")
    add("-" * 124)
    cols = [
        "event_id",
        "model",
        "representation",
        "n_folds",
        "brier_mean",
        "brier_skill_vs_climatology_mean",
        "roc_auc_mean",
        "average_precision_mean",
        "ece_10bins_mean",
    ]
    add(
        b_agg.sort_values(
            ["event_id", "brier_mean"]
        )[cols].to_string(index=False)
    )
    add("")

    add("GANHO NÃO LINEAR - HISTGRADIENTBOOSTING vs LOGISTIC L2")
    add("-" * 124)
    rows = []
    for event_id in sorted(b_agg["event_id"].unique()):
        h = b_agg[
            (b_agg["event_id"] == event_id)
            & (b_agg["model"] == "hist_gradient_boosting")
            & (b_agg["representation"] == "means_plus_std")
        ]
        l = b_agg[
            (b_agg["event_id"] == event_id)
            & (b_agg["model"] == "logistic_l2")
            & (b_agg["representation"] == "means_plus_std")
        ]
        if h.empty or l.empty:
            continue
        hr = h.iloc[0]
        lr = l.iloc[0]
        rows.append({
            "event_id": event_id,
            "delta_brier_hist_minus_logistic": (
                hr["brier_mean"] - lr["brier_mean"]
            ),
            "delta_bss_hist_minus_logistic": (
                hr["brier_skill_vs_climatology_mean"]
                - lr["brier_skill_vs_climatology_mean"]
            ),
            "delta_ap_hist_minus_logistic": (
                hr["average_precision_mean"]
                - lr["average_precision_mean"]
            ),
            "delta_auc_hist_minus_logistic": (
                hr["roc_auc_mean"] - lr["roc_auc_mean"]
            ),
        })
    add(pd.DataFrame(rows).to_string(index=False))
    add("")

    add("GANHO DE VARIABILIDADE ESPACIAL - MEANS+STD vs MEANS ONLY")
    add("-" * 124)
    rows = []
    for event_id in sorted(b_agg["event_id"].unique()):
        a = b_agg[
            (b_agg["event_id"] == event_id)
            & (b_agg["model"] == "logistic_l2")
            & (b_agg["representation"] == "means_plus_std")
        ]
        b = b_agg[
            (b_agg["event_id"] == event_id)
            & (b_agg["model"] == "logistic_l2")
            & (b_agg["representation"] == "means_only")
        ]
        if a.empty or b.empty:
            continue
        ar = a.iloc[0]
        br = b.iloc[0]
        rows.append({
            "event_id": event_id,
            "delta_brier_means_std_minus_means": (
                ar["brier_mean"] - br["brier_mean"]
            ),
            "delta_bss_means_std_minus_means": (
                ar["brier_skill_vs_climatology_mean"]
                - br["brier_skill_vs_climatology_mean"]
            ),
            "delta_ap_means_std_minus_means": (
                ar["average_precision_mean"]
                - br["average_precision_mean"]
            ),
        })
    add(pd.DataFrame(rows).to_string(index=False))
    add("")

    add("COEFICIENTES LOGÍSTICOS ESTÁVEIS - MEANS_PLUS_STD")
    add("-" * 124)
    s = stability[
        (stability["target_type"] == "binary")
        & (stability["model"] == "logistic_l2")
        & (stability["representation"] == "means_plus_std")
    ].copy()
    for event_id in ["gt_0", "ge_20", "ge_30", "ge_40", "ge_45"]:
        t = s[s["target"] == event_id].copy()
        t = t[t["n_folds"] >= 3]
        t = t.sort_values(
            ["sign_consistency", "coefficient_abs_mean"],
            ascending=[False, False],
        ).head(k)
        add(f"[{event_id}]")
        if t.empty:
            add("sem resultados")
        else:
            add(
                t[
                    [
                        "feature",
                        "n_folds",
                        "coefficient_mean",
                        "coefficient_std",
                        "coefficient_abs_mean",
                        "sign_consistency",
                    ]
                ].to_string(index=False)
            )
        add("")

    add("IMPORTÂNCIA AGREGADA POR FAMÍLIA - LOGISTIC L2")
    add("-" * 124)
    fam_rows = []
    s2 = s.copy()
    s2["predictor"] = s2["feature"].map(base_predictor)
    s2["family"] = s2["predictor"].map(family_map)
    for (target, family), g in s2.groupby(
        ["target", "family"],
        observed=True,
    ):
        fam_rows.append({
            "target": target,
            "family": family,
            "mean_abs_standardized_coefficient": float(
                g["coefficient_abs_mean"].mean()
            ),
            "max_abs_standardized_coefficient": float(
                g["coefficient_abs_mean"].max()
            ),
            "mean_sign_consistency": float(
                g["sign_consistency"].mean()
            ),
            "n_features": int(len(g)),
        })
    add(
        pd.DataFrame(fam_rows)
        .sort_values(
            ["target", "mean_abs_standardized_coefficient"],
            ascending=[True, False],
        )
        .to_string(index=False)
    )
    add("")

    add("MÉTRICAS CONTÍNUAS - MÉDIA DOS FOLDS")
    add("-" * 124)
    cols = [
        "target",
        "model",
        "representation",
        "n_folds",
        "rmse_mean",
        "rmse_skill_vs_climatology_mean",
        "mae_mean",
        "mae_skill_vs_climatology_mean",
        "bias_mean",
    ]
    add(
        c_agg.sort_values(
            ["target", "rmse_mean"]
        )[cols].to_string(index=False)
    )
    add("")

    add("COEFICIENTES RIDGE ESTÁVEIS - MEANS_PLUS_STD")
    add("-" * 124)
    rs = stability[
        (stability["target_type"] == "continuous")
        & (stability["model"] == "ridge_l2")
        & (stability["representation"] == "means_plus_std")
    ].copy()
    for target in [
        "max_target_log1p",
        "positive_pixel_fraction",
        "event_pixel_fraction_ge_30",
        "event_pixel_fraction_ge_40",
        "event_pixel_fraction_ge_45",
    ]:
        t = rs[rs["target"] == target].copy()
        t = t[t["n_folds"] >= 3]
        t = t.sort_values(
            ["sign_consistency", "coefficient_abs_mean"],
            ascending=[False, False],
        ).head(k)
        add(f"[{target}]")
        if t.empty:
            add("sem resultados")
        else:
            add(
                t[
                    [
                        "feature",
                        "coefficient_mean",
                        "coefficient_std",
                        "coefficient_abs_mean",
                        "sign_consistency",
                    ]
                ].to_string(index=False)
            )
        add("")

    add("DETALHE TEMPORAL - >=45 dBZ")
    add("-" * 124)
    t = binary[binary["event_id"] == "ge_45"].copy()
    add(
        t[
            [
                "fold_id",
                "model",
                "representation",
                "weighted_event_rate",
                "brier",
                "brier_skill_vs_climatology",
                "roc_auc",
                "average_precision",
                "ece_10bins",
            ]
        ].sort_values(
            ["fold_id", "brier"]
        ).to_string(index=False)
    )
    add("")

    errors = status[status["status"] != "ok"]
    add("STATUS DOS MODELOS")
    add("-" * 124)
    add(
        f"Registros de status: {len(status)} | "
        f"não-ok: {len(errors)}"
    )
    if not errors.empty:
        add(errors.to_string(index=False))
    add("")

    add("AVISOS / NOTAS")
    add("-" * 124)
    for item in summary.get("warnings", []):
        add(f"- WARNING: {item}")
    for item in summary.get("methodological_notes", []):
        add(f"- NOTE: {item}")

    add("")
    add("=" * 124)
    add("FIM DA FASE 12")
    add("=" * 124)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
