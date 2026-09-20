#!/usr/bin/env python3
"""Resumo textual da Fase 16 - baselines formais."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--phase16-dir",
        type=Path,
        default=Path("analysis_outputs/16_baselines"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_outputs/summary_phase16.txt"),
    )
    return p.parse_args()


def read_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def add_df(lines: list[str], df: pd.DataFrame, cols=None):
    if df.empty:
        lines.append("(sem dados)")
        return
    if cols is not None:
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
    lines.append(df.to_string(index=False))


def main():
    args = parse_args()
    d = args.phase16_dir

    prep = json.loads(
        (d / "baseline_preparation.json").read_text(encoding="utf-8")
    )
    protocol = json.loads(
        (d / "metric_protocol.json").read_text(encoding="utf-8")
    )

    train_manifest_path = (
        d / "learned_baseline_training_manifest.json"
    )
    train_manifest = (
        json.loads(train_manifest_path.read_text(encoding="utf-8"))
        if train_manifest_path.exists()
        else {}
    )

    eval_manifest_path = d / "evaluation_manifest.json"
    eval_manifest = (
        json.loads(eval_manifest_path.read_text(encoding="utf-8"))
        if eval_manifest_path.exists()
        else {}
    )

    hist = read_if_exists(d / "training_history.parquet")
    point = read_if_exists(d / "point_metrics.parquet")
    thr = read_if_exists(d / "threshold_metrics.parquet")
    fss = read_if_exists(d / "fss_metrics.parquet")
    patch = read_if_exists(d / "patch_max_metrics.parquet")
    prob = read_if_exists(d / "probabilistic_metrics.parquet")
    group = read_if_exists(d / "group_patch_metrics.parquet")
    cohorts = read_if_exists(d / "evaluation_cohorts.parquet")

    lines = []
    add = lines.append

    add("=" * 136)
    add("CORRDIFF - FASE 16 - BASELINES FORMAIS E PROTOCOLO DE AVALIAÇÃO")
    add("=" * 136)
    add(f"Versão: {prep.get('phase_version')}")
    add(f"Split usado para estatísticas data-dependent: {prep.get('fit_split')}")
    add(f"Patches de treino usados na preparação: {prep.get('n_train_patches')}")
    add(f"Canais: {prep.get('channels')}")
    add("")

    add("PROTOCOLO DE MÉTRICAS")
    add("-" * 136)
    add(f"Development split: {protocol.get('development_evaluation_split')}")
    add(f"Primary test bloqueado: {protocol.get('locked_primary_test')}")
    add(f"Stress/OOD bloqueado: {protocol.get('locked_stress_ood_test')}")
    add(f"Thresholds dBZ: {protocol.get('thresholds_dbz')}")
    add(f"FSS suportes km: {protocol.get('fss_nominal_support_km')}")
    add(f"Point metrics: {protocol.get('point_metrics')}")
    add(f"Categorical metrics: {protocol.get('categorical_metrics')}")
    add(f"Probabilistic metrics: {protocol.get('probabilistic_metrics')}")
    add("")

    add("BASELINES")
    add("-" * 136)
    add("zero")
    add("train_global_mean_field")
    add("train_slot_mean")
    add("climatology_month_hour_slot")
    add("persistence_1h_radar_reference  [usa radar t-1h; referência observacional]")
    add("pixel_mlp                      [1x1; sem contexto espacial]")
    add("unet_deterministic             [U-Net + MSE em log1p]")
    add("unet_gaussian                  [U-Net heteroscedástico + Gaussian NLL]")
    add("")

    if train_manifest:
        add("MANIFESTO DE TREINAMENTO")
        add("-" * 136)
        for k, v in train_manifest.items():
            add(f"{k}: {v}")
        add("")

    if not hist.empty:
        add("HISTÓRICO DE TREINAMENTO - MELHOR ÉPOCA POR MODELO")
        add("-" * 136)
        rows = []
        for model, g in hist.groupby("model", observed=True):
            best = g.sort_values("val_loss").iloc[0]
            rows.append({
                "model": model,
                "best_epoch": int(best["epoch"]),
                "train_loss": float(best["train_loss"]),
                "val_loss": float(best["val_loss"]),
                "val_rmse_log1p": float(best["val_rmse_log1p"]),
                "val_mae_log1p": float(best["val_mae_log1p"]),
                "val_bias_log1p": float(best["val_bias_log1p"]),
            })
        add_df(lines, pd.DataFrame(rows))
        add("")

    if eval_manifest:
        add("MANIFESTO DE AVALIAÇÃO")
        add("-" * 136)
        add(f"Splits avaliados: {eval_manifest.get('evaluated_splits')}")
        add(f"Baselines: {eval_manifest.get('baselines')}")
        add(f"Locked split override: {eval_manifest.get('locked_split_override_used')}")
        add(f"Persistence common: {eval_manifest.get('with_persistence_common')}")
        add("")

    if not cohorts.empty:
        add("COORTES DE AVALIAÇÃO")
        add("-" * 136)
        add_df(
            lines,
            cohorts.sort_values(["split", "cohort"]),
        )
        add("")

    if not point.empty:
        add("MÉTRICAS PONTUAIS")
        add("-" * 136)
        for split in point["split"].drop_duplicates():
            add(f"[{split}]")
            t = point[
                (point["split"] == split)
                & (point["cohort"] == "full_split")
            ].copy()
            add_df(
                lines,
                t.sort_values(
                    ["domain", "baseline"]
                ),
                [
                    "baseline",
                    "cohort",
                    "domain",
                    "rmse",
                    "mae",
                    "bias",
                    "n_pixels",
                ],
            )
            add("")

    if not thr.empty:
        add("MÉTRICAS CATEGÓRICAS - FULL SPLIT")
        add("-" * 136)
        t = thr[
            (thr["cohort"] == "full_split")
            & (thr["threshold_dbz"].isin([30.0, 40.0, 45.0]))
        ].copy()
        add_df(
            lines,
            t.sort_values(
                ["split", "threshold_dbz", "baseline"]
            ),
            [
                "split",
                "baseline",
                "threshold_dbz",
                "precision",
                "pod_recall",
                "far",
                "csi",
                "f1",
                "frequency_bias",
                "ets",
                "brier_deterministic",
            ],
        )
        add("")

        pcommon = thr[
            thr["cohort"] == "persistence_common"
        ].copy()
        if not pcommon.empty:
            add("MÉTRICAS CATEGÓRICAS - PERSISTENCE_COMMON")
            add("-" * 136)
            add_df(
                lines,
                pcommon[
                    pcommon["threshold_dbz"].isin(
                        [30.0, 40.0, 45.0]
                    )
                ].sort_values(
                    ["split", "threshold_dbz", "baseline"]
                ),
                [
                    "split",
                    "baseline",
                    "threshold_dbz",
                    "csi",
                    "pod_recall",
                    "far",
                    "ets",
                    "brier_deterministic",
                ],
            )
            add("")

    if not fss.empty:
        add("FSS - FULL SPLIT")
        add("-" * 136)
        t = fss[
            (fss["cohort"] == "full_split")
            & (fss["threshold_dbz"].isin([30.0, 40.0, 45.0]))
        ].copy()
        add_df(
            lines,
            t.sort_values(
                [
                    "split",
                    "threshold_dbz",
                    "nominal_support_km",
                    "baseline",
                ]
            ),
            [
                "split",
                "baseline",
                "threshold_dbz",
                "nominal_support_km",
                "fss",
            ],
        )
        add("")

    if not patch.empty:
        add("MÁXIMO POR PATCH")
        add("-" * 136)
        t = patch[
            patch["condition"].isin(
                ["all", "target_ge30", "target_ge40", "target_ge45"]
            )
        ].copy()
        add_df(
            lines,
            t.sort_values(
                ["split", "condition", "baseline"]
            ),
            [
                "split",
                "baseline",
                "cohort",
                "condition",
                "n_patches",
                "rmse_max_dbz",
                "mae_max_dbz",
                "bias_max_dbz",
                "target_mean_max_dbz",
                "pred_mean_max_dbz",
            ],
        )
        add("")

    if not prob.empty:
        add("MÉTRICAS PROBABILÍSTICAS")
        add("-" * 136)
        add_df(
            lines,
            prob.sort_values(
                ["split", "metric", "threshold_dbz"],
                na_position="first",
            ),
            [
                "split",
                "baseline",
                "cohort",
                "metric",
                "threshold_dbz",
                "value",
                "n_pixels",
            ],
        )
        add("")

    if not group.empty:
        add("DIAGNÓSTICO SAZONAL - MÁXIMO POR PATCH")
        add("-" * 136)
        add_df(
            lines,
            group.sort_values(
                ["split", "group_value", "baseline"]
            ),
            [
                "split",
                "baseline",
                "cohort",
                "group_type",
                "group_value",
                "n_patches",
                "rmse_max_dbz",
                "mae_max_dbz",
                "bias_max_dbz",
                "target_patch_event_rate_ge_30",
                "pred_patch_event_rate_ge_30",
                "target_patch_event_rate_ge_40",
                "pred_patch_event_rate_ge_40",
                "target_patch_event_rate_ge_45",
                "pred_patch_event_rate_ge_45",
            ],
        )
        add("")

    add("REGRAS DE INTERPRETAÇÃO")
    add("-" * 136)
    add("- zero/global/slot/climatology são referências não aprendidas.")
    add("- pixel_mlp testa quanto é possível obter sem contexto espacial.")
    add("- unet_deterministic testa mapeamento determinístico ERA5→Radar com contexto espacial.")
    add("- unet_gaussian é um baseline probabilístico simples, não um substituto para um modelo generativo zero-inflated.")
    add("- persistence_1h usa radar observado em t-1h; não é input-equivalente ao CorrDiff ERA5-only.")
    add("- Compare persistence somente na coorte persistence_common e, idealmente, rode os demais baselines nessa mesma coorte.")
    add("- Test primary 2023 e stress/OOD 2024 só devem ser avaliados depois do design freeze.")
    add("- Métricas pixelwise podem ser dominadas por zeros; CSI/FSS/event-conditioned max metrics devem ser lidas em conjunto.")
    add("- Resultados de 2024 devem ser acompanhados do coverage identificado nas Fases 14-15.")
    add("")

    add("=" * 136)
    add("FIM DA FASE 16")
    add("=" * 136)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
