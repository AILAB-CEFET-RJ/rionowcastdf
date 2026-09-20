#!/usr/bin/env python3
"""Resumo textual da Fase 16 v2 - baselines clássicos + handoff NVIDIA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def main():
    args = parse_args()
    d = args.phase16_dir

    prep = json.loads(
        (d / "baseline_preparation.json").read_text(encoding="utf-8")
    )
    protocol = json.loads(
        (d / "metric_protocol.json").read_text(encoding="utf-8")
    )

    eval_manifest_path = d / "classical_evaluation_manifest.json"
    eval_manifest = (
        json.loads(eval_manifest_path.read_text(encoding="utf-8"))
        if eval_manifest_path.exists()
        else {}
    )

    point = read_parquet(d / "point_metrics.parquet")
    thr = read_parquet(d / "threshold_metrics.parquet")
    fss = read_parquet(d / "fss_metrics.parquet")
    patch = read_parquet(d / "patch_max_metrics.parquet")
    seasonal = read_parquet(d / "seasonal_patch_metrics.parquet")
    cohorts = read_parquet(d / "evaluation_cohorts.parquet")

    lines = []
    add = lines.append

    add("=" * 136)
    add("CORRDIFF - FASE 16 v2 - BASELINES CLÁSSICOS E HANDOFF NVIDIA")
    add("=" * 136)
    add(f"Versão: {prep.get('phase_version')}")
    add(f"Fit split: {prep.get('fit_split')}")
    add(f"Patches de treino usados: {prep.get('n_train_patches')}")
    add(f"Treino neural nesta fase: {not prep.get('no_neural_training_in_phase16', False)}")
    add(f"Canais: {prep.get('channels')}")
    add("")

    add("PROTOCOLO")
    add("-" * 136)
    add(f"Development split: {protocol.get('development_evaluation_split')}")
    add(f"Primary test bloqueado: {protocol.get('locked_primary_test')}")
    add(f"Stress/OOD bloqueado: {protocol.get('locked_stress_ood_test')}")
    add(f"Thresholds dBZ: {protocol.get('thresholds_dbz')}")
    add(f"FSS km: {protocol.get('fss_nominal_support_km')}")
    add("")

    add("BASELINES DA FASE 16 v2")
    add("-" * 136)
    for baseline in prep.get("classical_baselines", []):
        add(f"- {baseline}")
    add("")

    if eval_manifest:
        add("MANIFESTO DE AVALIAÇÃO")
        add("-" * 136)
        for k, v in eval_manifest.items():
            add(f"{k}: {v}")
        add("")

    if not cohorts.empty:
        add("COORTES")
        add("-" * 136)
        add(cohorts.sort_values(["split", "cohort"]).to_string(index=False))
        add("")

    if not point.empty:
        add("MÉTRICAS PONTUAIS - FULL SPLIT")
        add("-" * 136)
        t = point[point["cohort"] == "full_split"].copy()
        add(
            t.sort_values(
                ["split", "domain", "baseline"]
            )[
                [
                    "split",
                    "baseline",
                    "domain",
                    "rmse",
                    "mae",
                    "bias",
                    "n_pixels",
                ]
            ].to_string(index=False)
        )
        add("")

    if not thr.empty:
        add("MÉTRICAS CATEGÓRICAS - FULL SPLIT")
        add("-" * 136)
        t = thr[
            (thr["cohort"] == "full_split")
            & (thr["threshold_dbz"].isin([30.0, 40.0, 45.0]))
        ].copy()
        add(
            t.sort_values(
                ["split", "threshold_dbz", "baseline"]
            )[
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
                ]
            ].to_string(index=False)
        )
        add("")

        common = thr[
            (thr["cohort"] == "persistence_common")
            & (thr["threshold_dbz"].isin([30.0, 40.0, 45.0]))
        ].copy()
        if not common.empty:
            add("MÉTRICAS CATEGÓRICAS - PERSISTENCE_COMMON")
            add("-" * 136)
            add(
                common.sort_values(
                    ["split", "threshold_dbz", "baseline"]
                )[
                    [
                        "split",
                        "baseline",
                        "threshold_dbz",
                        "csi",
                        "pod_recall",
                        "far",
                        "ets",
                    ]
                ].to_string(index=False)
            )
            add("")

    if not fss.empty:
        add("FSS - FULL SPLIT")
        add("-" * 136)
        t = fss[
            (fss["cohort"] == "full_split")
            & (fss["threshold_dbz"].isin([30.0, 40.0, 45.0]))
        ].copy()
        add(
            t.sort_values(
                [
                    "split",
                    "threshold_dbz",
                    "nominal_support_km",
                    "baseline",
                ]
            )[
                [
                    "split",
                    "baseline",
                    "threshold_dbz",
                    "nominal_support_km",
                    "fss",
                ]
            ].to_string(index=False)
        )
        add("")

    if not patch.empty:
        add("MÁXIMO POR PATCH")
        add("-" * 136)
        add(
            patch.sort_values(
                ["split", "condition", "baseline"]
            )[
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
                ]
            ].to_string(index=False)
        )
        add("")

    if not seasonal.empty:
        add("DIAGNÓSTICO SAZONAL - FULL SPLIT")
        add("-" * 136)
        t = seasonal[
            seasonal["cohort"] == "full_split"
        ].copy()
        add(
            t.sort_values(
                ["split", "season_code", "baseline"]
            ).to_string(index=False)
        )
        add("")

    handoff_manifest = (
        d / "nvidia_handoff" / "nvidia_handoff_manifest.json"
    )
    if handoff_manifest.exists():
        handoff = json.loads(
            handoff_manifest.read_text(encoding="utf-8")
        )
        add("HANDOFF NVIDIA")
        add("-" * 136)
        add(f"Diretório: {handoff.get('handoff_directory')}")
        add(f"Próximo ambiente: {handoff.get('next_environment')}")
        add(f"Próxima ação: {handoff.get('next_action')}")
        add("")

    add("INTERPRETAÇÃO")
    add("-" * 136)
    add("- zero/global/slot/climatology medem o piso não aprendido.")
    add("- persistence_1h_radar_reference usa radar t-1h e não é input-equivalente ao CorrDiff ERA5-only.")
    add("- A persistência deve ser comparada na coorte persistence_common.")
    add("- Nenhum treinamento neural faz parte da Fase 16 v2.")
    add("- Splits, normalização train-only e protocolo de métricas são entregues ao ambiente NVIDIA CorrDiff.")
    add("- A integração com classes/configs nativas do CorrDiff será feita na fase específica de integração do ambiente.")
    add("")

    add("=" * 136)
    add("FIM DA FASE 16 v2")
    add("=" * 136)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
