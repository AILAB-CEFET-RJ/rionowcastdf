#!/usr/bin/env python3
"""Produce a textual summary of the Phase 17 conditioning-ablation plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--phase17-dir",
        type=Path,
        default=Path(
            "analysis_outputs/17_conditioning_ablations"
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            "analysis_outputs/summary_phase17.txt"
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    d = args.phase17_dir

    matrix = pd.read_csv(d / "ablation_matrix.csv")
    stage_r = pd.read_csv(d / "planned_runs_stage_R.csv")
    stage_d = pd.read_csv(
        d / "planned_runs_stage_D_mandatory.csv"
    )
    optional = pd.read_csv(
        d / "optional_corrdiff_experiments.csv"
    )
    manifest = json.loads(
        (d / "phase17_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    adapter = json.loads(
        (
            d / "nvidia_ablation_adapter_contract.json"
        ).read_text(encoding="utf-8")
    )

    lines = []
    add = lines.append

    add("=" * 140)
    add(
        "CORRDIFF - FASE 17 - DESENHO FORMAL DAS ABLAÇÕES "
        "DE CONDICIONAMENTO"
    )
    add("=" * 140)
    add(f"Versão: {manifest['phase_version']}")
    add(f"Experimentos: {manifest['n_experiments']}")
    add(
        "Runs Stage R (regression screen): "
        f"{manifest['n_stage_R_runs']}"
    )
    add(
        "Runs Stage D obrigatórios (CorrDiff): "
        f"{manifest['n_stage_D_mandatory_runs']}"
    )
    add(
        "Experimentos CorrDiff opcionais: "
        f"{manifest['n_stage_D_optional_experiments']}"
    )
    add("Testes 2023/2024 bloqueados: True")
    add("")

    add("MATRIZ DE ABLAÇÕES")
    add("-" * 140)
    show = matrix[
        [
            "experiment_id",
            "name",
            "tier",
            "n_raw_channels",
            "n_derived_channels",
            "n_total_channels",
            "channels",
        ]
    ]
    add(show.to_string(index=False))
    add("")

    add("STAGE R - REGRESSION MODEL NATIVO")
    add("-" * 140)
    add(
        "Todos os 11 experimentos, sementes "
        + str(sorted(stage_r["seed"].unique().tolist()))
        + "."
    )
    add(
        "Treino: Phase-15 train; seleção/avaliação: validation; "
        "2023/2024 inacessíveis."
    )
    add("")

    add("STAGE D - CORRDIFF CONFIRMATÓRIO")
    add("-" * 140)
    mandatory = (
        stage_d[["experiment_id", "experiment_name"]]
        .drop_duplicates()
        .sort_values("experiment_id")
    )
    add("Obrigatórios:")
    add(mandatory.to_string(index=False))
    add("")
    add("Opcionais sob regra pré-declarada de orçamento:")
    add(
        optional[
            [
                "experiment_id",
                "experiment_name",
                "tier",
                "activation_rule",
            ]
        ].to_string(index=False)
    )
    add("")

    add("CONTRATO NVIDIA")
    add("-" * 140)
    add(
        "Raw order: "
        + ", ".join(adapter["raw_channel_order"])
    )
    add(
        "Derived order: "
        + ", ".join(adapter["derived_channel_order"])
    )
    add(
        "Target unchanged: "
        + str(adapter["target_is_unchanged"])
    )
    add(
        "Test splits locked: "
        + str(adapter["test_splits_must_remain_locked"])
    )
    add("")

    add("COMPARAÇÕES CIENTÍFICAS")
    add("-" * 140)
    add(
        "Necessidade: C00 vs C04/C05/C06 "
        "(leave-one-family-out)."
    )
    add(
        "Suficiência: C00 vs C01/C02/C03 "
        "(single-family conditioning)."
    )
    add(
        "Representação: C00 vs C07/C08/C09/C10 "
        "(derived channels)."
    )
    add("")

    add("REGRAS")
    add("-" * 140)
    add(
        "- Mesmo split, geometria, target, arquitetura, orçamento "
        "e política de checkpoint dentro de cada estágio."
    )
    add(
        "- Normalização ajustada somente no train para exatamente "
        "os canais presentes em cada experimento."
    )
    add(
        "- Variáveis derivadas são transformações determinísticas: "
        "podem facilitar representação, mas não adicionam informação."
    )
    add(
        "- delta_t representa diferenças de temperatura; não chamar "
        "automaticamente de instabilidade."
    )
    add(
        "- bulk_wind_diff é diferença vetorial em m/s; não é shear "
        "normalizado por altura."
    )
    add(
        "- Reportar as três sementes e sumarizar centro + dispersão; "
        "não selecionar a melhor seed."
    )
    add(
        "- Não colapsar a seleção em um único score: ler primeiro "
        "FSS/CSI >=30/40, depois extremos e então erro global."
    )
    add(
        "- Não usar 2023/2024 para escolha de canais, arquitetura, "
        "loss ou hiperparâmetros."
    )
    add("")

    add("PRÓXIMA FASE")
    add("-" * 140)
    add(
        "Fase 18: mapear este contrato para o checkout real do NVIDIA "
        "CorrDiff (Dataset/DataLoader/Hydra/regression/diffusion), "
        "implementar channel selection/derived channels e executar "
        "smoke tests sem abrir os testes temporais."
    )
    add("")
    add("=" * 140)
    add("FIM DA FASE 17")
    add("=" * 140)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
