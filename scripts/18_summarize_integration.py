#!/usr/bin/env python3
"""Summarize Phase 18 NVIDIA CorrDiff native-integration smoke results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PHASE_VERSION = "phase18-nvidia-corrdiff-native-integration-v1-smoke-only"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--phase18-dir",
        type=Path,
        default=Path(
            "analysis_outputs/18_nvidia_integration"
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            "analysis_outputs/summary_phase18.txt"
        ),
    )
    return p.parse_args()


def load_optional(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    args = parse_args()
    d = args.phase18_dir

    checkout = load_optional(
        d / "phase18_nvidia_checkout_audit.json"
    )
    stats = load_optional(
        d / "conditioning_stats"
        / "conditioning_stats_audit.json"
    )
    dataset = load_optional(
        d / "dataset_smoke_audit.json"
    )
    install = load_optional(
        d / "phase18_install_manifest.json"
    )

    native_candidates = sorted(
        d.glob("native_smoke_audit*.json")
    )
    native = [
        load_optional(p)
        for p in native_candidates
    ]

    lines = []
    add = lines.append

    add("=" * 140)
    add(
        "CORRDIFF - FASE 18 - INTEGRAÇÃO NATIVA NVIDIA "
        "+ SMOKE TESTS"
    )
    add("=" * 140)
    add(f"Versão: {PHASE_VERSION}")
    add("Escopo: integração/smoke; nenhum treinamento completo.")
    add("Test primary 2023: LOCKED")
    add("Stress/OOD 2024: LOCKED")
    add("")

    add("CHECKOUT NVIDIA")
    add("-" * 140)
    if checkout:
        add(f"Status: {checkout.get('status')}")
        add(f"CorrDiff dir: {checkout.get('corrdiff_dir')}")
        git = checkout.get("git", {})
        add(f"Git commit: {git.get('commit')}")
        add(f"Git branch: {git.get('branch')}")
        add(f"Missing files: {checkout.get('missing_files')}")
        add(
            "Strategy: "
            + str(checkout.get("integration_strategy"))
        )
    else:
        add("NOT RUN")
    add("")

    add("NORMALIZAÇÃO POR ABLAÇÃO")
    add("-" * 140)
    if stats:
        add(f"Fit split: {stats.get('fit_split')}")
        add(
            "Train patches: "
            + str(stats.get("n_train_patches"))
        )
        add(
            "Means finite: "
            + str(stats.get("all_means_finite"))
        )
        add(
            "Stds positive/finite: "
            + str(stats.get("all_stds_positive_finite"))
        )
        add(
            "Experiments with stats: "
            + str(len(stats.get("experiments", [])))
        )
        add(
            "Target normalization: "
            + str(stats.get("target_normalization"))
        )
    else:
        add("NOT RUN")
    add("")

    add("DATASET ADAPTER SMOKE")
    add("-" * 140)
    if dataset:
        add(f"Status: {dataset.get('status')}")
        for exp_id, exp in dataset.get(
            "experiments", {}
        ).items():
            for split_name, result in exp.items():
                add(
                    f"{exp_id} {split_name}: "
                    f"n={result.get('length')} "
                    f"channels={len(result.get('input_channels', []))} "
                    f"shape={result.get('image_shape')}"
                )
        add(
            "Locked split checks: "
            + json.dumps(
                dataset.get("test_lock_checks", {}),
                ensure_ascii=False,
            )
        )
    else:
        add("NOT RUN")
    add("")

    add("NATIVE CONFIG INSTALL")
    add("-" * 140)
    if install:
        add(
            "Strategy: "
            + str(install.get("strategy"))
        )
        add(
            "Experiments installed: "
            + str(install.get("experiments_installed"))
        )
        add(
            "Training duration smoke: "
            + str(install.get("training_duration_samples"))
        )
        add(
            "Tests accessed: "
            + str(install.get("tests_accessed"))
        )
    else:
        add("NOT RUN")
    add("")

    add("NATIVE NVIDIA SMOKES")
    add("-" * 140)
    if native:
        for item in native:
            add(
                f"{item.get('mode')} | "
                f"{item.get('experiment_id')} | "
                f"{item.get('stage')} | "
                f"{item.get('status')} | "
                f"returncode={item.get('result', {}).get('returncode')}"
            )
    else:
        add("NOT RUN")
    add("")

    add("PHASE-18 EXIT CRITERIA")
    add("-" * 140)
    add("1. Real NVIDIA checkout audit PASS.")
    add("2. C00 and C07 adapter smoke PASS on train + validation.")
    add("3. Locked 2023/2024 dataset access explicitly refused.")
    add("4. Train-only normalization generated for C00..C10.")
    add("5. Hydra native config composition PASS for C00 regression + diffusion.")
    add("6. Tiny native regression forward/backward training smoke PASS on GPU.")
    add("7. Tiny native diffusion smoke PASS using a regression smoke checkpoint.")
    add("8. No full training and no test evaluation performed.")
    add("")

    add("OPEN DESIGN ITEM BEFORE FULL TRAINING")
    add("-" * 140)
    add(
        "Absolute geographic patch position is not yet provided as per-sample "
        "conditioning. The Phase-18 adapter exposes relative 2-km patch coordinates "
        "because DownscalingDataset has dataset-level latitude/longitude accessors. "
        "Before full training, verify whether slot/absolute position should be encoded "
        "explicitly or whether raw meteorological conditioning alone is the intended design."
    )
    add("")

    add("NEXT PHASE")
    add("-" * 140)
    add(
        "Phase 19: freeze target/positional treatment and launch the formal "
        "native regression-model ablation campaign from Stage R (C00..C10, seeds 42/43/44)."
    )
    add("")
    add("=" * 140)
    add("FIM DA FASE 18")
    add("=" * 140)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
