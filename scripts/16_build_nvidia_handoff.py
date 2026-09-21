#!/usr/bin/env python3
"""
CorrDiff - Fase 16 v2
Gera um handoff portátil para o ambiente NVIDIA CorrDiff.

Não altera o repositório NVIDIA. A integração exata com configs/classes do
CorrDiff fica reservada para a fase de integração do ambiente.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


PHASE_VERSION = "phase16-baselines-v2.2-resolution-safe-persistence"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("datasets/corrdiff_2011_2024"),
    )
    p.add_argument(
        "--phase15-dir",
        type=Path,
        default=Path("analysis_outputs/15_formal_splits"),
    )
    p.add_argument(
        "--phase16-dir",
        type=Path,
        default=Path("analysis_outputs/16_baselines"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def main():
    args = parse_args()
    out = (
        args.output_dir
        if args.output_dir is not None
        else args.phase16_dir / "nvidia_handoff"
    )

    if out.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{out} exists; use --overwrite."
            )
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    required = [
        args.phase15_dir / "train_patch_indices.npy",
        args.phase15_dir / "validation_patch_indices.npy",
        args.phase15_dir / "test_primary_patch_indices.npy",
        args.phase15_dir / "test_stress_ood_patch_indices.npy",
        args.phase15_dir / "excluded_purge_patch_indices.npy",
        args.phase15_dir / "split_config.json",
        args.phase15_dir / "split_summary.parquet",
        args.phase16_dir / "train_input_normalization.npz",
        args.phase16_dir / "metric_protocol.json",
        args.phase16_dir / "baseline_preparation.json",
    ]

    for p in required:
        if not p.exists():
            raise FileNotFoundError(p)

    copied = []
    for source in required:
        dest = out / source.name
        shutil.copy2(source, dest)
        copied.append({
            "name": dest.name,
            "sha256": sha256(dest),
            "bytes": dest.stat().st_size,
            "source": str(source),
        })

    contract = {
        "phase_version": PHASE_VERSION,
        "purpose": (
            "portable data/split/normalization contract for subsequent "
            "NVIDIA CorrDiff integration"
        ),
        "dataset": {
            "zarr_path": str(args.dataset_dir / "train.zarr"),
            "input_key": "input",
            "target_key": "target",
            "timestamps_key": "timestamps",
            "input_shape_per_patch": [12, 32, 32],
            "target_shape_per_patch": [1, 32, 32],
            "patches_per_timestamp": 12,
            "grid_nominal_km": 2.0,
        },
        "channels_in_order": [
            "tcwv", "t2m", "u10", "v10",
            "t_850", "r_850", "u_850", "v_850",
            "t_500", "r_500", "u_500", "v_500",
        ],
        "target_semantics": {
            "stored_space": "log1p(clipped radar dBZ)",
            "inverse_for_diagnostics": "dbz = expm1(target)",
            "negative_dbz_in_raw_radar": (
                "not recoverable because builder clips negative dBZ to zero"
            ),
        },
        "split_contract": {
            "train": "2011-2020 with Phase-15 boundary purge",
            "validation": "2021-2022 with Phase-15 boundary purge",
            "test_primary": "2023, locked",
            "test_stress_ood": "2024, locked stress/OOD",
            "purge": "48 h on each side of temporal boundaries",
            "all_patches_same_timestamp_same_split": True,
        },
        "normalization_contract": {
            "artifact": "train_input_normalization.npz",
            "fit_on": "train only",
            "scope": "per input channel over all train patch pixels",
            "do_not_refit_on_validation_or_tests": True,
        },
        "evaluation_contract": {
            "artifact": "metric_protocol.json",
            "primary_thresholds_dbz": [20, 30, 40, 45],
            "primary_fss_supports_km": [2, 4, 8, 16],
            "2023_and_2024_must_be_reported_separately": True,
        },
        "integration_boundary": (
            "This handoff intentionally does not assume NVIDIA CorrDiff class, "
            "Hydra config or trainer names. Exact repository integration belongs "
            "to the later CorrDiff environment integration phase."
        ),
    }

    (
        out / "corrdiff_data_contract.json"
    ).write_text(
        json.dumps(contract, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    copied.append({
        "name": "corrdiff_data_contract.json",
        "sha256": sha256(out / "corrdiff_data_contract.json"),
        "bytes": (out / "corrdiff_data_contract.json").stat().st_size,
        "source": "generated",
    })

    manifest = {
        "phase_version": PHASE_VERSION,
        "handoff_directory": str(out),
        "files": copied,
        "next_environment": "NVIDIA CorrDiff environment",
        "next_action": (
            "consume split indices + train-only normalization in the native "
            "CorrDiff data pipeline; do not train neural baselines in Phase 16"
        ),
    }

    (
        out / "nvidia_handoff_manifest.json"
    ).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(out)
    print("files:", len(copied) + 1)


if __name__ == "__main__":
    main()
