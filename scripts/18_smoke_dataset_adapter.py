#!/usr/bin/env python3
"""
Phase 18 - smoke-test the RioNowcast custom dataset against the real
NVIDIA CorrDiff DownscalingDataset interface.

No neural training. No test-primary / stress-OOD access.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


PHASE_VERSION = "phase18-nvidia-corrdiff-native-integration-v1-smoke-only"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--corrdiff-dir", type=Path, required=True)
    p.add_argument("--adapter-path", type=Path, required=True)
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
        "--phase17-dir",
        type=Path,
        default=Path("analysis_outputs/17_conditioning_ablations"),
    )
    p.add_argument(
        "--stats-dir",
        type=Path,
        default=Path(
            "analysis_outputs/18_nvidia_integration/conditioning_stats"
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(
            "analysis_outputs/18_nvidia_integration/"
            "dataset_smoke_audit.json"
        ),
    )
    p.add_argument("--sample-count", type=int, default=8)
    return p.parse_args()


def import_adapter(adapter_path: Path, corrdiff_dir: Path):
    sys.path.insert(0, str(corrdiff_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            "rionowcast_corrdiff_dataset",
            adapter_path,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not create module spec.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        # Keep corrdiff path available while dataset class imports/uses datasets.base.
        pass


def validate_dataset(ds, expected_channels: int, sample_count: int):
    n = min(sample_count, len(ds))
    rows = []

    for i in range(n):
        y, x = ds[i]

        if not isinstance(x, torch.Tensor):
            raise RuntimeError("input is not torch.Tensor")
        if not isinstance(y, torch.Tensor):
            raise RuntimeError("target is not torch.Tensor")

        if tuple(x.shape) != (
            expected_channels,
            32,
            32,
        ):
            raise RuntimeError(
                f"unexpected input shape: {tuple(x.shape)}"
            )
        if tuple(y.shape) != (1, 32, 32):
            raise RuntimeError(
                f"unexpected target shape: {tuple(y.shape)}"
            )
        if not torch.isfinite(x).all():
            raise RuntimeError("non-finite normalized input")
        if not torch.isfinite(y).all():
            raise RuntimeError("non-finite target")

        rows.append({
            "idx": i,
            "input_mean": float(x.mean()),
            "input_std": float(x.std()),
            "target_min": float(y.min()),
            "target_max": float(y.max()),
            "target_mean": float(y.mean()),
        })

    return rows


def main():
    args = parse_args()

    corrdiff = args.corrdiff_dir.expanduser().resolve()
    adapter_path = args.adapter_path.expanduser().resolve()
    module = import_adapter(adapter_path, corrdiff)
    Dataset = module.RioNowcastCorrDiffDataset

    data_path = (
        args.dataset_dir / "train.zarr"
    ).expanduser().resolve()
    ablation = (
        args.phase17_dir / "ablation_matrix.json"
    ).expanduser().resolve()

    test_primary = np.load(
        args.phase15_dir / "test_primary_patch_indices.npy"
    ).astype(np.int64)
    stress = np.load(
        args.phase15_dir / "test_stress_ood_patch_indices.npy"
    ).astype(np.int64)

    audit = {
        "phase_version": PHASE_VERSION,
        "corrdiff_dir": str(corrdiff),
        "adapter": str(adapter_path),
        "experiments": {},
        "test_lock_checks": {},
    }

    for experiment_id, n_channels in [
        ("C00", 12),
        ("C07", 20),
    ]:
        audit["experiments"][experiment_id] = {}

        for split_name, idx_name in [
            ("train", "train_patch_indices.npy"),
            ("validation", "validation_patch_indices.npy"),
        ]:
            split_path = (
                args.phase15_dir / idx_name
            ).expanduser().resolve()

            split_idx = np.load(
                split_path
            ).astype(np.int64)

            # Strong split-lock audit.
            if np.intersect1d(
                split_idx,
                test_primary,
                assume_unique=False,
            ).size:
                raise RuntimeError(
                    f"{split_name} intersects test_primary"
                )
            if np.intersect1d(
                split_idx,
                stress,
                assume_unique=False,
            ).size:
                raise RuntimeError(
                    f"{split_name} intersects test_stress_ood"
                )

            ds = Dataset(
                data_path=str(data_path),
                split_indices_path=str(split_path),
                ablation_matrix_path=str(ablation),
                experiment_id=experiment_id,
                stats_path=str(
                    (
                        args.stats_dir
                        / f"{experiment_id}.npz"
                    ).expanduser().resolve()
                ),
                split_name=split_name,
                allow_locked_split=False,
            )

            channel_names = [
                c.name for c in ds.input_channels()
            ]
            rows = validate_dataset(
                ds,
                expected_channels=n_channels,
                sample_count=args.sample_count,
            )

            audit["experiments"][
                experiment_id
            ][split_name] = {
                "length": len(ds),
                "input_channels": channel_names,
                "image_shape": list(ds.image_shape()),
                "output_channels": [
                    c.name
                    for c in ds.output_channels()
                ],
                "info": ds.info(),
                "sample_diagnostics": rows,
            }

    # Explicitly confirm locked test files are refused.
    for split_name, idx_name in [
        (
            "test_primary",
            "test_primary_patch_indices.npy",
        ),
        (
            "test_stress_ood",
            "test_stress_ood_patch_indices.npy",
        ),
    ]:
        refused = False
        message = None
        try:
            Dataset(
                data_path=str(data_path),
                split_indices_path=str(
                    (
                        args.phase15_dir / idx_name
                    ).expanduser().resolve()
                ),
                ablation_matrix_path=str(ablation),
                experiment_id="C00",
                stats_path=str(
                    (
                        args.stats_dir / "C00.npz"
                    ).expanduser().resolve()
                ),
                split_name=split_name,
                allow_locked_split=False,
            )
        except RuntimeError as exc:
            refused = True
            message = str(exc)

        if not refused:
            raise RuntimeError(
                f"Locked split {split_name} was not refused."
            )

        audit["test_lock_checks"][split_name] = {
            "refused": True,
            "message": message,
        }

    audit["status"] = "PASS"

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(json.dumps(
        audit,
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
