#!/usr/bin/env python3
"""
Phase 18 - fit train-only normalization for all Phase-17 conditioning channels.

Reads the full 20-channel representational set (12 raw + 8 deterministic
derived) once over the Phase-15 train patches, then emits experiment-specific
stats for C00..C10.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr


PHASE_VERSION = "phase18-nvidia-corrdiff-native-integration-v1-smoke-only"

RAW_CHANNELS = [
    "tcwv", "t2m", "u10", "v10",
    "t_850", "r_850", "u_850", "v_850",
    "t_500", "r_500", "u_500", "v_500",
]
RAW_INDEX = {name: i for i, name in enumerate(RAW_CHANNELS)}

DERIVED_CHANNELS = [
    "wind_speed_10",
    "wind_speed_850",
    "wind_speed_500",
    "delta_t_500_850",
    "delta_t_850_surface",
    "delta_r_500_850",
    "bulk_wind_diff_10_850",
    "bulk_wind_diff_850_500",
]

ALL20 = RAW_CHANNELS + DERIVED_CHANNELS


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
        "--phase17-dir",
        type=Path,
        default=Path("analysis_outputs/17_conditioning_ablations"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "analysis_outputs/18_nvidia_integration/conditioning_stats"
        ),
    )
    p.add_argument("--chunk-size", type=int, default=2048)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def open_group(path: Path):
    try:
        return zarr.open_group(str(path), mode="r")
    except Exception:
        return zarr.open(str(path), mode="r")


def zarr_take_first_axis(arr, indices: np.ndarray):
    idx = np.asarray(indices, dtype=np.int64)
    selection = (
        idx,
        *[slice(None) for _ in range(arr.ndim - 1)],
    )
    if hasattr(arr, "oindex"):
        return np.asarray(arr.oindex[selection])
    if hasattr(arr, "get_orthogonal_selection"):
        return np.asarray(
            arr.get_orthogonal_selection(selection)
        )
    raise RuntimeError(
        "Zarr implementation lacks orthogonal indexing."
    )


def derive_batch(raw: np.ndarray) -> np.ndarray:
    """raw: [B,12,H,W] -> derived [B,8,H,W]."""
    def r(name):
        return raw[:, RAW_INDEX[name]]

    u10 = r("u10")
    v10 = r("v10")
    u850 = r("u_850")
    v850 = r("v_850")
    u500 = r("u_500")
    v500 = r("v_500")

    fields = [
        np.sqrt(u10 * u10 + v10 * v10),
        np.sqrt(u850 * u850 + v850 * v850),
        np.sqrt(u500 * u500 + v500 * v500),
        r("t_500") - r("t_850"),
        r("t_850") - r("t2m"),
        r("r_500") - r("r_850"),
        np.sqrt(
            (u10 - u850) ** 2
            + (v10 - v850) ** 2
        ),
        np.sqrt(
            (u850 - u500) ** 2
            + (v850 - v500) ** 2
        ),
    ]
    return np.stack(fields, axis=1)


def main():
    args = parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"{args.output_dir} not empty; use --overwrite"
            )
        import shutil
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    zarr_path = args.dataset_dir / "train.zarr"
    train_idx_path = args.phase15_dir / "train_patch_indices.npy"
    ablation_path = args.phase17_dir / "ablation_matrix.json"

    for p in [zarr_path, train_idx_path, ablation_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    root = open_group(zarr_path)
    x_arr = root["input"]
    train_idx = np.load(train_idx_path).astype(np.int64)

    if x_arr.shape[1] != 12:
        raise RuntimeError(
            f"Expected raw12, got input shape {x_arr.shape}"
        )

    n_channels = len(ALL20)
    sums = np.zeros(n_channels, dtype=np.float64)
    sumsq = np.zeros(n_channels, dtype=np.float64)
    counts = np.zeros(n_channels, dtype=np.int64)

    for start in range(0, len(train_idx), args.chunk_size):
        idx = train_idx[start:start + args.chunk_size]
        raw = np.asarray(
            zarr_take_first_axis(x_arr, idx),
            dtype=np.float64,
        )
        derived = derive_batch(raw)
        full = np.concatenate(
            [raw, derived],
            axis=1,
        )

        sums += full.sum(axis=(0, 2, 3))
        sumsq += np.square(full).sum(axis=(0, 2, 3))
        counts += (
            full.shape[0]
            * full.shape[2]
            * full.shape[3]
        )

        if start == 0 or (start // args.chunk_size) % 50 == 0:
            print(
                f"processed {min(start + len(idx), len(train_idx))}"
                f" / {len(train_idx)} train patches"
            )

    mean = sums / counts
    var = sumsq / counts - np.square(mean)
    std = np.sqrt(np.maximum(var, 1e-12))

    np.savez(
        args.output_dir / "all20_train_stats.npz",
        channel_names=np.array(ALL20, dtype=object),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        count=counts,
    )

    doc = json.loads(
        ablation_path.read_text(encoding="utf-8")
    )
    experiments = doc["experiments"]

    index = {name: i for i, name in enumerate(ALL20)}
    emitted = []

    for exp in experiments:
        channels = list(exp["channels"])
        positions = [index[c] for c in channels]

        out = args.output_dir / f"{exp['experiment_id']}.npz"
        np.savez(
            out,
            experiment_id=np.array(
                exp["experiment_id"],
                dtype=object,
            ),
            channel_names=np.array(channels, dtype=object),
            mean=mean[positions].astype(np.float32),
            std=std[positions].astype(np.float32),
            count=counts[positions],
            fit_split=np.array("train", dtype=object),
            phase_version=np.array(
                PHASE_VERSION,
                dtype=object,
            ),
        )

        emitted.append({
            "experiment_id": exp["experiment_id"],
            "name": exp["name"],
            "n_channels": len(channels),
            "stats_file": str(out),
        })

    audit = {
        "phase_version": PHASE_VERSION,
        "fit_split": "train",
        "n_train_patches": int(len(train_idx)),
        "all20_channel_order": ALL20,
        "all_means_finite": bool(
            np.isfinite(mean).all()
        ),
        "all_stds_positive_finite": bool(
            np.isfinite(std).all()
            and np.all(std > 0)
        ),
        "experiments": emitted,
        "target_normalization": (
            "identity in Phase 18; target remains stored log1p(clipped dBZ)"
        ),
    }

    (
        args.output_dir / "conditioning_stats_audit.json"
    ).write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
