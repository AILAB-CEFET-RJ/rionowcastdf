#!/usr/bin/env python3
"""
CorrDiff - Fase 16 v2
Preparação train-only para baselines clássicos e handoff NVIDIA CorrDiff.

Não treina redes neurais.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from phase16_common import open_group, zarr_take_first_axis, infer_datetime_unit


PHASE_VERSION = "phase16-baselines-v2-classical-nvidia-handoff"

CHANNELS = [
    "tcwv", "t2m", "u10", "v10",
    "t_850", "r_850", "u_850", "v_850",
    "t_500", "r_500", "u_500", "v_500",
]


def parse_args() -> argparse.Namespace:
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
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs/16_baselines"),
    )
    p.add_argument("--chunk-size", type=int, default=2048)
    p.add_argument("--local-timezone", default="America/Sao_Paulo")
    p.add_argument("--expected-patches-per-timestamp", type=int, default=12)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def setup_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{path} is not empty. Use --overwrite."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def setup_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("corrdiff.phase16v2.prepare")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(
        path / "phase16_prepare.log",
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def timestamps_to_utc(values: np.ndarray) -> pd.DatetimeIndex:
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.datetime64):
        return pd.DatetimeIndex(pd.to_datetime(arr, utc=True))
    return pd.DatetimeIndex(
        pd.to_datetime(
            arr,
            unit=infer_datetime_unit(arr),
            utc=True,
        )
    )


def validate_patch_order(
    timestamps: np.ndarray,
    expected_per_timestamp: int,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    ts = timestamps_to_utc(timestamps)
    n = len(ts)
    k = expected_per_timestamp

    if n % k != 0:
        raise RuntimeError(
            f"Patch count {n} is not divisible by {k}."
        )

    ns = ts.asi8.reshape(-1, k)
    if not np.all(ns == ns[:, [0]]):
        raise RuntimeError(
            "Expected contiguous groups of identical timestamps."
        )
    if np.any(np.diff(ns[:, 0]) <= 0):
        raise RuntimeError(
            "Timestamp groups are not strictly increasing."
        )

    slot = np.tile(
        np.arange(k, dtype=np.uint8),
        len(ns),
    )
    unique_ts = pd.DatetimeIndex(
        pd.to_datetime(ns[:, 0], utc=True)
    )
    return unique_ts, slot


def main() -> None:
    args = parse_args()
    setup_output(args.output_dir, args.overwrite)
    logger = setup_logger(args.output_dir)

    zarr_path = args.dataset_dir / "train.zarr"
    train_idx_path = args.phase15_dir / "train_patch_indices.npy"

    if not zarr_path.exists():
        raise FileNotFoundError(zarr_path)
    if not train_idx_path.exists():
        raise FileNotFoundError(train_idx_path)

    root = open_group(zarr_path)
    x_arr = root["input"]
    y_arr = root["target"]
    ts_arr = root["timestamps"]

    n = int(x_arr.shape[0])
    if int(y_arr.shape[0]) != n or int(ts_arr.shape[0]) != n:
        raise RuntimeError("input/target/timestamps length mismatch.")
    if int(x_arr.shape[1]) != len(CHANNELS):
        raise RuntimeError(
            f"Expected 12 input channels; got {x_arr.shape[1]}."
        )

    unique_ts, slot_codes = validate_patch_order(
        np.asarray(ts_arr[:]),
        args.expected_patches_per_timestamp,
    )

    local = unique_ts.tz_convert(args.local_timezone)
    month_patch = np.repeat(
        local.month.to_numpy(dtype=np.uint8),
        args.expected_patches_per_timestamp,
    )
    hour_patch = np.repeat(
        local.hour.to_numpy(dtype=np.uint8),
        args.expected_patches_per_timestamp,
    )

    np.save(args.output_dir / "patch_slot_codes.npy", slot_codes)
    np.save(args.output_dir / "patch_month_local.npy", month_patch)
    np.save(args.output_dir / "patch_hour_local.npy", hour_patch)

    train_idx = np.load(train_idx_path).astype(np.int64)
    if train_idx.ndim != 1:
        raise RuntimeError("train_patch_indices.npy must be 1-D.")

    c, h, w = (
        int(x_arr.shape[1]),
        int(x_arr.shape[2]),
        int(x_arr.shape[3]),
    )

    sum_x = np.zeros(c, dtype=np.float64)
    sumsq_x = np.zeros(c, dtype=np.float64)
    count_x = np.zeros(c, dtype=np.int64)

    # climatology in stored target space: [month, hour, slot, y, x]
    clim_sum = np.zeros(
        (12, 24, args.expected_patches_per_timestamp, h, w),
        dtype=np.float64,
    )
    clim_count = np.zeros(
        (12, 24, args.expected_patches_per_timestamp),
        dtype=np.int64,
    )

    slot_sum = np.zeros(
        (args.expected_patches_per_timestamp, h, w),
        dtype=np.float64,
    )
    slot_count = np.zeros(
        args.expected_patches_per_timestamp,
        dtype=np.int64,
    )

    global_sum = np.zeros((h, w), dtype=np.float64)
    global_count = 0

    logger.info("=" * 88)
    logger.info("CORRDIFF PHASE 16 v2 - PREPARE CLASSICAL BASELINES")
    logger.info("Train patches: %d", len(train_idx))
    logger.info("Input shape: %s", tuple(x_arr.shape))
    logger.info("Target shape: %s", tuple(y_arr.shape))
    logger.info("=" * 88)

    for start in range(0, len(train_idx), args.chunk_size):
        idx = train_idx[start:start + args.chunk_size]

        x = np.asarray(
            zarr_take_first_axis(x_arr, idx),
            dtype=np.float64,
        )
        y = np.asarray(
            zarr_take_first_axis(
                y_arr,
                idx,
                tail_selection=(0, slice(None), slice(None)),
            ),
            dtype=np.float64,
        )

        sum_x += x.sum(axis=(0, 2, 3))
        sumsq_x += np.square(x).sum(axis=(0, 2, 3))
        count_x += x.shape[0] * h * w

        keys = (
            (
                month_patch[idx].astype(np.int64) - 1
            ) * 24
            + hour_patch[idx].astype(np.int64)
        ) * args.expected_patches_per_timestamp + slot_codes[idx].astype(np.int64)

        flat_sum = clim_sum.reshape(-1, h, w)
        flat_count = clim_count.reshape(-1)

        for key in np.unique(keys):
            m = keys == key
            flat_sum[key] += y[m].sum(axis=0)
            flat_count[key] += int(m.sum())

        slots = slot_codes[idx].astype(np.int64)
        for slot in np.unique(slots):
            m = slots == slot
            slot_sum[slot] += y[m].sum(axis=0)
            slot_count[slot] += int(m.sum())

        global_sum += y.sum(axis=0)
        global_count += int(len(y))

        if start == 0 or (start // args.chunk_size) % 50 == 0:
            logger.info(
                "Processed %d / %d train patches",
                min(start + len(idx), len(train_idx)),
                len(train_idx),
            )

    mean_x = sum_x / count_x
    var_x = sumsq_x / count_x - np.square(mean_x)
    std_x = np.sqrt(np.maximum(var_x, 1e-12))

    np.savez(
        args.output_dir / "train_input_normalization.npz",
        channel_names=np.array(CHANNELS, dtype=object),
        mean=mean_x.astype(np.float32),
        std=std_x.astype(np.float32),
        count=count_x,
    )

    slot_mean = slot_sum / np.maximum(
        slot_count[:, None, None],
        1,
    )
    global_mean = global_sum / max(global_count, 1)

    clim_mean = np.empty_like(clim_sum, dtype=np.float32)
    for month in range(12):
        for hour in range(24):
            for slot in range(args.expected_patches_per_timestamp):
                nbin = int(clim_count[month, hour, slot])
                if nbin > 0:
                    clim_mean[month, hour, slot] = (
                        clim_sum[month, hour, slot] / nbin
                    ).astype(np.float32)
                elif slot_count[slot] > 0:
                    clim_mean[month, hour, slot] = (
                        slot_mean[slot].astype(np.float32)
                    )
                else:
                    clim_mean[month, hour, slot] = (
                        global_mean.astype(np.float32)
                    )

    np.savez_compressed(
        args.output_dir / "train_classical_baselines.npz",
        global_mean_target_log1p=global_mean.astype(np.float32),
        slot_mean_target_log1p=slot_mean.astype(np.float32),
        month_hour_slot_mean_target_log1p=clim_mean,
        month_hour_slot_counts=clim_count,
        slot_counts=slot_count,
        local_timezone=np.array(
            args.local_timezone,
            dtype=object,
        ),
    )

    metric_protocol = {
        "phase_version": PHASE_VERSION,
        "development_evaluation_split": "validation",
        "locked_primary_test": "test_primary",
        "locked_stress_ood_test": "test_stress_ood",
        "thresholds_dbz": [20, 30, 40, 45],
        "point_metrics": ["RMSE", "MAE", "bias"],
        "point_domains": [
            "stored_log1p",
            "reconstructed_dBZ",
        ],
        "categorical_metrics": [
            "precision",
            "POD/recall",
            "FAR",
            "CSI",
            "F1",
            "frequency_bias",
            "ETS",
            "deterministic_Brier",
        ],
        "fss_nominal_support_km": [2, 4, 8, 16],
        "patch_intensity_metrics": [
            "RMSE/MAE/bias of patch maximum dBZ",
            "event-conditioned max >=30/40/45",
        ],
        "reporting_groups": [
            "overall",
            "season",
            "persistence_common",
        ],
        "selection_policy": (
            "No learned-model selection is performed in Phase 16 v2. "
            "2023/2024 remain locked for future NVIDIA CorrDiff experiments."
        ),
    }
    (
        args.output_dir / "metric_protocol.json"
    ).write_text(
        json.dumps(metric_protocol, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    prep = {
        "phase_version": PHASE_VERSION,
        "fit_split": "train",
        "fit_split_policy": "train-only",
        "n_train_patches": int(len(train_idx)),
        "n_unique_timestamps_total": int(len(unique_ts)),
        "channels": CHANNELS,
        "input_shape": list(x_arr.shape),
        "target_shape": list(y_arr.shape),
        "patches_per_timestamp": int(
            args.expected_patches_per_timestamp
        ),
        "local_timezone": args.local_timezone,
        "classical_baselines": [
            "zero",
            "train_global_mean_field",
            "train_slot_mean",
            "climatology_month_hour_slot",
            "persistence_1h_radar_reference",
        ],
        "no_neural_training_in_phase16": True,
    }
    (
        args.output_dir / "baseline_preparation.json"
    ).write_text(
        json.dumps(prep, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("Preparation complete: %s", args.output_dir)


if __name__ == "__main__":
    main()
