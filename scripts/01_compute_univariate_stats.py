#!/usr/bin/env python3
"""Phase 1 univariate analysis for the CorrDiff ERA5 -> Radar dataset.

This script characterizes the marginal distributions of the 12 ERA5 predictor
channels and the radar target produced by the CorrDiff builder. It is designed
to build on Phase 0 rather than repeat the entire expensive audit scan.

Main ideas
----------
* Read a time-spread sample of complete Zarr blocks/chunks for detailed
  distributional diagnostics.
* Reuse exact Phase 0 means/std/min/max and exact target-event rates when those
  artifacts are available.
* Keep the stored target domain (log1p) separate from the reconstructed
  post-clip radar-legend domain (expm1(target)).
* Do not label the reconstructed radar domain as dBZ unless the source product
  documentation explicitly confirms that unit.
* Quantify physically suspicious relative-humidity values (<0 or >100) without
  silently clipping them.

Expected input
--------------
<dataset_dir>/train.zarr
<dataset_dir>/metadata.json
<phase0_dir>/channel_summary.parquet                        # optional
<phase0_dir>/target_event_rates_global.parquet              # optional
<phase0_dir>/radar_target_semantics.json                    # optional

Outputs
-------
<output_dir>/
    analysis_summary.json
    sampling_blocks.parquet
    input_summary.parquet
    input_quantiles.parquet
    input_histograms.parquet
    relative_humidity_quality.parquet
    target_summary.parquet
    target_quantiles.parquet
    target_histograms.parquet
    target_threshold_rates_sampled.parquet
    target_threshold_rates_reference.parquet                # when Phase 0 exists
    univariate_samples.npz
    phase1.log

The sampled distributions are diagnostics of the *training-patch distribution*.
Because patches overlap, central spatial pixels can be represented multiple times.
This is intentional for model-input characterization and is not the same as a
spatially de-duplicated radar climatology.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import zarr

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable: Iterable, **_: Any) -> Iterable:
        return iterable


PHASE1_VERSION = "phase1-univariate-v1"
QUANTILE_LEVELS = (
    ("p001", 0.001),
    ("p01", 0.01),
    ("p05", 0.05),
    ("p10", 0.10),
    ("p25", 0.25),
    ("p50", 0.50),
    ("p75", 0.75),
    ("p90", 0.90),
    ("p95", 0.95),
    ("p99", 0.99),
    ("p999", 0.999),
)
RADAR_THRESHOLDS = (
    ("gt_0", "> 0", 0.0, False),
    ("ge_20", ">= 20", 20.0, True),
    ("ge_25", ">= 25", 25.0, True),
    ("ge_30", ">= 30", 30.0, True),
    ("ge_35", ">= 35", 35.0, True),
    ("ge_40", ">= 40", 40.0, True),
    ("ge_45", ">= 45", 45.0, True),
    ("ge_50", ">= 50", 50.0, True),
)

# These are expected ERA5 conventional units. They are recorded as hints rather
# than silently treated as metadata embedded in the Zarr.
UNIT_HINTS = {
    "tcwv": "kg m^-2",
    "t2m": "K",
    "u10": "m s^-1",
    "v10": "m s^-1",
    "t_850": "K",
    "r_850": "%",
    "u_850": "m s^-1",
    "v_850": "m s^-1",
    "t_500": "K",
    "r_500": "%",
    "u_500": "m s^-1",
    "v_500": "m s^-1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Phase 1 univariate diagnostics for CorrDiff Zarr data."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("datasets/corrdiff_2011_2024"),
        help="Directory containing train.zarr and metadata.json.",
    )
    parser.add_argument(
        "--zarr-path",
        type=Path,
        default=None,
        help="Optional explicit Zarr path. Defaults to <dataset-dir>/train.zarr.",
    )
    parser.add_argument(
        "--phase0-dir",
        type=Path,
        default=Path("analysis_outputs/00_quality"),
        help=(
            "Phase 0 output directory. Exact Phase 0 statistics/event rates are "
            "reused when available."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs/01_univariate"),
        help="Directory where Phase 1 outputs are written.",
    )
    parser.add_argument(
        "--sample-patches",
        type=int,
        default=32768,
        help=(
            "Approximate number of patches sampled for detailed distributions. "
            "Sampling is performed by complete contiguous Zarr blocks for efficient I/O."
        ),
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=0,
        help=(
            "Sampling block size in patches. 0 uses the first Zarr chunk dimension "
            "(recommended)."
        ),
    )
    parser.add_argument(
        "--sampling-strategy",
        choices=("systematic", "random"),
        default="systematic",
        help=(
            "systematic spreads sampled blocks through the full time-ordered dataset; "
            "random chooses blocks uniformly without replacement."
        ),
    )
    parser.add_argument(
        "--reservoir-values",
        type=int,
        default=300000,
        help=(
            "Maximum stored visualization/quantile sample values per input channel "
            "and target domain."
        ),
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=120,
        help="Number of histogram bins used for sampled distributions.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260914,
        help="Random seed for block and within-block sampling.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    return parser.parse_args()


def setup_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("corrdiff.phase1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    file_handler = logging.FileHandler(output_dir / "phase1.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {path}. Use --overwrite to reuse it."
        )
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def open_zarr_group(path: Path) -> Any:
    try:
        return zarr.open_group(str(path), mode="r")
    except Exception:
        return zarr.open(str(path), mode="r")


def channel_unit_hint(channel: str) -> str:
    if channel in UNIT_HINTS:
        return UNIT_HINTS[channel]
    if channel.startswith("t_"):
        return "K"
    if channel.startswith("r_"):
        return "%"
    if channel.startswith("u_") or channel.startswith("v_"):
        return "m s^-1"
    return "unknown"


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def sanitized_key(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value)


def select_block_ids(
    total_patches: int,
    block_size: int,
    sample_patches: int,
    strategy: str,
    rng: np.random.Generator,
) -> np.ndarray:
    total_blocks = math.ceil(total_patches / block_size)
    needed = min(total_blocks, max(1, math.ceil(sample_patches / block_size)))

    if needed == total_blocks:
        return np.arange(total_blocks, dtype=np.int64)

    if strategy == "random":
        return np.sort(rng.choice(total_blocks, size=needed, replace=False)).astype(np.int64)

    # One block from each equal-width interval over the time-ordered dataset.
    interval = total_blocks / needed
    positions = (np.arange(needed, dtype=np.float64) + rng.random(needed)) * interval
    block_ids = np.floor(positions).astype(np.int64)
    block_ids = np.clip(block_ids, 0, total_blocks - 1)
    # The intervals are non-overlapping when needed <= total_blocks, so duplicates
    # are extremely unlikely/impossible apart from floating edge effects.
    block_ids = np.unique(block_ids)
    if len(block_ids) < needed:
        missing = needed - len(block_ids)
        remaining = np.setdiff1d(np.arange(total_blocks), block_ids, assume_unique=True)
        fill = rng.choice(remaining, size=missing, replace=False)
        block_ids = np.sort(np.concatenate([block_ids, fill]))
    return block_ids.astype(np.int64)


class MultiChannelAccumulator:
    def __init__(self, channels: list[str]) -> None:
        n = len(channels)
        self.channels = channels
        self.total = np.zeros(n, dtype=np.int64)
        self.finite = np.zeros(n, dtype=np.int64)
        self.nan = np.zeros(n, dtype=np.int64)
        self.posinf = np.zeros(n, dtype=np.int64)
        self.neginf = np.zeros(n, dtype=np.int64)
        self.sum = np.zeros(n, dtype=np.float64)
        self.sumsq = np.zeros(n, dtype=np.float64)
        self.min = np.full(n, np.inf, dtype=np.float64)
        self.max = np.full(n, -np.inf, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        # values: (C, K)
        if values.ndim != 2:
            raise ValueError(f"Expected (C,K), got {values.shape}")
        self.total += values.shape[1]
        finite = np.isfinite(values)
        self.finite += finite.sum(axis=1)
        self.nan += np.isnan(values).sum(axis=1)
        self.posinf += np.isposinf(values).sum(axis=1)
        self.neginf += np.isneginf(values).sum(axis=1)
        for idx in range(values.shape[0]):
            vals = values[idx, finite[idx]].astype(np.float64, copy=False)
            if vals.size == 0:
                continue
            self.sum[idx] += vals.sum(dtype=np.float64)
            self.sumsq[idx] += np.square(vals).sum(dtype=np.float64)
            self.min[idx] = min(self.min[idx], float(vals.min()))
            self.max[idx] = max(self.max[idx], float(vals.max()))

    def rows(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for idx, channel in enumerate(self.channels):
            finite = int(self.finite[idx])
            total = int(self.total[idx])
            if finite:
                mean = self.sum[idx] / finite
                variance = max(self.sumsq[idx] / finite - mean * mean, 0.0)
                std = math.sqrt(variance)
                min_value = float(self.min[idx])
                max_value = float(self.max[idx])
            else:
                mean = std = min_value = max_value = float("nan")
            result.append(
                {
                    "channel_index": idx,
                    "channel": channel,
                    "expected_unit": channel_unit_hint(channel),
                    "sampled_total_count": total,
                    "sampled_finite_count": finite,
                    "sampled_finite_ratio": finite / total if total else np.nan,
                    "sampled_nan_count": int(self.nan[idx]),
                    "sampled_posinf_count": int(self.posinf[idx]),
                    "sampled_neginf_count": int(self.neginf[idx]),
                    "sampled_mean": mean,
                    "sampled_std": std,
                    "sampled_min": min_value,
                    "sampled_max": max_value,
                }
            )
        return result


class ScalarAccumulator:
    def __init__(self) -> None:
        self.total = 0
        self.finite = 0
        self.nan = 0
        self.posinf = 0
        self.neginf = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.min = np.inf
        self.max = -np.inf

    def update(self, values: np.ndarray) -> None:
        arr = np.asarray(values).reshape(-1)
        self.total += arr.size
        finite = np.isfinite(arr)
        self.finite += int(finite.sum())
        self.nan += int(np.isnan(arr).sum())
        self.posinf += int(np.isposinf(arr).sum())
        self.neginf += int(np.isneginf(arr).sum())
        vals = arr[finite].astype(np.float64, copy=False)
        if vals.size:
            self.sum += float(vals.sum(dtype=np.float64))
            self.sumsq += float(np.square(vals).sum(dtype=np.float64))
            self.min = min(self.min, float(vals.min()))
            self.max = max(self.max, float(vals.max()))

    def row(self, name: str, unit: str, domain: str) -> dict[str, Any]:
        if self.finite:
            mean = self.sum / self.finite
            variance = max(self.sumsq / self.finite - mean * mean, 0.0)
            std = math.sqrt(variance)
            min_value = float(self.min)
            max_value = float(self.max)
        else:
            mean = std = min_value = max_value = float("nan")
        return {
            "name": name,
            "domain": domain,
            "expected_unit": unit,
            "sampled_total_count": int(self.total),
            "sampled_finite_count": int(self.finite),
            "sampled_finite_ratio": self.finite / self.total if self.total else np.nan,
            "sampled_nan_count": int(self.nan),
            "sampled_posinf_count": int(self.posinf),
            "sampled_neginf_count": int(self.neginf),
            "sampled_mean": mean,
            "sampled_std": std,
            "sampled_min": min_value,
            "sampled_max": max_value,
        }


def distribution_shape(values: np.ndarray) -> tuple[float, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size < 3:
        return np.nan, np.nan
    mean = float(vals.mean())
    centered = vals - mean
    m2 = float(np.mean(centered**2))
    if m2 <= 0:
        return 0.0, -3.0
    skew = float(np.mean(centered**3) / (m2 ** 1.5))
    excess_kurtosis = float(np.mean(centered**4) / (m2 * m2) - 3.0)
    return skew, excess_kurtosis


def make_quantile_rows(
    name: str,
    values: np.ndarray,
    kind: str,
    domain: str,
    unit: str,
) -> list[dict[str, Any]]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return []
    rows: list[dict[str, Any]] = []
    for label, q in QUANTILE_LEVELS:
        rows.append(
            {
                "kind": kind,
                "name": name,
                "domain": domain,
                "expected_unit": unit,
                "quantile": label,
                "probability": q,
                "value": float(np.quantile(vals, q)),
                "sample_value_count": int(vals.size),
            }
        )
    return rows


def histogram_rows(
    name: str,
    values: np.ndarray,
    bins: int,
    kind: str,
    domain: str,
    unit: str,
    subset: str = "all",
) -> list[dict[str, Any]]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return []
    if np.all(vals == vals[0]):
        span = max(abs(float(vals[0])) * 0.01, 1e-6)
        edges = np.linspace(float(vals[0]) - span, float(vals[0]) + span, bins + 1)
        counts, edges = np.histogram(vals, bins=edges)
    else:
        counts, edges = np.histogram(vals, bins=bins)
    total = int(counts.sum())
    rows: list[dict[str, Any]] = []
    for idx, count in enumerate(counts):
        left = float(edges[idx])
        right = float(edges[idx + 1])
        rows.append(
            {
                "kind": kind,
                "name": name,
                "domain": domain,
                "expected_unit": unit,
                "subset": subset,
                "bin_index": idx,
                "bin_left": left,
                "bin_right": right,
                "bin_center": (left + right) / 2.0,
                "count": int(count),
                "ratio": int(count) / total if total else np.nan,
                "sample_value_count": total,
            }
        )
    return rows


def phase0_channel_reference(phase0_dir: Path) -> pd.DataFrame:
    path = phase0_dir / "channel_summary.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    return df.copy()


def phase0_target_events(phase0_dir: Path) -> pd.DataFrame:
    path = phase0_dir / "target_event_rates_global.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path).copy()


def merge_phase0_exact(input_df: pd.DataFrame, phase0_df: pd.DataFrame) -> pd.DataFrame:
    if phase0_df.empty or "channel" not in phase0_df.columns:
        input_df["phase0_reference_available"] = False
        return input_df

    ref = phase0_df.copy()
    if "kind" in ref.columns:
        ref = ref[ref["kind"] == "input"]
    keep = [column for column in ("channel", "finite_ratio", "mean", "std", "min", "max") if column in ref.columns]
    ref = ref[keep].drop_duplicates("channel")
    rename = {
        "finite_ratio": "phase0_exact_finite_ratio",
        "mean": "phase0_exact_mean",
        "std": "phase0_exact_std",
        "min": "phase0_exact_min",
        "max": "phase0_exact_max",
    }
    ref = ref.rename(columns=rename)
    merged = input_df.merge(ref, on="channel", how="left")
    merged["phase0_reference_available"] = merged.get("phase0_exact_mean", pd.Series(np.nan, index=merged.index)).notna()
    if "phase0_exact_mean" in merged.columns:
        merged["sample_mean_abs_diff_vs_phase0"] = (
            merged["sampled_mean"] - merged["phase0_exact_mean"]
        ).abs()
        merged["sample_mean_diff_in_phase0_std"] = merged["sample_mean_abs_diff_vs_phase0"] / merged["phase0_exact_std"].replace(0, np.nan)
    if "phase0_exact_std" in merged.columns:
        merged["sample_std_abs_diff_vs_phase0"] = (
            merged["sampled_std"] - merged["phase0_exact_std"]
        ).abs()
        merged["sample_std_relative_diff_vs_phase0"] = merged["sample_std_abs_diff_vs_phase0"] / merged["phase0_exact_std"].replace(0, np.nan)
    return merged


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    prepare_output(args.output_dir, args.overwrite)
    logger = setup_logging(args.output_dir)
    rng = np.random.default_rng(args.seed)

    zarr_path = args.zarr_path or (args.dataset_dir / "train.zarr")
    metadata_path = args.dataset_dir / "metadata.json"
    metadata = load_json(metadata_path)

    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr dataset not found: {zarr_path}")

    root = open_zarr_group(zarr_path)
    for required in ("input", "target", "mask", "timestamps"):
        if required not in root:
            raise KeyError(f"Required Zarr array not found: {required}")

    input_ds = root["input"]
    target_ds = root["target"]
    mask_ds = root["mask"]
    timestamps_ds = root["timestamps"]

    n_samples, n_channels, patch_h, patch_w = input_ds.shape
    channels = list(metadata.get("channels") or root.attrs.get("channels") or [])
    if not channels:
        channels = [f"channel_{idx}" for idx in range(n_channels)]
    channels = [str(value) for value in channels]
    if len(channels) != n_channels:
        raise ValueError(
            f"Channel metadata length {len(channels)} does not match input C={n_channels}"
        )

    if target_ds.shape[0] != n_samples or mask_ds.shape[0] != n_samples:
        raise ValueError("Input/target/mask sample dimensions are inconsistent")

    zarr_chunk = getattr(input_ds, "chunks", None)
    inferred_block = int(zarr_chunk[0]) if zarr_chunk else 256
    block_size = args.block_size if args.block_size > 0 else inferred_block
    block_ids = select_block_ids(
        total_patches=n_samples,
        block_size=block_size,
        sample_patches=args.sample_patches,
        strategy=args.sampling_strategy,
        rng=rng,
    )

    per_block_reservoir = max(
        1, math.ceil(args.reservoir_values / max(len(block_ids), 1) * 1.10)
    )

    logger.info("=" * 72)
    logger.info("CORRDIFF PHASE 1 - UNIVARIATE ANALYSIS")
    logger.info("Version              : %s", PHASE1_VERSION)
    logger.info("Zarr                 : %s", zarr_path)
    logger.info("Samples              : %d", n_samples)
    logger.info("Channels             : %d", n_channels)
    logger.info("Patch                : %dx%d", patch_h, patch_w)
    logger.info("Sampling block       : %d patches", block_size)
    logger.info("Selected blocks      : %d", len(block_ids))
    logger.info("Approx sampled patch : %d", min(n_samples, len(block_ids) * block_size))
    logger.info("Reservoir/channel    : %d", args.reservoir_values)
    logger.info("=" * 72)

    input_acc = MultiChannelAccumulator(channels)
    target_stored_acc = ScalarAccumulator()
    target_radar_acc = ScalarAccumulator()
    target_positive_acc = ScalarAccumulator()

    input_sample_parts: list[list[np.ndarray]] = [[] for _ in channels]
    target_stored_parts: list[np.ndarray] = []
    target_radar_parts: list[np.ndarray] = []

    rh_quality_counts: dict[str, dict[str, int]] = {}
    rh_indices = [idx for idx, name in enumerate(channels) if name.startswith("r_")]
    for idx in rh_indices:
        rh_quality_counts[channels[idx]] = {
            "finite": 0,
            "below_0": 0,
            "above_100": 0,
            "within_0_100": 0,
        }

    threshold_counts = {
        label: {
            "patch_count": 0,
            "valid_pixel_count": 0,
            "event_pixel_count": 0,
            "event_patch_count": 0,
        }
        for label, _, _, _ in RADAR_THRESHOLDS
    }

    block_rows: list[dict[str, Any]] = []
    sampled_patch_total = 0

    for block_id in tqdm(block_ids, desc="Phase 1 sampled blocks"):
        start = int(block_id) * block_size
        end = min(start + block_size, n_samples)
        if start >= end:
            continue

        x = np.asarray(input_ds[start:end])
        y = np.asarray(target_ds[start:end])
        mask = np.asarray(mask_ds[start:end]).astype(bool)
        timestamps = np.asarray(timestamps_ds[start:end], dtype=np.int64)
        batch = end - start
        sampled_patch_total += batch

        # Flatten sample/spatial axes while preserving channel axis.
        x_flat = x.transpose(1, 0, 2, 3).reshape(n_channels, -1)
        input_acc.update(x_flat)

        finite_target = mask & np.isfinite(y)
        stored_valid = y[finite_target].astype(np.float64, copy=False)
        target_stored_acc.update(stored_valid)
        radar_valid = np.expm1(stored_valid)
        target_radar_acc.update(radar_valid)
        positive = radar_valid[radar_valid > 0]
        target_positive_acc.update(positive)

        # Relative-humidity diagnostics over the sampled training distribution.
        for idx in rh_indices:
            vals = x_flat[idx]
            finite = vals[np.isfinite(vals)]
            counts = rh_quality_counts[channels[idx]]
            counts["finite"] += int(finite.size)
            counts["below_0"] += int((finite < 0).sum())
            counts["above_100"] += int((finite > 100).sum())
            counts["within_0_100"] += int(((finite >= 0) & (finite <= 100)).sum())

        # Target threshold rates in the sampled blocks.
        radar_batch = np.expm1(y.astype(np.float64, copy=False))
        valid_pixels_per_patch = finite_target.reshape(batch, -1).sum(axis=1)
        for label, condition, threshold, inclusive in RADAR_THRESHOLDS:
            if inclusive:
                event = finite_target & (radar_batch >= threshold)
            else:
                event = finite_target & (radar_batch > threshold)
            event_pixels_per_patch = event.reshape(batch, -1).sum(axis=1)
            bucket = threshold_counts[label]
            bucket["patch_count"] += batch
            bucket["valid_pixel_count"] += int(valid_pixels_per_patch.sum())
            bucket["event_pixel_count"] += int(event_pixels_per_patch.sum())
            bucket["event_patch_count"] += int((event_pixels_per_patch > 0).sum())

        # Bounded value sample for quantiles, histograms and notebook ECDFs.
        spatial_positions = batch * patch_h * patch_w
        take = min(per_block_reservoir, spatial_positions)
        positions = rng.choice(spatial_positions, size=take, replace=False)
        for idx in range(n_channels):
            input_sample_parts[idx].append(x_flat[idx, positions].astype(np.float32, copy=False))

        if stored_valid.size:
            target_take = min(per_block_reservoir, stored_valid.size)
            target_positions = rng.choice(stored_valid.size, size=target_take, replace=False)
            target_stored_parts.append(stored_valid[target_positions].astype(np.float32, copy=False))
            target_radar_parts.append(radar_valid[target_positions].astype(np.float32, copy=False))

        if timestamps.size:
            first_ts = pd.to_datetime(int(timestamps.min()), unit="s", utc=True)
            last_ts = pd.to_datetime(int(timestamps.max()), unit="s", utc=True)
        else:
            first_ts = last_ts = pd.NaT
        block_rows.append(
            {
                "block_id": int(block_id),
                "start_sample": start,
                "end_sample_exclusive": end,
                "patch_count": batch,
                "first_timestamp": first_ts.isoformat() if not pd.isna(first_ts) else None,
                "last_timestamp": last_ts.isoformat() if not pd.isna(last_ts) else None,
                "first_year": int(first_ts.year) if not pd.isna(first_ts) else None,
                "last_year": int(last_ts.year) if not pd.isna(last_ts) else None,
            }
        )

    # Final bounded reservoirs.
    input_samples: dict[str, np.ndarray] = {}
    for idx, channel in enumerate(channels):
        if input_sample_parts[idx]:
            values = np.concatenate(input_sample_parts[idx])
            values = values[np.isfinite(values)]
            if values.size > args.reservoir_values:
                positions = rng.choice(values.size, size=args.reservoir_values, replace=False)
                values = values[positions]
        else:
            values = np.empty(0, dtype=np.float32)
        input_samples[channel] = values.astype(np.float32, copy=False)

    if target_stored_parts:
        target_stored_sample = np.concatenate(target_stored_parts)
        target_radar_sample = np.concatenate(target_radar_parts)
        finite = np.isfinite(target_stored_sample) & np.isfinite(target_radar_sample)
        target_stored_sample = target_stored_sample[finite]
        target_radar_sample = target_radar_sample[finite]
        if target_stored_sample.size > args.reservoir_values:
            positions = rng.choice(
                target_stored_sample.size, size=args.reservoir_values, replace=False
            )
            target_stored_sample = target_stored_sample[positions]
            target_radar_sample = target_radar_sample[positions]
    else:
        target_stored_sample = np.empty(0, dtype=np.float32)
        target_radar_sample = np.empty(0, dtype=np.float32)

    # ------------------------------------------------------------------
    # Input summary, quantiles and histograms.
    # ------------------------------------------------------------------
    input_rows = input_acc.rows()
    quantile_rows: list[dict[str, Any]] = []
    histogram_data: list[dict[str, Any]] = []

    for row in input_rows:
        channel = row["channel"]
        values = input_samples[channel]
        skew, kurt = distribution_shape(values)
        row["sampled_skewness_from_reservoir"] = skew
        row["sampled_excess_kurtosis_from_reservoir"] = kurt
        row["reservoir_value_count"] = int(values.size)
        quantile_rows.extend(
            make_quantile_rows(
                channel,
                values,
                kind="input",
                domain="era5_native",
                unit=row["expected_unit"],
            )
        )
        histogram_data.extend(
            histogram_rows(
                channel,
                values,
                args.hist_bins,
                kind="input",
                domain="era5_native",
                unit=row["expected_unit"],
            )
        )

    input_df = pd.DataFrame(input_rows)
    phase0_channels = phase0_channel_reference(args.phase0_dir)
    input_df = merge_phase0_exact(input_df, phase0_channels)
    input_df.to_parquet(args.output_dir / "input_summary.parquet", index=False)
    pd.DataFrame(quantile_rows).to_parquet(
        args.output_dir / "input_quantiles.parquet", index=False
    )
    pd.DataFrame(histogram_data).to_parquet(
        args.output_dir / "input_histograms.parquet", index=False
    )

    # ------------------------------------------------------------------
    # Relative humidity quality diagnostics.
    # ------------------------------------------------------------------
    rh_rows: list[dict[str, Any]] = []
    for channel, counts in rh_quality_counts.items():
        finite = counts["finite"]
        rh_rows.append(
            {
                "channel": channel,
                "expected_unit": "%",
                "sampled_finite_count": finite,
                "below_0_count": counts["below_0"],
                "below_0_ratio": counts["below_0"] / finite if finite else np.nan,
                "above_100_count": counts["above_100"],
                "above_100_ratio": counts["above_100"] / finite if finite else np.nan,
                "outside_0_100_count": counts["below_0"] + counts["above_100"],
                "outside_0_100_ratio": (
                    (counts["below_0"] + counts["above_100"]) / finite
                    if finite
                    else np.nan
                ),
                "within_0_100_count": counts["within_0_100"],
                "within_0_100_ratio": counts["within_0_100"] / finite if finite else np.nan,
                "diagnostic_scope": "sampled training-patch distribution; values are not clipped",
            }
        )
    pd.DataFrame(rh_rows).to_parquet(
        args.output_dir / "relative_humidity_quality.parquet", index=False
    )

    # ------------------------------------------------------------------
    # Target summaries in stored and reconstructed radar-legend domains.
    # ------------------------------------------------------------------
    target_rows = [
        target_stored_acc.row(
            "target_stored_valid",
            unit="log1p(radar-legend value)",
            domain="stored_log1p",
        ),
        target_radar_acc.row(
            "target_radar_legend_valid",
            unit="radar-legend numeric value (physical unit not declared by builder)",
            domain="reconstructed_expm1",
        ),
        target_positive_acc.row(
            "target_radar_legend_positive",
            unit="radar-legend numeric value (physical unit not declared by builder)",
            domain="reconstructed_expm1_positive_only",
        ),
    ]

    # Add distribution shape from bounded samples.
    stored_skew, stored_kurt = distribution_shape(target_stored_sample)
    radar_skew, radar_kurt = distribution_shape(target_radar_sample)
    positive_radar_sample = target_radar_sample[target_radar_sample > 0]
    pos_skew, pos_kurt = distribution_shape(positive_radar_sample)
    target_rows[0]["sampled_skewness_from_reservoir"] = stored_skew
    target_rows[0]["sampled_excess_kurtosis_from_reservoir"] = stored_kurt
    target_rows[0]["reservoir_value_count"] = int(target_stored_sample.size)
    target_rows[1]["sampled_skewness_from_reservoir"] = radar_skew
    target_rows[1]["sampled_excess_kurtosis_from_reservoir"] = radar_kurt
    target_rows[1]["reservoir_value_count"] = int(target_radar_sample.size)
    target_rows[2]["sampled_skewness_from_reservoir"] = pos_skew
    target_rows[2]["sampled_excess_kurtosis_from_reservoir"] = pos_kurt
    target_rows[2]["reservoir_value_count"] = int(positive_radar_sample.size)

    phase0_target_row: dict[str, Any] = {}
    if not phase0_channels.empty and "kind" in phase0_channels.columns:
        target_ref = phase0_channels[phase0_channels["kind"] == "target"]
        if not target_ref.empty:
            phase0_target_row = target_ref.iloc[0].to_dict()
            target_rows[0]["phase0_reference_available"] = True
            for src, dst in (
                ("mean", "phase0_exact_mean"),
                ("std", "phase0_exact_std"),
                ("min", "phase0_exact_min"),
                ("max", "phase0_exact_max"),
                ("finite_ratio", "phase0_exact_finite_ratio"),
            ):
                if src in phase0_target_row:
                    target_rows[0][dst] = phase0_target_row[src]
        else:
            target_rows[0]["phase0_reference_available"] = False
    else:
        target_rows[0]["phase0_reference_available"] = False

    pd.DataFrame(target_rows).to_parquet(
        args.output_dir / "target_summary.parquet", index=False
    )

    target_quantile_rows: list[dict[str, Any]] = []
    target_quantile_rows.extend(
        make_quantile_rows(
            "target_stored_valid",
            target_stored_sample,
            kind="target",
            domain="stored_log1p",
            unit="log1p(radar-legend value)",
        )
    )
    target_quantile_rows.extend(
        make_quantile_rows(
            "target_radar_legend_valid",
            target_radar_sample,
            kind="target",
            domain="reconstructed_expm1",
            unit="radar-legend numeric value; physical unit unresolved",
        )
    )
    target_quantile_rows.extend(
        make_quantile_rows(
            "target_radar_legend_positive",
            positive_radar_sample,
            kind="target",
            domain="reconstructed_expm1_positive_only",
            unit="radar-legend numeric value; physical unit unresolved",
        )
    )
    pd.DataFrame(target_quantile_rows).to_parquet(
        args.output_dir / "target_quantiles.parquet", index=False
    )

    target_hist_rows: list[dict[str, Any]] = []
    target_hist_rows.extend(
        histogram_rows(
            "target_stored_valid",
            target_stored_sample,
            args.hist_bins,
            kind="target",
            domain="stored_log1p",
            unit="log1p(radar-legend value)",
            subset="all_valid",
        )
    )
    target_hist_rows.extend(
        histogram_rows(
            "target_radar_legend_valid",
            target_radar_sample,
            args.hist_bins,
            kind="target",
            domain="reconstructed_expm1",
            unit="radar-legend numeric value; physical unit unresolved",
            subset="all_valid",
        )
    )
    target_hist_rows.extend(
        histogram_rows(
            "target_radar_legend_positive",
            positive_radar_sample,
            args.hist_bins,
            kind="target",
            domain="reconstructed_expm1_positive_only",
            unit="radar-legend numeric value; physical unit unresolved",
            subset="positive_only",
        )
    )
    pd.DataFrame(target_hist_rows).to_parquet(
        args.output_dir / "target_histograms.parquet", index=False
    )

    # ------------------------------------------------------------------
    # Target threshold rates in sampled blocks + exact Phase 0 reference.
    # ------------------------------------------------------------------
    sampled_threshold_rows: list[dict[str, Any]] = []
    threshold_meta = {label: (condition, threshold) for label, condition, threshold, _ in RADAR_THRESHOLDS}
    for label, counts in threshold_counts.items():
        condition, threshold = threshold_meta[label]
        valid_pixels = counts["valid_pixel_count"]
        patches = counts["patch_count"]
        sampled_threshold_rows.append(
            {
                "threshold": label,
                "condition": condition,
                "legend_value_threshold": threshold,
                "patch_count": int(patches),
                "valid_pixel_count": int(valid_pixels),
                "event_pixel_count": int(counts["event_pixel_count"]),
                "event_pixel_ratio": (
                    counts["event_pixel_count"] / valid_pixels if valid_pixels else np.nan
                ),
                "event_patch_count": int(counts["event_patch_count"]),
                "event_patch_ratio": (
                    counts["event_patch_count"] / patches if patches else np.nan
                ),
                "scope": "sampled training-patch distribution",
            }
        )
    sampled_threshold_df = pd.DataFrame(sampled_threshold_rows).sort_values(
        "legend_value_threshold"
    )
    sampled_threshold_df.to_parquet(
        args.output_dir / "target_threshold_rates_sampled.parquet", index=False
    )

    phase0_events = phase0_target_events(args.phase0_dir)
    if not phase0_events.empty:
        phase0_events.to_parquet(
            args.output_dir / "target_threshold_rates_reference.parquet", index=False
        )

    # ------------------------------------------------------------------
    # Store bounded samples for notebook visualizations.
    # ------------------------------------------------------------------
    sample_payload: dict[str, np.ndarray] = {
        "target_stored_valid": target_stored_sample.astype(np.float32, copy=False),
        "target_radar_legend_valid": target_radar_sample.astype(np.float32, copy=False),
        "target_radar_legend_positive": positive_radar_sample.astype(np.float32, copy=False),
    }
    for channel, values in input_samples.items():
        sample_payload[f"input__{sanitized_key(channel)}"] = values.astype(np.float32, copy=False)
    np.savez_compressed(args.output_dir / "univariate_samples.npz", **sample_payload)

    # Block metadata.
    pd.DataFrame(block_rows).to_parquet(
        args.output_dir / "sampling_blocks.parquet", index=False
    )

    # ------------------------------------------------------------------
    # Compact JSON synthesis + quality flags.
    # ------------------------------------------------------------------
    warnings: list[dict[str, str]] = []
    representativeness: dict[str, Any] = {"phase0_reference_available": False}
    if "sample_mean_diff_in_phase0_std" in input_df.columns:
        available = input_df[input_df["phase0_reference_available"]].copy()
        if not available.empty:
            max_mean_z = float(available["sample_mean_diff_in_phase0_std"].max())
            max_std_rel = float(available["sample_std_relative_diff_vs_phase0"].max())
            representativeness = {
                "phase0_reference_available": True,
                "max_abs_sample_mean_difference_in_phase0_std": max_mean_z,
                "max_sample_std_relative_difference": max_std_rel,
            }
            if max_mean_z > 0.10:
                warnings.append(
                    {
                        "code": "sample_mean_shift_vs_phase0",
                        "detail": (
                            "At least one sampled channel mean differs from the Phase 0 exact mean "
                            f"by more than 0.10 standard deviations (max={max_mean_z:.4f})."
                        ),
                    }
                )
            if max_std_rel > 0.10:
                warnings.append(
                    {
                        "code": "sample_std_shift_vs_phase0",
                        "detail": (
                            "At least one sampled channel std differs from the Phase 0 exact std "
                            f"by more than 10% (max={max_std_rel:.4f})."
                        ),
                    }
                )

    rh_summary = {}
    if rh_rows:
        rh_summary = {
            row["channel"]: {
                "below_0_ratio": row["below_0_ratio"],
                "above_100_ratio": row["above_100_ratio"],
                "outside_0_100_ratio": row["outside_0_100_ratio"],
            }
            for row in rh_rows
        }
        for row in rh_rows:
            if row["outside_0_100_ratio"] > 0:
                warnings.append(
                    {
                        "code": f"{row['channel']}_outside_0_100",
                        "detail": (
                            f"Sampled {row['channel']} contains {row['outside_0_100_ratio']:.6%} "
                            "of values outside [0,100]. Values were measured, not clipped."
                        ),
                    }
                )

    event_reference = phase0_events if not phase0_events.empty else sampled_threshold_df
    target_sparsity: dict[str, Any] = {}
    if not event_reference.empty and "threshold" in event_reference.columns:
        row0 = event_reference[event_reference["threshold"] == "gt_0"]
        if not row0.empty:
            row0 = row0.iloc[0]
            target_sparsity = {
                "event_pixel_ratio_gt_0": safe_float(row0.get("event_pixel_ratio")),
                "event_patch_ratio_gt_0": safe_float(row0.get("event_patch_ratio")),
                "dry_valid_pixel_ratio": (
                    1.0 - float(row0["event_pixel_ratio"])
                    if pd.notna(row0.get("event_pixel_ratio"))
                    else None
                ),
                "dry_patch_ratio": (
                    1.0 - float(row0["event_patch_ratio"])
                    if pd.notna(row0.get("event_patch_ratio"))
                    else None
                ),
                "source": (
                    "phase0 exact full scan"
                    if not phase0_events.empty
                    else "phase1 sampled blocks"
                ),
            }

    elapsed = time.perf_counter() - started
    summary = {
        "phase": 1,
        "version": PHASE1_VERSION,
        "dataset": {
            "zarr_path": str(zarr_path),
            "num_patches": int(n_samples),
            "channels": channels,
            "input_shape": list(input_ds.shape),
            "target_shape": list(target_ds.shape),
            "mask_shape": list(mask_ds.shape),
            "target_transform": metadata.get(
                "target_transform", root.attrs.get("target_transform", "unknown")
            ),
        },
        "sampling": {
            "strategy": args.sampling_strategy,
            "seed": args.seed,
            "zarr_chunk_first_dimension": inferred_block,
            "block_size": block_size,
            "selected_block_count": int(len(block_ids)),
            "sampled_patch_count": int(sampled_patch_total),
            "requested_sample_patches": int(args.sample_patches),
            "reservoir_values_per_channel_limit": int(args.reservoir_values),
            "representativeness_vs_phase0": representativeness,
        },
        "radar_target": {
            "stored_domain": "log1p after invalid fill and clip(min=0)",
            "reconstructed_domain": "expm1(stored target), i.e. post-clip radar-legend numeric value",
            "physical_unit": "not declared by builder; do not label dBZ until source documentation confirms it",
            "target_sparsity": target_sparsity,
        },
        "relative_humidity_quality": rh_summary,
        "warnings": warnings,
        "runtime_seconds": elapsed,
    }
    with (args.output_dir / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    logger.info("Phase 1 completed in %.1f s", elapsed)
    logger.info("Sampled patches       : %d", sampled_patch_total)
    logger.info("Output                : %s", args.output_dir)
    if warnings:
        logger.info("Warnings              : %d", len(warnings))
        for warning in warnings:
            logger.info("  - %s: %s", warning["code"], warning["detail"])
    else:
        logger.info("Warnings              : none")


if __name__ == "__main__":
    main()
