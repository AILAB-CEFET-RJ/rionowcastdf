#!/usr/bin/env python3
"""Phase 0 audit for the CorrDiff ERA5 -> Radar dataset.

This script performs a structural, temporal, spatial and numerical quality audit
of the dataset produced by the CorrDiff builder. It is designed for large Zarr
stores and scans the arrays in bounded batches instead of loading the full
2011-2024 dataset into RAM.

Expected builder layout
-----------------------
<dataset_dir>/
    train.zarr/
        input        (N, C, P, P) float32
        target       (N, 1, P, P) float32
        mask         (N, 1, P, P) float32
        timestamps   (N,)         int64, Unix seconds in the current builder
        patch_row    (N,)         int16
        patch_col    (N,)         int16
    metadata.json
    normalization.npz
    stats/
        input_mean.npy
        input_std.npy
        target_mean.npy
        target_std.npy
        channel_statistics.csv

Outputs
-------
<output_dir>/
    dataset_summary.json
    channel_summary.parquet
    missing_data.parquet
    temporal_coverage.parquet
    yearly_coverage.parquet
    monthly_coverage.parquet
    hourly_coverage.parquet
    patch_position_counts.parquet
    expected_patch_spatial_coverage.npy
    actual_patch_spatial_coverage.npy
    radar_legend_mapping.parquet
    radar_cache_sample_statistics.parquet       # when --radar-cache-dir is supplied
    radar_target_mapping_validation.parquet     # when cache + patch coordinates exist
    radar_target_semantics.json
    audit_warnings.json
    audit.log

The audit intentionally keeps interpretation separate from measurement. The
current builder implementation converts radar PNG RGB colors into numerical
values using the configured radar legend, stores those values as float32 NPY
cache grids, and later stores ``log1p(clip(cache, 0, +inf))`` in the Zarr target.
The builder calls the cache field ``reflectivity`` but does not declare a physical
unit such as dBZ. This audit verifies the cache -> Zarr transformation directly
when ``--radar-cache-dir`` is provided, while keeping the physical-unit question
explicitly unresolved until the original radar product legend is documented.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import socket
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
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


AUDIT_VERSION = "phase0-audit-v2-radar-semantics"
REQUIRED_ARRAYS = ("input", "target", "mask", "timestamps")
OPTIONAL_ARRAYS = ("patch_row", "patch_col")
QUANTILE_NAMES = ("p01", "p05", "p25", "p50", "p75", "p95", "p99")
QUANTILE_VALUES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)

# Radar semantics copied from the cache-generation implementation audited for
# this dataset builder. These constants are not a claim about the physical unit;
# the builder names them reflectivity values but does not declare "dBZ".
BUILDER_RADAR_LEGEND_VALUES = np.asarray(
    [50, 45, 40, 35, 30, 25, 20, 0], dtype=np.float32
)
BUILDER_RADAR_LEGEND_COLORS = np.asarray(
    [
        (197, 0, 197),
        (227, 6, 5),
        (255, 112, 0),
        (195, 230, 0),
        (4, 85, 4),
        (19, 122, 19),
        (0, 167, 12),
        (0, 0, 0),
    ],
    dtype=np.float32,
)
TARGET_MAPPING_ATOL = 2e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the CorrDiff Zarr dataset produced by dataset_builder."
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
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs/00_quality"),
        help="Directory where audit outputs are written.",
    )
    parser.add_argument(
        "--batch-samples",
        type=int,
        default=512,
        help="Number of samples read from Zarr per numerical-audit batch.",
    )
    parser.add_argument(
        "--quantile-sample-values",
        type=int,
        default=200_000,
        help=(
            "Approximate number of finite pixel values sampled per channel for "
            "quantiles. Mean/std/min/max/non-finite counts remain exact in full scan."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260913,
        help="Random seed used only for quantile sampling.",
    )
    parser.add_argument(
        "--radar-cache-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing YYYYMMDD_HH_MM.npy radar cache files. "
            "When supplied, Phase 0 verifies the PNG-legend cache semantics and "
            "the exact cache -> Zarr target mapping. You may also set the "
            "CORRDIFF_RADAR_CACHE_DIR environment variable."
        ),
    )
    parser.add_argument(
        "--radar-cache-file-sample",
        type=int,
        default=128,
        help=(
            "Number of distinct dataset timestamps whose radar cache grids are "
            "sampled for cache-value statistics."
        ),
    )
    parser.add_argument(
        "--radar-target-validation-samples",
        type=int,
        default=256,
        help=(
            "Number of Zarr samples compared directly against their source radar "
            "cache patch to validate mask, clip and log1p behavior."
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Skip the expensive full pixel scan. Structural/temporal/spatial checks "
            "still run and builder normalization statistics are imported when present."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing files already present in output-dir.",
    )
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    return value


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, ensure_ascii=False)


def normalize_timestamp(ts: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(ts)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def decode_unix_timestamps(values: np.ndarray) -> tuple[pd.DatetimeIndex, str]:
    """Decode integer timestamps, auto-detecting the most likely Unix unit."""
    values = np.asarray(values)
    if values.size == 0:
        return pd.DatetimeIndex([]), "unknown"

    if np.issubdtype(values.dtype, np.datetime64):
        return pd.DatetimeIndex(values), "datetime64"

    finite = values[np.isfinite(values)] if np.issubdtype(values.dtype, np.floating) else values
    if finite.size == 0:
        return pd.DatetimeIndex([]), "unknown"

    magnitude = float(np.median(np.abs(finite.astype(np.float64))))
    if magnitude >= 1e17:
        unit = "ns"
    elif magnitude >= 1e14:
        unit = "us"
    elif magnitude >= 1e11:
        unit = "ms"
    else:
        unit = "s"

    decoded = pd.to_datetime(values.astype(np.int64), unit=unit, utc=True)
    return pd.DatetimeIndex(decoded).tz_convert(None), unit


def iter_slices(total: int, batch_size: int) -> Iterable[slice]:
    for start in range(0, total, batch_size):
        yield slice(start, min(start + batch_size, total))


def array_description(array: Any) -> dict[str, Any]:
    compressor = getattr(array, "compressor", None)
    return {
        "shape": list(array.shape),
        "chunks": list(array.chunks) if getattr(array, "chunks", None) else None,
        "dtype": str(array.dtype),
        "compressor": repr(compressor),
    }


@dataclass
class NumericAccumulator:
    names: list[str]
    total_count: np.ndarray = field(init=False)
    finite_count: np.ndarray = field(init=False)
    nan_count: np.ndarray = field(init=False)
    posinf_count: np.ndarray = field(init=False)
    neginf_count: np.ndarray = field(init=False)
    sums: np.ndarray = field(init=False)
    sums_sq: np.ndarray = field(init=False)
    minima: np.ndarray = field(init=False)
    maxima: np.ndarray = field(init=False)
    quantile_samples: list[list[np.ndarray]] = field(init=False)

    def __post_init__(self) -> None:
        size = len(self.names)
        self.total_count = np.zeros(size, dtype=np.int64)
        self.finite_count = np.zeros(size, dtype=np.int64)
        self.nan_count = np.zeros(size, dtype=np.int64)
        self.posinf_count = np.zeros(size, dtype=np.int64)
        self.neginf_count = np.zeros(size, dtype=np.int64)
        self.sums = np.zeros(size, dtype=np.float64)
        self.sums_sq = np.zeros(size, dtype=np.float64)
        self.minima = np.full(size, np.inf, dtype=np.float64)
        self.maxima = np.full(size, -np.inf, dtype=np.float64)
        self.quantile_samples = [[] for _ in range(size)]

    def update_channels(
        self,
        values: np.ndarray,
        rng: np.random.Generator,
        per_channel_sample: int,
    ) -> None:
        """Update from values with shape (B, C, H, W)."""
        if values.ndim != 4:
            raise ValueError(f"Expected 4-D array, got {values.shape}")
        if values.shape[1] != len(self.names):
            raise ValueError(
                f"Accumulator has {len(self.names)} names but values have {values.shape[1]} channels"
            )

        axes = (0, 2, 3)
        finite = np.isfinite(values)
        self.total_count += np.prod([values.shape[0], values.shape[2], values.shape[3]])
        self.finite_count += finite.sum(axis=axes, dtype=np.int64)
        self.nan_count += np.isnan(values).sum(axis=axes, dtype=np.int64)
        self.posinf_count += np.isposinf(values).sum(axis=axes, dtype=np.int64)
        self.neginf_count += np.isneginf(values).sum(axis=axes, dtype=np.int64)

        safe = np.where(finite, values, 0.0).astype(np.float64, copy=False)
        self.sums += safe.sum(axis=axes, dtype=np.float64)
        self.sums_sq += np.square(safe).sum(axis=axes, dtype=np.float64)

        mins = np.where(finite, values, np.inf).min(axis=axes)
        maxs = np.where(finite, values, -np.inf).max(axis=axes)
        self.minima = np.minimum(self.minima, mins)
        self.maxima = np.maximum(self.maxima, maxs)

        if per_channel_sample > 0:
            for channel_index in range(values.shape[1]):
                finite_values = values[:, channel_index][finite[:, channel_index]]
                if finite_values.size == 0:
                    continue
                take = min(per_channel_sample, finite_values.size)
                if take == finite_values.size:
                    sampled = finite_values.astype(np.float32, copy=False)
                else:
                    indices = rng.choice(finite_values.size, size=take, replace=False)
                    sampled = finite_values[indices].astype(np.float32, copy=False)
                self.quantile_samples[channel_index].append(sampled)

    def update_single(
        self,
        values: np.ndarray,
        valid: np.ndarray | None,
        rng: np.random.Generator,
        sample_count: int,
    ) -> None:
        """Update single named variable from an arbitrary-shaped tensor."""
        if len(self.names) != 1:
            raise ValueError("update_single requires a one-variable accumulator")
        values = np.asarray(values)
        if valid is None:
            selected = values.ravel()
        else:
            selected = values[np.asarray(valid, dtype=bool)]

        self.total_count[0] += selected.size
        finite = np.isfinite(selected)
        self.finite_count[0] += int(finite.sum())
        self.nan_count[0] += int(np.isnan(selected).sum())
        self.posinf_count[0] += int(np.isposinf(selected).sum())
        self.neginf_count[0] += int(np.isneginf(selected).sum())

        finite_values = selected[finite]
        if finite_values.size:
            values64 = finite_values.astype(np.float64, copy=False)
            self.sums[0] += values64.sum(dtype=np.float64)
            self.sums_sq[0] += np.square(values64).sum(dtype=np.float64)
            self.minima[0] = min(self.minima[0], float(finite_values.min()))
            self.maxima[0] = max(self.maxima[0], float(finite_values.max()))

            if sample_count > 0:
                take = min(sample_count, finite_values.size)
                if take == finite_values.size:
                    sampled = finite_values.astype(np.float32, copy=False)
                else:
                    idx = rng.choice(finite_values.size, size=take, replace=False)
                    sampled = finite_values[idx].astype(np.float32, copy=False)
                self.quantile_samples[0].append(sampled)

    def to_rows(
        self,
        kind: str,
        quantile_cap: int,
        rng: np.random.Generator,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for index, name in enumerate(self.names):
            finite_count = int(self.finite_count[index])
            if finite_count:
                mean = self.sums[index] / finite_count
                variance = max(self.sums_sq[index] / finite_count - mean * mean, 0.0)
                std = math.sqrt(variance)
                minimum = float(self.minima[index])
                maximum = float(self.maxima[index])
            else:
                mean = std = minimum = maximum = float("nan")

            samples = (
                np.concatenate(self.quantile_samples[index])
                if self.quantile_samples[index]
                else np.empty(0, dtype=np.float32)
            )
            if samples.size > quantile_cap > 0:
                choose = rng.choice(samples.size, size=quantile_cap, replace=False)
                samples = samples[choose]

            quantiles: dict[str, float] = {}
            if samples.size:
                values = np.quantile(samples.astype(np.float64), QUANTILE_VALUES)
                quantiles = {
                    name_: float(value) for name_, value in zip(QUANTILE_NAMES, values)
                }
            else:
                quantiles = {name_: float("nan") for name_ in QUANTILE_NAMES}

            total = int(self.total_count[index])
            rows.append(
                {
                    "kind": kind,
                    "channel_index": index,
                    "channel": name,
                    "total_count": total,
                    "finite_count": finite_count,
                    "nan_count": int(self.nan_count[index]),
                    "posinf_count": int(self.posinf_count[index]),
                    "neginf_count": int(self.neginf_count[index]),
                    "finite_ratio": finite_count / total if total else float("nan"),
                    "mean": float(mean),
                    "std": float(std),
                    "min": minimum,
                    "max": maximum,
                    **quantiles,
                    "quantile_sample_count": int(samples.size),
                    "quantiles_are_approximate": True,
                }
            )
        return rows


def configure_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("corrdiff_phase0_audit")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(output_dir / "audit.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def expected_patch_positions(
    grid_shape: tuple[int, int] | None,
    patch_size: int | None,
    stride: int | None,
) -> list[tuple[int, int]]:
    if not grid_shape or not patch_size or not stride:
        return []
    height, width = grid_shape
    return [
        (row, col)
        for row in range(0, height - patch_size + 1, stride)
        for col in range(0, width - patch_size + 1, stride)
    ]


def build_coverage_map(
    grid_shape: tuple[int, int],
    patch_size: int,
    positions: Iterable[tuple[int, int]],
) -> np.ndarray:
    coverage = np.zeros(grid_shape, dtype=np.int32)
    height, width = grid_shape
    for row, col in positions:
        if row < 0 or col < 0 or row + patch_size > height or col + patch_size > width:
            continue
        coverage[row : row + patch_size, col : col + patch_size] += 1
    return coverage


def safe_fraction(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def radar_legend_dataframe() -> pd.DataFrame:
    """Return the RGB -> numerical-value legend embedded in the builder."""
    return pd.DataFrame(
        {
            "legend_index": np.arange(len(BUILDER_RADAR_LEGEND_VALUES), dtype=np.int16),
            "value": BUILDER_RADAR_LEGEND_VALUES.astype(np.float32),
            "r": BUILDER_RADAR_LEGEND_COLORS[:, 0].astype(np.int16),
            "g": BUILDER_RADAR_LEGEND_COLORS[:, 1].astype(np.int16),
            "b": BUILDER_RADAR_LEGEND_COLORS[:, 2].astype(np.int16),
        }
    )


def cache_path_for_timestamp(cache_dir: Path, timestamp: pd.Timestamp) -> Path:
    return cache_dir / f"{timestamp:%Y%m%d_%H_%M}.npy"


def decode_one_timestamp(raw_value: Any, unit: str) -> pd.Timestamp:
    if unit == "datetime64":
        return normalize_timestamp(raw_value)
    return pd.Timestamp(pd.to_datetime(int(raw_value), unit=unit, utc=True)).tz_convert(None)


def _finite_quantiles(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {name: None for name in QUANTILE_NAMES}
    qs = np.quantile(values, QUANTILE_VALUES)
    return {name: float(value) for name, value in zip(QUANTILE_NAMES, qs)}


def audit_radar_cache_and_target_mapping(
    *,
    cache_dir: Path | None,
    arrays: dict[str, Any],
    timestamp_raw: np.ndarray,
    timestamp_unit: str,
    unique_timestamps: pd.DatetimeIndex,
    patch_rows: np.ndarray | None,
    patch_cols: np.ndarray | None,
    patch_size: int,
    radar_grid_shape: tuple[int, int] | None,
    file_sample_count: int,
    target_validation_samples: int,
    seed: int,
    output_dir: Path,
    warn: Any,
    note: Any,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Audit builder radar semantics and, when possible, verify cache -> target."""

    legend_df = radar_legend_dataframe()
    legend_df.to_parquet(output_dir / "radar_legend_mapping.parquet", index=False)

    semantics: dict[str, Any] = {
        "builder_semantics": {
            "source_format": "PNG converted to RGB",
            "rgb_to_numeric_method": (
                "two nearest configured legend colors in RGB Euclidean distance, "
                "followed by linear interpolation between their numerical legend values"
            ),
            "legend_values": BUILDER_RADAR_LEGEND_VALUES.tolist(),
            "legend_colors_rgb": BUILDER_RADAR_LEGEND_COLORS.astype(int).tolist(),
            "cache_array_dtype": "float32",
            "cache_spatial_mapping": (
                "precomputed geographic-grid coordinates are converted to integer source-image "
                "pixel indices; the numerical field is sampled as reflectivity[py, px]"
            ),
            "zarr_target_transform": "nan_to_num(nan/+-inf -> 0) -> clip(min=0) -> log1p -> float32",
            "zarr_mask_definition": "isfinite(raw cache patch), before clip/log1p",
            "inverse_of_log1p_stage": "expm1(target) recovers the clipped non-negative cache value, not negative raw cache values",
            "physical_field_name_in_builder": "reflectivity",
            "physical_unit_declared_in_builder": None,
            "dbz_confirmed_by_builder_alone": False,
        },
        "cache_validation": {
            "status": "SKIPPED",
            "reason": "radar cache directory not supplied",
        },
        "target_mapping_validation": {
            "status": "SKIPPED",
            "reason": "radar cache directory not supplied",
        },
    }

    if cache_dir is None:
        note(
            "radar_cache_not_checked",
            "Radar cache directory was not supplied. Builder semantics are documented, but direct cache -> Zarr verification was skipped.",
        )
        write_json(output_dir / "radar_target_semantics.json", semantics)
        return semantics

    cache_dir = cache_dir.expanduser().resolve()
    semantics["cache_dir"] = str(cache_dir)
    if not cache_dir.is_dir():
        warn("radar_cache_dir_missing", f"Radar cache directory does not exist: {cache_dir}")
        semantics["cache_validation"] = {
            "status": "FAILED",
            "reason": "radar cache directory does not exist",
        }
        semantics["target_mapping_validation"] = {
            "status": "FAILED",
            "reason": "radar cache directory does not exist",
        }
        write_json(output_dir / "radar_target_semantics.json", semantics)
        return semantics

    rng = np.random.default_rng(seed)

    # --------------------------------------------------------------
    # Cache-grid sample audit using distinct timestamps represented in Zarr.
    # This avoids enumerating a potentially multi-million-file 2-minute cache.
    # --------------------------------------------------------------
    cache_rows: list[dict[str, Any]] = []
    aggregate_finite_chunks: list[np.ndarray] = []
    file_sample_count = max(0, int(file_sample_count))
    n_unique = len(unique_timestamps)
    if file_sample_count > 0 and n_unique > 0:
        chosen = rng.choice(n_unique, size=min(file_sample_count, n_unique), replace=False)
        chosen_timestamps = unique_timestamps[np.sort(chosen)]
    else:
        chosen_timestamps = pd.DatetimeIndex([])

    sampled_negative = 0
    sampled_above_legend = 0
    sampled_below_legend = 0
    sampled_nonfinite = 0
    sampled_total = 0
    found_files = 0
    missing_files = 0
    shape_mismatch_files = 0
    legend_min = float(np.min(BUILDER_RADAR_LEGEND_VALUES))
    legend_max = float(np.max(BUILDER_RADAR_LEGEND_VALUES))

    for timestamp in chosen_timestamps:
        cache_file = cache_path_for_timestamp(cache_dir, timestamp)
        if not cache_file.exists():
            missing_files += 1
            cache_rows.append(
                {
                    "timestamp": timestamp,
                    "cache_file": str(cache_file),
                    "exists": False,
                }
            )
            continue
        try:
            grid = np.load(cache_file, allow_pickle=False)
        except Exception as error:
            cache_rows.append(
                {
                    "timestamp": timestamp,
                    "cache_file": str(cache_file),
                    "exists": True,
                    "read_error": repr(error),
                }
            )
            continue

        found_files += 1
        grid = np.asarray(grid)
        finite = np.isfinite(grid)
        finite_values = grid[finite].astype(np.float64, copy=False)
        sampled_total += int(grid.size)
        sampled_nonfinite += int((~finite).sum())
        if finite_values.size:
            sampled_negative += int((finite_values < 0).sum())
            sampled_above_legend += int((finite_values > legend_max + 1e-6).sum())
            sampled_below_legend += int((finite_values < legend_min - 1e-6).sum())
            aggregate_finite_chunks.append(finite_values)
        if radar_grid_shape is not None and tuple(grid.shape) != tuple(radar_grid_shape):
            shape_mismatch_files += 1

        q = _finite_quantiles(finite_values)
        cache_rows.append(
            {
                "timestamp": timestamp,
                "cache_file": str(cache_file),
                "exists": True,
                "shape": str(tuple(int(v) for v in grid.shape)),
                "dtype": str(grid.dtype),
                "total_count": int(grid.size),
                "finite_count": int(finite.sum()),
                "finite_ratio": safe_fraction(int(finite.sum()), int(grid.size)),
                "negative_count": int((finite_values < 0).sum()) if finite_values.size else 0,
                "below_legend_min_count": int((finite_values < legend_min - 1e-6).sum()) if finite_values.size else 0,
                "above_legend_max_count": int((finite_values > legend_max + 1e-6).sum()) if finite_values.size else 0,
                "min": float(np.min(finite_values)) if finite_values.size else np.nan,
                "max": float(np.max(finite_values)) if finite_values.size else np.nan,
                "mean": float(np.mean(finite_values)) if finite_values.size else np.nan,
                "std": float(np.std(finite_values)) if finite_values.size else np.nan,
                **q,
            }
        )

    cache_df = pd.DataFrame(cache_rows)
    cache_df.to_parquet(output_dir / "radar_cache_sample_statistics.parquet", index=False)

    aggregate_values = (
        np.concatenate(aggregate_finite_chunks) if aggregate_finite_chunks else np.asarray([], dtype=np.float64)
    )
    aggregate_q = _finite_quantiles(aggregate_values)
    cache_validation = {
        "status": "PASS" if found_files > 0 else "WARN",
        "sampling_basis": "distinct timestamps present in Zarr; no full cache-directory enumeration",
        "requested_files": int(len(chosen_timestamps)),
        "found_files": int(found_files),
        "missing_files": int(missing_files),
        "shape_mismatch_files": int(shape_mismatch_files),
        "sampled_pixel_count": int(sampled_total),
        "sampled_nonfinite_count": int(sampled_nonfinite),
        "sampled_finite_ratio": safe_fraction(sampled_total - sampled_nonfinite, sampled_total),
        "sampled_negative_value_count": int(sampled_negative),
        "sampled_below_legend_min_count": int(sampled_below_legend),
        "sampled_above_legend_max_count": int(sampled_above_legend),
        "sampled_min": float(np.min(aggregate_values)) if aggregate_values.size else None,
        "sampled_max": float(np.max(aggregate_values)) if aggregate_values.size else None,
        "sampled_mean": float(np.mean(aggregate_values)) if aggregate_values.size else None,
        "sampled_std": float(np.std(aggregate_values)) if aggregate_values.size else None,
        **{f"sampled_{k}": v for k, v in aggregate_q.items()},
    }
    semantics["cache_validation"] = cache_validation

    if shape_mismatch_files > 0:
        warn(
            "radar_cache_shape_mismatch",
            f"Found {shape_mismatch_files} sampled radar cache files whose shape differs from metadata radar_grid_shape={radar_grid_shape}.",
        )
    if sampled_negative > 0:
        note(
            "radar_cache_negative_values",
            f"Found {sampled_negative} negative values in sampled radar cache pixels. The builder clips them to zero before log1p, so their original magnitude is not recoverable from target.",
        )
    if sampled_below_legend > 0 or sampled_above_legend > 0:
        warn(
            "radar_cache_values_outside_legend_range",
            (
                f"Sampled cache contains {sampled_below_legend} values below {legend_min} and "
                f"{sampled_above_legend} values above {legend_max}. The RGB interpolation code does not explicitly clamp alpha to [0,1], so investigate source-image colors."
            ),
        )

    # --------------------------------------------------------------
    # Direct cache -> Zarr patch validation.
    # --------------------------------------------------------------
    validation_rows: list[dict[str, Any]] = []
    n_samples = int(arrays["target"].shape[0])
    if patch_rows is None or patch_cols is None:
        semantics["target_mapping_validation"] = {
            "status": "SKIPPED",
            "reason": "patch_row/patch_col are absent",
        }
        note(
            "radar_target_mapping_not_checked",
            "Direct cache -> Zarr target validation requires patch_row and patch_col arrays.",
        )
    elif target_validation_samples <= 0 or n_samples == 0:
        semantics["target_mapping_validation"] = {
            "status": "SKIPPED",
            "reason": "validation sample count is zero or dataset is empty",
        }
    else:
        target_validation_samples = min(int(target_validation_samples), n_samples)
        # Oversample candidate indices so missing cache files do not immediately reduce coverage.
        candidate_count = min(n_samples, max(target_validation_samples * 4, target_validation_samples))
        candidate_indices = rng.choice(n_samples, size=candidate_count, replace=False)
        compared = 0
        missing_cache_for_sample = 0

        for sample_index in candidate_indices:
            if compared >= target_validation_samples:
                break
            timestamp = decode_one_timestamp(timestamp_raw[int(sample_index)], timestamp_unit)
            cache_file = cache_path_for_timestamp(cache_dir, timestamp)
            if not cache_file.exists():
                missing_cache_for_sample += 1
                continue
            try:
                cache_grid = np.load(cache_file, allow_pickle=False)
            except Exception:
                continue

            row = int(patch_rows[int(sample_index)])
            col = int(patch_cols[int(sample_index)])
            raw_patch = np.asarray(
                cache_grid[row : row + patch_size, col : col + patch_size],
                dtype=np.float32,
            )
            stored_target = np.asarray(arrays["target"][int(sample_index), 0], dtype=np.float32)
            stored_mask = np.asarray(arrays["mask"][int(sample_index), 0], dtype=np.float32)
            if raw_patch.shape != stored_target.shape:
                validation_rows.append(
                    {
                        "sample_index": int(sample_index),
                        "timestamp": timestamp,
                        "patch_row": row,
                        "patch_col": col,
                        "cache_file": str(cache_file),
                        "shape_match": False,
                        "raw_patch_shape": str(raw_patch.shape),
                        "stored_target_shape": str(stored_target.shape),
                    }
                )
                compared += 1
                continue

            expected_mask = np.isfinite(raw_patch).astype(np.float32)
            filled = np.nan_to_num(raw_patch, nan=0.0, posinf=0.0, neginf=0.0)
            clipped = np.clip(filled, 0.0, None).astype(np.float32, copy=False)
            expected_target = np.log1p(clipped).astype(np.float32, copy=False)
            inverse_target = np.expm1(stored_target).astype(np.float32, copy=False)

            target_abs_diff = np.abs(stored_target.astype(np.float64) - expected_target.astype(np.float64))
            inverse_abs_diff = np.abs(inverse_target.astype(np.float64) - clipped.astype(np.float64))
            mask_mismatch = int((~np.isclose(stored_mask, expected_mask, atol=0.0, rtol=0.0)).sum())
            finite_raw = raw_patch[np.isfinite(raw_patch)]

            validation_rows.append(
                {
                    "sample_index": int(sample_index),
                    "timestamp": timestamp,
                    "patch_row": row,
                    "patch_col": col,
                    "cache_file": str(cache_file),
                    "shape_match": True,
                    "raw_cache_min": float(np.min(finite_raw)) if finite_raw.size else np.nan,
                    "raw_cache_max": float(np.max(finite_raw)) if finite_raw.size else np.nan,
                    "raw_cache_negative_count": int((finite_raw < 0).sum()) if finite_raw.size else 0,
                    "raw_cache_nonfinite_count": int((~np.isfinite(raw_patch)).sum()),
                    "stored_target_min": float(np.min(stored_target)),
                    "stored_target_max": float(np.max(stored_target)),
                    "max_abs_diff_stored_vs_expected_log1p": float(np.max(target_abs_diff)),
                    "mean_abs_diff_stored_vs_expected_log1p": float(np.mean(target_abs_diff)),
                    "max_abs_diff_expm1_vs_clipped_cache": float(np.max(inverse_abs_diff)),
                    "mean_abs_diff_expm1_vs_clipped_cache": float(np.mean(inverse_abs_diff)),
                    "mask_mismatch_count": mask_mismatch,
                    "mapping_matches_within_tolerance": bool(
                        np.max(target_abs_diff) <= TARGET_MAPPING_ATOL and mask_mismatch == 0
                    ),
                }
            )
            compared += 1

        validation_df = pd.DataFrame(validation_rows)
        validation_df.to_parquet(
            output_dir / "radar_target_mapping_validation.parquet", index=False
        )
        valid_shape_df = (
            validation_df.loc[validation_df.get("shape_match", False) == True]
            if not validation_df.empty and "shape_match" in validation_df.columns
            else pd.DataFrame()
        )
        if valid_shape_df.empty:
            mapping_summary = {
                "status": "WARN",
                "requested_samples": int(target_validation_samples),
                "compared_samples": int(compared),
                "missing_cache_candidates": int(missing_cache_for_sample),
                "reason": "no shape-compatible cache/Zarr samples were compared",
            }
        else:
            max_target_diff = float(valid_shape_df["max_abs_diff_stored_vs_expected_log1p"].max())
            max_inverse_diff = float(valid_shape_df["max_abs_diff_expm1_vs_clipped_cache"].max())
            total_mask_mismatches = int(valid_shape_df["mask_mismatch_count"].sum())
            all_match = bool(valid_shape_df["mapping_matches_within_tolerance"].all())
            negative_source_values = int(valid_shape_df["raw_cache_negative_count"].sum())
            mapping_summary = {
                "status": "PASS" if all_match else "FAIL",
                "requested_samples": int(target_validation_samples),
                "compared_samples": int(compared),
                "shape_compatible_samples": int(len(valid_shape_df)),
                "missing_cache_candidates": int(missing_cache_for_sample),
                "absolute_tolerance": TARGET_MAPPING_ATOL,
                "max_abs_diff_stored_vs_expected_log1p": max_target_diff,
                "max_abs_diff_expm1_vs_clipped_cache": max_inverse_diff,
                "total_mask_mismatch_count": total_mask_mismatches,
                "source_negative_values_seen_in_compared_patches": negative_source_values,
                "all_compared_samples_match_builder_transform": all_match,
            }
            if not all_match:
                warn(
                    "radar_target_mapping_mismatch",
                    (
                        "Direct cache -> Zarr validation did not reproduce every target/mask within "
                        f"tolerance {TARGET_MAPPING_ATOL}. See radar_target_mapping_validation.parquet."
                    ),
                )
            else:
                note(
                    "radar_target_mapping_verified",
                    (
                        f"Directly reproduced cache -> Zarr target/mask for {len(valid_shape_df)} sampled patches: "
                        "isfinite mask + nan_to_num + clip(min=0) + log1p."
                    ),
                )
            if negative_source_values > 0:
                note(
                    "radar_negative_values_irreversible_after_clip",
                    (
                        f"Compared cache patches contained {negative_source_values} negative finite source values. "
                        "They map to target=0 after clipping, so expm1(target) cannot reconstruct their original magnitude."
                    ),
                )
        semantics["target_mapping_validation"] = mapping_summary

    write_json(output_dir / "radar_target_semantics.json", semantics)
    return semantics


def run_audit(args: argparse.Namespace) -> int:
    started_perf = time.perf_counter()
    started_utc = datetime.now(timezone.utc)

    dataset_dir = args.dataset_dir.expanduser().resolve()
    zarr_path = (
        args.zarr_path.expanduser().resolve()
        if args.zarr_path is not None
        else dataset_dir / "train.zarr"
    )
    output_dir = args.output_dir.expanduser().resolve()
    radar_cache_arg = args.radar_cache_dir
    if radar_cache_arg is None:
        env_cache = os.environ.get("CORRDIFF_RADAR_CACHE_DIR")
        radar_cache_arg = Path(env_cache) if env_cache else None
    radar_cache_dir = radar_cache_arg.expanduser().resolve() if radar_cache_arg is not None else None

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use --overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output_dir)

    logger.info("CorrDiff Phase 0 audit %s", AUDIT_VERSION)
    logger.info("Dataset directory: %s", dataset_dir)
    logger.info("Zarr path       : %s", zarr_path)
    logger.info("Output directory: %s", output_dir)
    logger.info("Radar cache     : %s", radar_cache_dir if radar_cache_dir is not None else "not supplied")
    logger.info("Mode            : %s", "quick" if args.quick else "full")

    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr dataset not found: {zarr_path}")

    metadata_path = dataset_dir / "metadata.json"
    metadata = load_json(metadata_path)
    normalization_path = dataset_dir / "normalization.npz"

    root = zarr.open_group(str(zarr_path), mode="r")
    array_names = set(root.array_keys())
    missing_required = [name for name in REQUIRED_ARRAYS if name not in array_names]
    if missing_required:
        raise KeyError(f"Missing required Zarr arrays: {missing_required}")

    arrays = {name: root[name] for name in sorted(array_names)}
    array_info = {name: array_description(array) for name, array in arrays.items()}

    warnings: list[dict[str, str]] = []
    notes: list[dict[str, str]] = []

    def warn(code: str, message: str) -> None:
        logger.warning("%s: %s", code, message)
        warnings.append({"code": code, "message": message})

    def note(code: str, message: str) -> None:
        logger.info("%s: %s", code, message)
        notes.append({"code": code, "message": message})

    n_samples = int(arrays["input"].shape[0])
    input_shape = tuple(int(v) for v in arrays["input"].shape)
    target_shape = tuple(int(v) for v in arrays["target"].shape)
    mask_shape = tuple(int(v) for v in arrays["mask"].shape)
    timestamp_shape = tuple(int(v) for v in arrays["timestamps"].shape)

    sample_dims = {
        "input": input_shape[0],
        "target": target_shape[0],
        "mask": mask_shape[0],
        "timestamps": timestamp_shape[0],
    }
    for optional in OPTIONAL_ARRAYS:
        if optional in arrays:
            sample_dims[optional] = int(arrays[optional].shape[0])
    sample_dimension_consistent = len(set(sample_dims.values())) == 1
    if not sample_dimension_consistent:
        warn("sample_dimension_mismatch", f"Sample dimensions differ: {sample_dims}")

    if len(input_shape) != 4:
        warn("input_rank", f"input should be 4-D (N,C,H,W), found {input_shape}")
    if len(target_shape) != 4 or target_shape[1] != 1:
        warn("target_shape", f"target should be (N,1,H,W), found {target_shape}")
    if len(mask_shape) != 4 or mask_shape[1] != 1:
        warn("mask_shape", f"mask should be (N,1,H,W), found {mask_shape}")
    if target_shape != mask_shape:
        warn("target_mask_shape", f"target {target_shape} != mask {mask_shape}")
    if input_shape[0] != target_shape[0]:
        warn("input_target_samples", "input and target sample counts differ")
    if input_shape[2:] != target_shape[2:]:
        warn("input_target_spatial", f"input spatial {input_shape[2:]} != target {target_shape[2:]}")

    channels = list(root.attrs.get("channels", metadata.get("channels", [])))
    if not channels:
        channels = [f"channel_{index}" for index in range(input_shape[1])]
        warn("channel_names_missing", "Channel names absent; generated generic names.")
    if len(channels) != input_shape[1]:
        warn(
            "channel_count_mismatch",
            f"Found {len(channels)} channel names for C={input_shape[1]}; using positional fallback where needed.",
        )
        channels = [
            channels[index] if index < len(channels) else f"channel_{index}"
            for index in range(input_shape[1])
        ]

    metadata_num_samples = metadata.get("num_samples")
    if metadata_num_samples is not None and int(metadata_num_samples) != n_samples:
        warn(
            "metadata_num_samples",
            f"metadata.json reports {metadata_num_samples} samples, Zarr has {n_samples}.",
        )

    patch_size = int(metadata.get("patch_size", input_shape[-1]))
    stride = int(metadata.get("stride", 0)) if metadata.get("stride") is not None else None
    radar_grid_shape_raw = metadata.get("radar_grid_shape")
    radar_grid_shape = (
        tuple(int(v) for v in radar_grid_shape_raw)
        if radar_grid_shape_raw and len(radar_grid_shape_raw) == 2
        else None
    )
    target_transform = root.attrs.get(
        "target_transform", metadata.get("target_transform", "unknown")
    )
    if target_transform == "log1p":
        note(
            "target_transform_log1p",
            (
                "Stored target uses log1p. In the audited builder, the source cache is created from PNG RGB colors "
                "mapped to numerical radar-legend values and saved as float32; the builder itself does not declare "
                "the physical unit as dBZ. Direct cache -> target validation is performed when --radar-cache-dir is supplied."
            ),
        )

    # ------------------------------------------------------------------
    # Temporal audit: timestamps are small enough to read in full.
    # ------------------------------------------------------------------
    logger.info("Reading timestamp index...")
    timestamp_raw = np.asarray(arrays["timestamps"][:])
    unique_raw, sample_counts = np.unique(timestamp_raw, return_counts=True)
    unique_dt, timestamp_unit = decode_unix_timestamps(unique_raw)
    sample_count_by_timestamp = pd.Series(sample_counts, index=unique_dt, name="samples")
    sample_count_by_timestamp.index.name = "timestamp"

    actual_first = unique_dt.min() if len(unique_dt) else None
    actual_last = unique_dt.max() if len(unique_dt) else None

    expected_index = pd.DatetimeIndex([])
    if metadata.get("start_date") and metadata.get("end_date") and metadata.get("time_frequency"):
        expected_start = normalize_timestamp(metadata["start_date"])
        expected_end = normalize_timestamp(metadata["end_date"])
        expected_index = pd.date_range(
            expected_start, expected_end, freq=str(metadata["time_frequency"])
        )
    elif len(unique_dt):
        # Fall back to the observed interval at hourly spacing only if metadata is absent.
        warn(
            "expected_time_index_missing",
            "metadata.json lacks start/end/frequency; temporal coverage ratio cannot be fully validated.",
        )

    if len(expected_index):
        temporal = pd.DataFrame({"timestamp": expected_index})
        counts_lookup = sample_count_by_timestamp.to_dict()
        temporal["samples"] = temporal["timestamp"].map(counts_lookup).fillna(0).astype(np.int32)
        temporal["available"] = temporal["samples"] > 0
        temporal["year"] = temporal["timestamp"].dt.year.astype(np.int16)
        temporal["month"] = temporal["timestamp"].dt.month.astype(np.int8)
        temporal["hour_utc"] = temporal["timestamp"].dt.hour.astype(np.int8)

        observed_outside = unique_dt.difference(expected_index)
        if len(observed_outside):
            warn(
                "timestamps_outside_metadata_range",
                f"{len(observed_outside)} observed timestamps are outside the metadata expected index.",
            )
    else:
        temporal = pd.DataFrame(
            {
                "timestamp": unique_dt,
                "samples": sample_counts.astype(np.int32),
                "available": np.ones(len(unique_dt), dtype=bool),
            }
        )
        temporal["year"] = temporal["timestamp"].dt.year.astype(np.int16)
        temporal["month"] = temporal["timestamp"].dt.month.astype(np.int8)
        temporal["hour_utc"] = temporal["timestamp"].dt.hour.astype(np.int8)

    temporal.to_parquet(output_dir / "temporal_coverage.parquet", index=False)

    yearly = (
        temporal.groupby("year", as_index=False)
        .agg(
            expected_timestamps=("timestamp", "size"),
            available_timestamps=("available", "sum"),
            samples=("samples", "sum"),
        )
        .sort_values("year")
    )
    yearly["missing_timestamps"] = yearly["expected_timestamps"] - yearly["available_timestamps"]
    yearly["coverage_ratio"] = yearly["available_timestamps"] / yearly["expected_timestamps"]
    yearly["mean_patches_per_available_timestamp"] = np.where(
        yearly["available_timestamps"] > 0,
        yearly["samples"] / yearly["available_timestamps"],
        np.nan,
    )
    yearly.to_parquet(output_dir / "yearly_coverage.parquet", index=False)

    monthly = (
        temporal.groupby(["year", "month"], as_index=False)
        .agg(
            expected_timestamps=("timestamp", "size"),
            available_timestamps=("available", "sum"),
            samples=("samples", "sum"),
        )
        .sort_values(["year", "month"])
    )
    monthly["missing_timestamps"] = monthly["expected_timestamps"] - monthly["available_timestamps"]
    monthly["coverage_ratio"] = monthly["available_timestamps"] / monthly["expected_timestamps"]
    monthly["mean_patches_per_available_timestamp"] = np.where(
        monthly["available_timestamps"] > 0,
        monthly["samples"] / monthly["available_timestamps"],
        np.nan,
    )
    monthly.to_parquet(output_dir / "monthly_coverage.parquet", index=False)

    hourly = (
        temporal.groupby("hour_utc", as_index=False)
        .agg(
            expected_timestamps=("timestamp", "size"),
            available_timestamps=("available", "sum"),
            samples=("samples", "sum"),
        )
        .sort_values("hour_utc")
    )
    hourly["coverage_ratio"] = hourly["available_timestamps"] / hourly["expected_timestamps"]
    hourly["mean_patches_per_available_timestamp"] = np.where(
        hourly["available_timestamps"] > 0,
        hourly["samples"] / hourly["available_timestamps"],
        np.nan,
    )
    hourly.to_parquet(output_dir / "hourly_coverage.parquet", index=False)

    expected_timestamps = int(len(expected_index)) if len(expected_index) else None
    available_timestamps = int(len(unique_dt))
    temporal_coverage_ratio = (
        safe_fraction(int(temporal["available"].sum()), len(temporal))
        if len(temporal)
        else None
    )
    if temporal_coverage_ratio is not None and temporal_coverage_ratio < 1.0:
        note(
            "temporal_gaps_present",
            f"Dataset contains {int((~temporal['available']).sum())} expected timestamps with no written patch; inspect temporal_coverage.parquet and builder counters.",
        )

    patches_per_timestamp = {
        "min": int(sample_counts.min()) if sample_counts.size else None,
        "p25": float(np.quantile(sample_counts, 0.25)) if sample_counts.size else None,
        "median": float(np.median(sample_counts)) if sample_counts.size else None,
        "mean": float(sample_counts.mean()) if sample_counts.size else None,
        "p75": float(np.quantile(sample_counts, 0.75)) if sample_counts.size else None,
        "max": int(sample_counts.max()) if sample_counts.size else None,
    }

    # ------------------------------------------------------------------
    # Patch-position and spatial-footprint audit.
    # ------------------------------------------------------------------
    expected_positions = expected_patch_positions(radar_grid_shape, patch_size, stride)
    patch_position_df = pd.DataFrame(columns=["patch_row", "patch_col", "samples"])
    actual_positions: list[tuple[int, int]] = []
    patch_rows: np.ndarray | None = None
    patch_cols: np.ndarray | None = None

    if "patch_row" in arrays and "patch_col" in arrays:
        logger.info("Reading patch positions...")
        patch_rows = np.asarray(arrays["patch_row"][:], dtype=np.int32)
        patch_cols = np.asarray(arrays["patch_col"][:], dtype=np.int32)
        pairs = np.column_stack([patch_rows, patch_cols])
        unique_pairs, pair_counts = np.unique(pairs, axis=0, return_counts=True)
        patch_position_df = pd.DataFrame(
            {
                "patch_row": unique_pairs[:, 0].astype(np.int32),
                "patch_col": unique_pairs[:, 1].astype(np.int32),
                "samples": pair_counts.astype(np.int64),
            }
        ).sort_values(["patch_row", "patch_col"])
        patch_position_df.to_parquet(output_dir / "patch_position_counts.parquet", index=False)
        actual_positions = [tuple(map(int, pair)) for pair in unique_pairs]

        if expected_positions:
            unexpected = sorted(set(actual_positions) - set(expected_positions))
            missing_positions = sorted(set(expected_positions) - set(actual_positions))
            if unexpected:
                warn("unexpected_patch_positions", f"Unexpected patch anchors found: {unexpected[:20]}")
            if missing_positions:
                note(
                    "missing_patch_positions",
                    f"Some expected patch anchors have no samples: {missing_positions[:20]}",
                )
    else:
        warn(
            "patch_coordinates_missing",
            "patch_row/patch_col are absent; spatial patch coverage cannot be reconstructed exactly.",
        )

    expected_spatial_coverage_ratio = None
    actual_spatial_coverage_ratio = None
    if radar_grid_shape and patch_size:
        if expected_positions:
            expected_map = build_coverage_map(radar_grid_shape, patch_size, expected_positions)
            np.save(output_dir / "expected_patch_spatial_coverage.npy", expected_map)
            expected_spatial_coverage_ratio = float((expected_map > 0).mean())
            if expected_spatial_coverage_ratio < 1.0:
                note(
                    "patch_geometry_uncovered_edges",
                    (
                        f"Patch geometry covers {expected_spatial_coverage_ratio:.2%} of the {radar_grid_shape[0]}x{radar_grid_shape[1]} radar grid. "
                        "This follows directly from patch_size/stride and should be considered when reconstructing full fields."
                    ),
                )
        if actual_positions:
            actual_map = build_coverage_map(radar_grid_shape, patch_size, actual_positions)
            np.save(output_dir / "actual_patch_spatial_coverage.npy", actual_map)
            actual_spatial_coverage_ratio = float((actual_map > 0).mean())

    # ------------------------------------------------------------------
    # Radar-cache semantics and direct cache -> Zarr target validation.
    # ------------------------------------------------------------------
    logger.info("Auditing radar cache semantics and target transformation...")
    radar_semantics = audit_radar_cache_and_target_mapping(
        cache_dir=radar_cache_dir,
        arrays=arrays,
        timestamp_raw=timestamp_raw,
        timestamp_unit=timestamp_unit,
        unique_timestamps=unique_dt,
        patch_rows=patch_rows,
        patch_cols=patch_cols,
        patch_size=patch_size,
        radar_grid_shape=radar_grid_shape,
        file_sample_count=args.radar_cache_file_sample,
        target_validation_samples=args.radar_target_validation_samples,
        seed=args.seed + 1009,
        output_dir=output_dir,
        warn=warn,
        note=note,
        logger=logger,
    )

    # ------------------------------------------------------------------
    # Numerical scan.
    # ------------------------------------------------------------------
    channel_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    mask_metrics: dict[str, Any] = {}
    target_integrity: dict[str, Any] = {}

    rng = np.random.default_rng(args.seed)
    if not args.quick:
        logger.info("Starting full numerical scan over %d samples...", n_samples)
        input_acc = NumericAccumulator(channels)
        target_acc = NumericAccumulator(["radar_target_valid"])

        mask_total = 0
        mask_finite = 0
        mask_nan = 0
        mask_posinf = 0
        mask_neginf = 0
        mask_valid_pixels = 0
        mask_binary_violations = 0
        target_negative_finite = 0
        target_invalid_nonzero = 0
        target_all_nan = 0
        target_all_posinf = 0
        target_all_neginf = 0

        total_batches = math.ceil(n_samples / args.batch_samples) if n_samples else 0
        per_batch_quantile = max(
            1,
            math.ceil(args.quantile_sample_values / max(total_batches, 1)),
        )

        for sample_slice in tqdm(
            iter_slices(n_samples, args.batch_samples),
            total=total_batches,
            desc="Phase 0 full scan",
        ):
            x = np.asarray(arrays["input"][sample_slice])
            y = np.asarray(arrays["target"][sample_slice])
            m = np.asarray(arrays["mask"][sample_slice])

            input_acc.update_channels(x, rng, per_batch_quantile)

            finite_m = np.isfinite(m)
            valid_m = finite_m & (m > 0.5)
            mask_total += int(m.size)
            mask_finite += int(finite_m.sum())
            mask_nan += int(np.isnan(m).sum())
            mask_posinf += int(np.isposinf(m).sum())
            mask_neginf += int(np.isneginf(m).sum())
            mask_valid_pixels += int(valid_m.sum())
            mask_binary_violations += int(
                (finite_m & ~(np.isclose(m, 0.0) | np.isclose(m, 1.0))).sum()
            )

            target_acc.update_single(y, valid_m, rng, per_batch_quantile)
            finite_y = np.isfinite(y)
            target_negative_finite += int((finite_y & (y < 0.0)).sum())
            target_invalid_nonzero += int(
                ((~valid_m) & finite_y & (np.abs(y) > 1e-12)).sum()
            )
            target_all_nan += int(np.isnan(y).sum())
            target_all_posinf += int(np.isposinf(y).sum())
            target_all_neginf += int(np.isneginf(y).sum())

        channel_rows.extend(input_acc.to_rows("input", args.quantile_sample_values, rng))
        target_rows = target_acc.to_rows("target", args.quantile_sample_values, rng)
        channel_rows.extend(target_rows)

        for row in channel_rows:
            missing_rows.append(
                {
                    "kind": row["kind"],
                    "channel_index": row["channel_index"],
                    "channel": row["channel"],
                    "total_count": row["total_count"],
                    "finite_count": row["finite_count"],
                    "nan_count": row["nan_count"],
                    "posinf_count": row["posinf_count"],
                    "neginf_count": row["neginf_count"],
                    "finite_ratio": row["finite_ratio"],
                }
            )

        mask_metrics = {
            "total_values": mask_total,
            "finite_count": mask_finite,
            "nan_count": mask_nan,
            "posinf_count": mask_posinf,
            "neginf_count": mask_neginf,
            "valid_pixel_count": mask_valid_pixels,
            "valid_pixel_ratio": safe_fraction(mask_valid_pixels, mask_total),
            "binary_violation_count": mask_binary_violations,
        }
        target_integrity = {
            "negative_finite_value_count": target_negative_finite,
            "invalid_mask_region_nonzero_count": target_invalid_nonzero,
            "nan_count_all_stored_target_pixels": target_all_nan,
            "posinf_count_all_stored_target_pixels": target_all_posinf,
            "neginf_count_all_stored_target_pixels": target_all_neginf,
        }

        missing_rows.append(
            {
                "kind": "mask",
                "channel_index": 0,
                "channel": "radar_valid_mask",
                "total_count": mask_total,
                "finite_count": mask_finite,
                "nan_count": mask_nan,
                "posinf_count": mask_posinf,
                "neginf_count": mask_neginf,
                "finite_ratio": safe_fraction(mask_finite, mask_total),
            }
        )

        # Compare with builder normalization outputs when available.
        if normalization_path.exists():
            normalization = np.load(normalization_path)
            builder_input_mean = np.asarray(normalization.get("input_mean", []), dtype=np.float64)
            builder_input_std = np.asarray(normalization.get("input_std", []), dtype=np.float64)
            builder_target_mean = np.asarray(normalization.get("target_mean", []), dtype=np.float64)
            builder_target_std = np.asarray(normalization.get("target_std", []), dtype=np.float64)

            for row in channel_rows:
                if row["kind"] == "input" and row["channel_index"] < len(builder_input_mean):
                    idx = row["channel_index"]
                    row["builder_mean"] = float(builder_input_mean[idx])
                    row["builder_std"] = float(builder_input_std[idx])
                    row["mean_abs_diff_vs_builder"] = abs(row["mean"] - row["builder_mean"])
                    row["std_abs_diff_vs_builder"] = abs(row["std"] - row["builder_std"])
                elif row["kind"] == "target" and builder_target_mean.size:
                    row["builder_mean"] = float(builder_target_mean.ravel()[0])
                    row["builder_std"] = float(builder_target_std.ravel()[0])
                    row["mean_abs_diff_vs_builder"] = abs(row["mean"] - row["builder_mean"])
                    row["std_abs_diff_vs_builder"] = abs(row["std"] - row["builder_std"])

        channel_df = pd.DataFrame(channel_rows)
        channel_df.to_parquet(output_dir / "channel_summary.parquet", index=False)
        pd.DataFrame(missing_rows).to_parquet(output_dir / "missing_data.parquet", index=False)

        input_nonfinite = sum(
            int(row["nan_count"]) + int(row["posinf_count"]) + int(row["neginf_count"])
            for row in channel_rows
            if row["kind"] == "input"
        )
        if input_nonfinite > 0:
            warn(
                "input_nonfinite_values",
                f"Found {input_nonfinite} non-finite values in stored input patches.",
            )
        if mask_binary_violations > 0:
            warn(
                "mask_not_binary",
                f"Found {mask_binary_violations} finite mask values different from 0/1.",
            )
        if target_invalid_nonzero > 0:
            warn(
                "target_nonzero_outside_mask",
                f"Found {target_invalid_nonzero} non-zero target values where mask is invalid.",
            )
        if target_negative_finite > 0 and target_transform == "log1p":
            warn(
                "negative_log1p_target",
                f"Found {target_negative_finite} negative stored target values despite target_transform=log1p.",
            )
    else:
        logger.info("Quick mode: skipping full pixel scan.")
        if normalization_path.exists():
            normalization = np.load(normalization_path)
            means = np.asarray(normalization.get("input_mean", []), dtype=np.float64)
            stds = np.asarray(normalization.get("input_std", []), dtype=np.float64)
            for idx, channel in enumerate(channels):
                channel_rows.append(
                    {
                        "kind": "input",
                        "channel_index": idx,
                        "channel": channel,
                        "mean": float(means[idx]) if idx < len(means) else np.nan,
                        "std": float(stds[idx]) if idx < len(stds) else np.nan,
                        "statistics_source": "builder_normalization_npz",
                    }
                )
            target_mean = np.asarray(normalization.get("target_mean", []), dtype=np.float64)
            target_std = np.asarray(normalization.get("target_std", []), dtype=np.float64)
            channel_rows.append(
                {
                    "kind": "target",
                    "channel_index": 0,
                    "channel": "radar_target_valid",
                    "mean": float(target_mean.ravel()[0]) if target_mean.size else np.nan,
                    "std": float(target_std.ravel()[0]) if target_std.size else np.nan,
                    "statistics_source": "builder_normalization_npz",
                }
            )
            pd.DataFrame(channel_rows).to_parquet(
                output_dir / "channel_summary.parquet", index=False
            )
        else:
            warn(
                "quick_mode_no_normalization",
                "Quick mode was requested but normalization.npz is absent, so numerical channel statistics are unavailable.",
            )

    # Builder counters are useful context for missing timestamps/patches.
    counters = metadata.get("counters", {}) if isinstance(metadata.get("counters", {}), dict) else {}

    expected_patch_count = len(expected_positions) if expected_positions else None
    if expected_patch_count and patches_per_timestamp.get("max") is not None:
        if patches_per_timestamp["max"] > expected_patch_count:
            warn(
                "patches_per_timestamp_exceeds_geometry",
                f"Observed max patches/timestamp={patches_per_timestamp['max']} > expected anchors={expected_patch_count}.",
            )

    summary = {
        "audit_version": AUDIT_VERSION,
        "audit_mode": "quick" if args.quick else "full",
        "audit_started_utc": started_utc.isoformat(),
        "audit_finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started_perf,
        "system": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "pid": os.getpid(),
        },
        "paths": {
            "dataset_dir": dataset_dir,
            "zarr_path": zarr_path,
            "metadata_path": metadata_path,
            "normalization_path": normalization_path,
            "radar_cache_dir": radar_cache_dir,
            "output_dir": output_dir,
        },
        "dataset": {
            "num_samples": n_samples,
            "num_channels": input_shape[1] if len(input_shape) >= 2 else None,
            "channels": channels,
            "input_shape": input_shape,
            "target_shape": target_shape,
            "mask_shape": mask_shape,
            "sample_dimensions": sample_dims,
            "sample_dimension_consistent": sample_dimension_consistent,
            "zarr_arrays": array_info,
            "zarr_attrs": dict(root.attrs),
            "target_transform": target_transform,
        },
        "configuration": {
            "start_date": metadata.get("start_date"),
            "end_date": metadata.get("end_date"),
            "time_frequency": metadata.get("time_frequency"),
            "patch_size": patch_size,
            "stride": stride,
            "radar_grid_shape": radar_grid_shape,
            "radar_resolution_km": metadata.get("radar_resolution_km"),
            "surface_variables": metadata.get("surface_variables", root.attrs.get("surface_variables")),
            "pressure_variables": metadata.get("pressure_variables", root.attrs.get("pressure_variables")),
            "pressure_levels_hpa": metadata.get("pressure_levels_hpa", root.attrs.get("pressure_levels_hpa")),
        },
        "temporal": {
            "timestamp_storage_unit_detected": timestamp_unit,
            "expected_timestamps": expected_timestamps,
            "available_timestamps": available_timestamps,
            "missing_timestamps": (
                expected_timestamps - int(temporal["available"].sum())
                if expected_timestamps is not None
                else None
            ),
            "coverage_ratio": temporal_coverage_ratio,
            "first_observed_timestamp": actual_first,
            "last_observed_timestamp": actual_last,
            "patches_per_timestamp": patches_per_timestamp,
        },
        "patch_geometry": {
            "expected_patch_positions_per_full_field": expected_patch_count,
            "actual_unique_patch_positions": len(actual_positions) if actual_positions else None,
            "expected_spatial_coverage_ratio": expected_spatial_coverage_ratio,
            "actual_spatial_coverage_ratio": actual_spatial_coverage_ratio,
        },
        "mask": mask_metrics,
        "target_integrity": target_integrity,
        "radar_target_semantics": radar_semantics,
        "builder_counters": counters,
        "warnings_count": len(warnings),
        "notes_count": len(notes),
        "status": "WARN" if warnings else "PASS",
    }

    write_json(output_dir / "dataset_summary.json", summary)
    write_json(output_dir / "audit_warnings.json", {"warnings": warnings, "notes": notes})

    logger.info("=" * 72)
    logger.info("PHASE 0 AUDIT COMPLETED")
    logger.info("Samples                  : %d", n_samples)
    logger.info("Unique timestamps        : %d", available_timestamps)
    if expected_timestamps is not None:
        logger.info("Expected timestamps      : %d", expected_timestamps)
        logger.info("Temporal coverage        : %.2f%%", 100.0 * (temporal_coverage_ratio or 0.0))
    logger.info("Channels                 : %d", len(channels))
    logger.info("Warnings                 : %d", len(warnings))
    logger.info("Notes                    : %d", len(notes))
    logger.info("Output                    : %s", output_dir)
    logger.info("Elapsed                   : %.2f s", summary["elapsed_seconds"])
    logger.info("=" * 72)
    return 0


def main() -> int:
    args = parse_args()
    if args.batch_samples <= 0:
        raise ValueError("--batch-samples must be > 0")
    if args.quantile_sample_values < 0:
        raise ValueError("--quantile-sample-values must be >= 0")
    if args.radar_cache_file_sample < 0:
        raise ValueError("--radar-cache-file-sample must be >= 0")
    if args.radar_target_validation_samples < 0:
        raise ValueError("--radar-target-validation-samples must be >= 0")
    return run_audit(args)


if __name__ == "__main__":
    raise SystemExit(main())
