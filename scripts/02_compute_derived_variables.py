#!/usr/bin/env python3
"""CorrDiff Phase 2 v2 — physically motivated derived variables.

Correction in v2
----------------
The v1 script sampled an independent random reservoir for each derived variable
and then calculated correlations by row position. Those rows did not refer to
the same pixels, so `derived_correlations.parquet` was invalid.

v2 keeps a SINGLE ALIGNED multivariate reservoir: whenever a pixel is selected,
all derived variables for that same patch/pixel are stored together. Pearson
and Spearman correlations therefore compare physically corresponding samples.

The script does not modify the CorrDiff training dataset.

Expected channels
-----------------
tcwv, t2m, u10, v10, t_850, r_850, u_850, v_850,
t_500, r_500, u_500, v_500

Derived fields
--------------
wind_speed_10            = sqrt(u10^2 + v10^2)
wind_speed_850           = sqrt(u_850^2 + v_850^2)
wind_speed_500           = sqrt(u_500^2 + v_500^2)
delta_t_500_850          = t_500 - t_850
delta_t_850_surface      = t_850 - t2m
delta_r_500_850          = r_500 - r_850
bulk_wind_diff_10_850    = sqrt((u_850-u10)^2 + (v_850-v10)^2)
bulk_wind_diff_850_500   = sqrt((u_500-u_850)^2 + (v_500-v_850)^2)

The bulk wind differences are vector-wind differences (m/s), not
height-normalized shear rates (s^-1).

Outputs
-------
<output_dir>/
  analysis_summary.json
  formula_catalog.parquet
  sampling_blocks.parquet
  derived_summary.parquet
  derived_quantiles.parquet
  derived_histograms.parquet
  derived_correlations.parquet
  derived_samples.npz
  phase2.log
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import zarr

try:
    from scipy.stats import skew, kurtosis, spearmanr
except ImportError as exc:
    raise SystemExit("Phase 2 v2 requires scipy. Install with: pip install scipy") from exc


PHASE_VERSION = "phase2-derived-v2-aligned"

FORMULAS: list[dict[str, str]] = [
    {"name": "wind_speed_10", "formula": "sqrt(u10^2 + v10^2)",
     "unit": "m s^-1", "interpretation": "Magnitude do vento a 10 m."},
    {"name": "wind_speed_850", "formula": "sqrt(u_850^2 + v_850^2)",
     "unit": "m s^-1", "interpretation": "Magnitude do vento em 850 hPa."},
    {"name": "wind_speed_500", "formula": "sqrt(u_500^2 + v_500^2)",
     "unit": "m s^-1", "interpretation": "Magnitude do vento em 500 hPa."},
    {"name": "delta_t_500_850", "formula": "t_500 - t_850",
     "unit": "K", "interpretation": "Diferença vertical de temperatura 500-850 hPa."},
    {"name": "delta_t_850_surface", "formula": "t_850 - t2m",
     "unit": "K", "interpretation": "Diferença de temperatura entre 850 hPa e 2 m."},
    {"name": "delta_r_500_850", "formula": "r_500 - r_850",
     "unit": "percentage points", "interpretation": "Diferença de umidade relativa 500-850 hPa."},
    {"name": "bulk_wind_diff_10_850",
     "formula": "sqrt((u_850-u10)^2 + (v_850-v10)^2)",
     "unit": "m s^-1",
     "interpretation": "Diferença vetorial de vento 10 m-850 hPa; proxy de cisalhamento."},
    {"name": "bulk_wind_diff_850_500",
     "formula": "sqrt((u_500-u_850)^2 + (v_500-v_850)^2)",
     "unit": "m s^-1",
     "interpretation": "Diferença vetorial de vento 850-500 hPa; proxy de cisalhamento."},
]

REQUIRED_CHANNELS = [
    "tcwv", "t2m", "u10", "v10", "t_850", "r_850", "u_850", "v_850",
    "t_500", "r_500", "u_500", "v_500",
]

QUANTILES = [
    ("p001", 0.001), ("p01", 0.01), ("p05", 0.05), ("p25", 0.25),
    ("p50", 0.50), ("p75", 0.75), ("p95", 0.95), ("p99", 0.99),
    ("p999", 0.999),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CorrDiff Phase 2 v2: derived variables with aligned correlations"
    )
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--phase1-dir", type=Path,
                   default=Path("analysis_outputs/01_univariate"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("analysis_outputs/02_derived"))
    p.add_argument("--sample-patches", type=int, default=32768)
    p.add_argument("--block-size", type=int, default=0)
    p.add_argument(
        "--reservoir-values", type=int, default=300000,
        help="Maximum aligned pixel rows retained for quantiles/correlations.",
    )
    p.add_argument("--hist-bins", type=int, default=120)
    p.add_argument("--seed", type=int, default=20260917)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {path}; use --overwrite"
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def setup_logger(out: Path) -> logging.Logger:
    logger = logging.getLogger("corrdiff.phase2.v2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(out / "phase2.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def open_group(path: Path) -> Any:
    try:
        return zarr.open_group(str(path), mode="r")
    except Exception:
        return zarr.open(str(path), mode="r")


def select_blocks(
    n: int,
    block_size: int,
    sample_patches: int,
    rng: np.random.Generator,
) -> np.ndarray:
    total_blocks = math.ceil(n / block_size)
    needed = min(total_blocks, max(1, math.ceil(sample_patches / block_size)))
    if needed == total_blocks:
        return np.arange(total_blocks, dtype=np.int64)
    width = total_blocks / needed
    pos = (np.arange(needed) + rng.random(needed)) * width
    ids = np.unique(
        np.clip(np.floor(pos).astype(np.int64), 0, total_blocks - 1)
    )
    if ids.size < needed:
        remain = np.setdiff1d(
            np.arange(total_blocks, dtype=np.int64), ids, assume_unique=True
        )
        ids = np.sort(
            np.concatenate([
                ids,
                rng.choice(remain, needed - ids.size, replace=False),
            ])
        )
    return ids


def derive(x: np.ndarray, index: dict[str, int]) -> dict[str, np.ndarray]:
    def c(name: str) -> np.ndarray:
        return x[:, index[name], :, :].astype(np.float32, copy=False)

    u10, v10, t2m = c("u10"), c("v10"), c("t2m")
    t850, r850, u850, v850 = c("t_850"), c("r_850"), c("u_850"), c("v_850")
    t500, r500, u500, v500 = c("t_500"), c("r_500"), c("u_500"), c("v_500")

    return {
        "wind_speed_10": np.hypot(u10, v10),
        "wind_speed_850": np.hypot(u850, v850),
        "wind_speed_500": np.hypot(u500, v500),
        "delta_t_500_850": t500 - t850,
        "delta_t_850_surface": t850 - t2m,
        "delta_r_500_850": r500 - r850,
        "bulk_wind_diff_10_850": np.hypot(u850 - u10, v850 - v10),
        "bulk_wind_diff_850_500": np.hypot(u500 - u850, v500 - v850),
    }


def aligned_block_sample(
    derived: dict[str, np.ndarray],
    names: list[str],
    quota: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return rows [pixel, variable] using one shared pixel index."""
    flat = [derived[name].reshape(-1) for name in names]
    valid = np.ones(flat[0].size, dtype=bool)
    for values in flat:
        valid &= np.isfinite(values)
    valid_ids = np.flatnonzero(valid)
    if valid_ids.size == 0:
        return np.empty((0, len(names)), dtype=np.float32)
    take = min(quota, valid_ids.size)
    selected = rng.choice(valid_ids, size=take, replace=False)
    return np.column_stack([values[selected] for values in flat]).astype(
        np.float32, copy=False
    )


def main() -> None:
    args = parse_args()
    prepare_output(args.output_dir, args.overwrite)
    logger = setup_logger(args.output_dir)
    rng = np.random.default_rng(args.seed)

    root = open_group(args.dataset_dir / "train.zarr")
    xds = root["input"]
    n, c, h, w = map(int, xds.shape)

    channels = list(root.attrs.get("channels", []))
    if not channels:
        metadata = args.dataset_dir / "metadata.json"
        if metadata.exists():
            channels = list(
                json.loads(metadata.read_text(encoding="utf-8")).get("channels", [])
            )
    if len(channels) != c:
        raise RuntimeError(
            f"Could not resolve {c} channel names; got {channels}"
        )
    missing = [name for name in REQUIRED_CHANNELS if name not in channels]
    if missing:
        raise RuntimeError(f"Required channels missing: {missing}")
    channel_index = {name: channels.index(name) for name in channels}

    chunk0 = int(getattr(xds, "chunks", (256,))[0])
    block_size = args.block_size or chunk0

    sampling_source = "phase2_systematic"
    phase1_blocks = args.phase1_dir / "sampling_blocks.parquet"
    if phase1_blocks.exists() and args.block_size == 0:
        p1 = pd.read_parquet(phase1_blocks)
        if "block_id" in p1.columns and len(p1):
            candidate = np.sort(
                p1["block_id"].dropna().astype(np.int64).unique()
            )
            candidate = candidate[
                (candidate >= 0) &
                (candidate < math.ceil(n / block_size))
            ]
            if len(candidate):
                block_ids = candidate
                sampling_source = "phase1_sampling_blocks"
            else:
                block_ids = select_blocks(
                    n, block_size, args.sample_patches, rng
                )
        else:
            block_ids = select_blocks(
                n, block_size, args.sample_patches, rng
            )
    else:
        block_ids = select_blocks(
            n, block_size, args.sample_patches, rng
        )

    names = [row["name"] for row in FORMULAS]
    unit_map = {row["name"]: row["unit"] for row in FORMULAS}
    formula_map = {row["name"]: row["formula"] for row in FORMULAS}

    logger.info("Dataset shape: %s", xds.shape)
    logger.info(
        "Sampling %d blocks of %d patches (%s)",
        len(block_ids), block_size, sampling_source,
    )

    count = {name: 0 for name in names}
    sums = {name: 0.0 for name in names}
    sumsq = {name: 0.0 for name in names}
    mins = {name: np.inf for name in names}
    maxs = {name: -np.inf for name in names}

    per_block_quota = max(
        1, math.ceil(args.reservoir_values / max(1, len(block_ids)))
    )
    aligned_chunks: list[np.ndarray] = []
    block_rows: list[dict[str, Any]] = []

    for block_id in block_ids:
        start = int(block_id * block_size)
        end = min(n, start + block_size)
        x = np.asarray(xds[start:end], dtype=np.float32)
        d = derive(x, channel_index)

        block_rows.append({
            "block_id": int(block_id),
            "start_patch": start,
            "end_patch": end,
            "patches": end - start,
        })

        for name, arr in d.items():
            values = arr.reshape(-1)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            v64 = values.astype(np.float64, copy=False)
            count[name] += int(values.size)
            sums[name] += float(v64.sum(dtype=np.float64))
            sumsq[name] += float(np.square(v64).sum(dtype=np.float64))
            mins[name] = min(mins[name], float(values.min()))
            maxs[name] = max(maxs[name], float(values.max()))

        aligned = aligned_block_sample(
            d, names, per_block_quota, rng
        )
        if aligned.size:
            aligned_chunks.append(aligned)

    pd.DataFrame(block_rows).to_parquet(
        args.output_dir / "sampling_blocks.parquet", index=False
    )
    pd.DataFrame(FORMULAS).to_parquet(
        args.output_dir / "formula_catalog.parquet", index=False
    )

    if aligned_chunks:
        aligned_matrix = np.concatenate(aligned_chunks, axis=0)
        if len(aligned_matrix) > args.reservoir_values:
            keep = rng.choice(
                len(aligned_matrix),
                size=args.reservoir_values,
                replace=False,
            )
            aligned_matrix = aligned_matrix[keep]
    else:
        aligned_matrix = np.empty((0, len(names)), dtype=np.float32)

    summary_rows: list[dict[str, Any]] = []
    quantile_rows: list[dict[str, Any]] = []
    hist_rows: list[dict[str, Any]] = []

    for j, name in enumerate(names):
        vals = (
            aligned_matrix[:, j]
            if aligned_matrix.size
            else np.empty(0, dtype=np.float32)
        )
        nval = count[name]
        mean = sums[name] / nval if nval else np.nan
        var = (
            max(sumsq[name] / nval - mean * mean, 0.0)
            if nval else np.nan
        )
        std = math.sqrt(var) if nval else np.nan

        summary_rows.append({
            "name": name,
            "formula": formula_map[name],
            "unit": unit_map[name],
            "sampled_pixel_count": nval,
            "aligned_reservoir_count": int(vals.size),
            "mean": mean,
            "std": std,
            "min": mins[name] if nval else np.nan,
            "max": maxs[name] if nval else np.nan,
            "skewness_reservoir": (
                float(skew(vals, bias=False)) if vals.size > 2 else np.nan
            ),
            "excess_kurtosis_reservoir": (
                float(kurtosis(vals, fisher=True, bias=False))
                if vals.size > 3 else np.nan
            ),
        })

        for label, q in QUANTILES:
            quantile_rows.append({
                "name": name,
                "quantile": label,
                "probability": q,
                "value": (
                    float(np.quantile(vals, q))
                    if vals.size else np.nan
                ),
            })

        if vals.size:
            hist_counts, edges = np.histogram(
                vals, bins=args.hist_bins
            )
            for i, cnt in enumerate(hist_counts):
                hist_rows.append({
                    "name": name,
                    "bin_index": i,
                    "left": float(edges[i]),
                    "right": float(edges[i + 1]),
                    "count": int(cnt),
                })

    pd.DataFrame(summary_rows).to_parquet(
        args.output_dir / "derived_summary.parquet", index=False
    )
    pd.DataFrame(quantile_rows).to_parquet(
        args.output_dir / "derived_quantiles.parquet", index=False
    )
    pd.DataFrame(hist_rows).to_parquet(
        args.output_dir / "derived_histograms.parquet", index=False
    )

    corr_rows: list[dict[str, Any]] = []
    if len(aligned_matrix) >= 3:
        pearson = np.corrcoef(
            aligned_matrix.astype(np.float64, copy=False), rowvar=False
        )
        spearman = spearmanr(
            aligned_matrix.astype(np.float64, copy=False), axis=0
        ).statistic
        spearman = np.asarray(spearman, dtype=np.float64)

        for i, a in enumerate(names):
            for j, b in enumerate(names):
                corr_rows.append({
                    "var_a": a,
                    "var_b": b,
                    "pearson_r": float(pearson[i, j]),
                    "spearman_rho": float(spearman[i, j]),
                    "n_aligned": int(len(aligned_matrix)),
                })

    pd.DataFrame(
        corr_rows,
        columns=[
            "var_a", "var_b", "pearson_r",
            "spearman_rho", "n_aligned",
        ],
    ).to_parquet(
        args.output_dir / "derived_correlations.parquet",
        index=False,
    )

    np.savez_compressed(
        args.output_dir / "derived_samples.npz",
        aligned_values=aligned_matrix.astype(np.float32),
        variable_names=np.asarray(names, dtype="U64"),
        **{
            name: aligned_matrix[:, j].astype(np.float32)
            for j, name in enumerate(names)
        },
    )

    summary = {
        "phase_version": PHASE_VERSION,
        "dataset_dir": str(args.dataset_dir),
        "dataset_shape": [n, c, h, w],
        "channels": channels,
        "derived_variables": names,
        "sample_patches_requested": int(args.sample_patches),
        "sampling_source": sampling_source,
        "sampled_blocks": int(len(block_ids)),
        "sampled_patches": int(
            sum(row["patches"] for row in block_rows)
        ),
        "aligned_reservoir_rows": int(len(aligned_matrix)),
        "reservoir_values_requested": int(args.reservoir_values),
        "correlation_alignment_verified": True,
        "correlation_note": (
            "All derived variables in each correlation row come from "
            "the same selected patch/pixel."
        ),
        "note": (
            "Bulk wind differences are vector-wind differences in m/s, "
            "not height-normalized shear rates."
        ),
    }

    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(
        "Phase 2 v2 complete. Aligned correlation sample: %d rows",
        len(aligned_matrix),
    )
    logger.info("Output: %s", args.output_dir)


if __name__ == "__main__":
    main()
