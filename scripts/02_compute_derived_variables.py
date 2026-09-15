#!/usr/bin/env python3
"""Phase 2: physically motivated derived variables for the CorrDiff dataset.

The script derives diagnostic fields from the 12 ERA5 channels stored in the
CorrDiff Zarr dataset. It intentionally does not alter the training dataset.
The goal is to characterize physically meaningful combinations that can later
be used in relationship, extreme-event and ablation analyses.

Expected canonical channels
---------------------------
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

The two bulk wind differences are proxies for vertical wind shear expressed as
wind-vector differences (m/s); they are not normalized by geometric height and
therefore are not shear rates in s^-1.

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
    from scipy.stats import skew, kurtosis
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Phase 2 requires scipy. Install with: pip install scipy") from exc


PHASE_VERSION = "phase2-derived-v1"

FORMULAS: list[dict[str, str]] = [
    {
        "name": "wind_speed_10",
        "formula": "sqrt(u10^2 + v10^2)",
        "unit": "m s^-1",
        "interpretation": "Magnitude do vento a 10 m.",
    },
    {
        "name": "wind_speed_850",
        "formula": "sqrt(u_850^2 + v_850^2)",
        "unit": "m s^-1",
        "interpretation": "Magnitude do vento em 850 hPa.",
    },
    {
        "name": "wind_speed_500",
        "formula": "sqrt(u_500^2 + v_500^2)",
        "unit": "m s^-1",
        "interpretation": "Magnitude do vento em 500 hPa.",
    },
    {
        "name": "delta_t_500_850",
        "formula": "t_500 - t_850",
        "unit": "K",
        "interpretation": "Diferença vertical de temperatura 500-850 hPa.",
    },
    {
        "name": "delta_t_850_surface",
        "formula": "t_850 - t2m",
        "unit": "K",
        "interpretation": "Diferença de temperatura entre 850 hPa e 2 m.",
    },
    {
        "name": "delta_r_500_850",
        "formula": "r_500 - r_850",
        "unit": "percentage points",
        "interpretation": "Diferença de umidade relativa 500-850 hPa.",
    },
    {
        "name": "bulk_wind_diff_10_850",
        "formula": "sqrt((u_850-u10)^2 + (v_850-v10)^2)",
        "unit": "m s^-1",
        "interpretation": "Diferença vetorial de vento 10 m-850 hPa; proxy de cisalhamento.",
    },
    {
        "name": "bulk_wind_diff_850_500",
        "formula": "sqrt((u_500-u_850)^2 + (v_500-v_850)^2)",
        "unit": "m s^-1",
        "interpretation": "Diferença vetorial de vento 850-500 hPa; proxy de cisalhamento.",
    },
]

REQUIRED_CHANNELS = [
    "tcwv", "t2m", "u10", "v10", "t_850", "r_850", "u_850", "v_850",
    "t_500", "r_500", "u_500", "v_500",
]

QUANTILES = [("p001", 0.001), ("p01", 0.01), ("p05", 0.05), ("p25", 0.25),
             ("p50", 0.50), ("p75", 0.75), ("p95", 0.95), ("p99", 0.99), ("p999", 0.999)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CorrDiff Phase 2 derived-variable analysis")
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--phase1-dir", type=Path, default=Path("analysis_outputs/01_univariate"))
    p.add_argument("--output-dir", type=Path, default=Path("analysis_outputs/02_derived"))
    p.add_argument("--sample-patches", type=int, default=32768)
    p.add_argument("--block-size", type=int, default=0)
    p.add_argument("--reservoir-values", type=int, default=300000)
    p.add_argument("--hist-bins", type=int, default=120)
    p.add_argument("--seed", type=int, default=20260915)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {path}; use --overwrite")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def setup_logger(out: Path) -> logging.Logger:
    logger = logging.getLogger("corrdiff.phase2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    fh = logging.FileHandler(out / "phase2.log", encoding="utf-8"); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger


def open_group(path: Path) -> Any:
    try:
        return zarr.open_group(str(path), mode="r")
    except Exception:
        return zarr.open(str(path), mode="r")


def select_blocks(n: int, block_size: int, sample_patches: int, rng: np.random.Generator) -> np.ndarray:
    total_blocks = math.ceil(n / block_size)
    needed = min(total_blocks, max(1, math.ceil(sample_patches / block_size)))
    if needed == total_blocks:
        return np.arange(total_blocks, dtype=np.int64)
    width = total_blocks / needed
    pos = (np.arange(needed) + rng.random(needed)) * width
    ids = np.unique(np.clip(np.floor(pos).astype(np.int64), 0, total_blocks - 1))
    if ids.size < needed:
        remain = np.setdiff1d(np.arange(total_blocks), ids, assume_unique=True)
        ids = np.sort(np.concatenate([ids, rng.choice(remain, needed - ids.size, replace=False)]))
    return ids


def derive(x: np.ndarray, index: dict[str, int]) -> dict[str, np.ndarray]:
    def c(name: str) -> np.ndarray:
        return x[:, index[name], :, :].astype(np.float32, copy=False)

    u10, v10 = c("u10"), c("v10")
    t2m = c("t2m")
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


def main() -> None:
    args = parse_args()
    prepare_output(args.output_dir, args.overwrite)
    logger = setup_logger(args.output_dir)
    rng = np.random.default_rng(args.seed)

    zarr_path = args.dataset_dir / "train.zarr"
    root = open_group(zarr_path)
    xds = root["input"]
    n, c, h, w = map(int, xds.shape)
    channels = list(root.attrs.get("channels", []))
    if not channels:
        meta_path = args.dataset_dir / "metadata.json"
        if meta_path.exists():
            channels = list(json.loads(meta_path.read_text(encoding="utf-8")).get("channels", []))
    if len(channels) != c:
        raise RuntimeError(f"Could not resolve {c} channel names; got {channels}")
    missing = [name for name in REQUIRED_CHANNELS if name not in channels]
    if missing:
        raise RuntimeError(f"Required channels missing: {missing}")
    idx = {name: channels.index(name) for name in channels}

    chunk0 = int(getattr(xds, "chunks", (256,))[0])
    block_size = args.block_size or chunk0
    sampling_source = "phase2_systematic"
    phase1_blocks = args.phase1_dir / "sampling_blocks.parquet"
    if phase1_blocks.exists() and args.block_size == 0:
        p1 = pd.read_parquet(phase1_blocks)
        if "block_id" in p1.columns and len(p1):
            candidate = np.sort(p1["block_id"].dropna().astype(np.int64).unique())
            candidate = candidate[(candidate >= 0) & (candidate < math.ceil(n / block_size))]
            if len(candidate):
                block_ids = candidate
                sampling_source = "phase1_sampling_blocks"
            else:
                block_ids = select_blocks(n, block_size, args.sample_patches, rng)
        else:
            block_ids = select_blocks(n, block_size, args.sample_patches, rng)
    else:
        block_ids = select_blocks(n, block_size, args.sample_patches, rng)
    logger.info("Dataset shape: %s", xds.shape)
    logger.info("Sampling %d blocks of %d patches (%s)", len(block_ids), block_size, sampling_source)

    names = [row["name"] for row in FORMULAS]
    reservoirs: dict[str, list[np.ndarray]] = {name: [] for name in names}
    per_block_quota = max(1, math.ceil(args.reservoir_values / len(block_ids)))
    block_rows: list[dict[str, Any]] = []

    # Running moments over every derived pixel in sampled blocks.
    count = {name: 0 for name in names}
    sums = {name: 0.0 for name in names}
    sumsq = {name: 0.0 for name in names}
    mins = {name: np.inf for name in names}
    maxs = {name: -np.inf for name in names}

    for block_id in block_ids:
        start = int(block_id * block_size)
        end = min(n, start + block_size)
        x = np.asarray(xds[start:end], dtype=np.float32)
        d = derive(x, idx)
        block_rows.append({"block_id": int(block_id), "start_patch": start, "end_patch": end, "patches": end-start})
        for name, arr in d.items():
            vals = arr.reshape(-1)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            count[name] += int(vals.size)
            v64 = vals.astype(np.float64, copy=False)
            sums[name] += float(v64.sum())
            sumsq[name] += float(np.square(v64).sum())
            mins[name] = min(mins[name], float(vals.min()))
            maxs[name] = max(maxs[name], float(vals.max()))
            take = min(per_block_quota, vals.size)
            chosen = rng.choice(vals.size, size=take, replace=False)
            reservoirs[name].append(vals[chosen].astype(np.float32, copy=False))

    pd.DataFrame(block_rows).to_parquet(args.output_dir / "sampling_blocks.parquet", index=False)
    pd.DataFrame(FORMULAS).to_parquet(args.output_dir / "formula_catalog.parquet", index=False)

    final_samples: dict[str, np.ndarray] = {}
    summary_rows: list[dict[str, Any]] = []
    quantile_rows: list[dict[str, Any]] = []
    hist_rows: list[dict[str, Any]] = []

    unit_map = {row["name"]: row["unit"] for row in FORMULAS}
    formula_map = {row["name"]: row["formula"] for row in FORMULAS}

    for name in names:
        vals = np.concatenate(reservoirs[name]) if reservoirs[name] else np.empty(0, dtype=np.float32)
        if vals.size > args.reservoir_values:
            vals = vals[rng.choice(vals.size, args.reservoir_values, replace=False)]
        final_samples[name] = vals.astype(np.float32, copy=False)
        nval = count[name]
        mean = sums[name] / nval if nval else np.nan
        var = max(sumsq[name] / nval - mean * mean, 0.0) if nval else np.nan
        std = math.sqrt(var) if nval else np.nan
        summary_rows.append({
            "name": name, "formula": formula_map[name], "unit": unit_map[name],
            "sampled_pixel_count": nval, "mean": mean, "std": std,
            "min": mins[name] if nval else np.nan, "max": maxs[name] if nval else np.nan,
            "skewness_reservoir": float(skew(vals, bias=False)) if vals.size > 2 else np.nan,
            "excess_kurtosis_reservoir": float(kurtosis(vals, fisher=True, bias=False)) if vals.size > 3 else np.nan,
        })
        for label, q in QUANTILES:
            quantile_rows.append({"name": name, "quantile": label, "probability": q,
                                  "value": float(np.quantile(vals, q)) if vals.size else np.nan})
        if vals.size:
            counts_h, edges = np.histogram(vals, bins=args.hist_bins)
            for i, cnt in enumerate(counts_h):
                hist_rows.append({"name": name, "bin_index": i, "left": float(edges[i]),
                                  "right": float(edges[i+1]), "count": int(cnt)})

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_parquet(args.output_dir / "derived_summary.parquet", index=False)
    pd.DataFrame(quantile_rows).to_parquet(args.output_dir / "derived_quantiles.parquet", index=False)
    pd.DataFrame(hist_rows).to_parquet(args.output_dir / "derived_histograms.parquet", index=False)

    # Correlation matrix computed on equally truncated reservoir lengths.
    min_len = min((len(v) for v in final_samples.values()), default=0)
    if min_len:
        matrix = np.column_stack([final_samples[name][:min_len] for name in names])
        corr = pd.DataFrame(matrix, columns=names).corr(method="pearson")
        corr_long = corr.rename_axis("var_a").reset_index().melt(id_vars="var_a", var_name="var_b", value_name="pearson_r")
        corr_long.to_parquet(args.output_dir / "derived_correlations.parquet", index=False)
    else:
        pd.DataFrame(columns=["var_a", "var_b", "pearson_r"]).to_parquet(args.output_dir / "derived_correlations.parquet", index=False)

    np.savez_compressed(args.output_dir / "derived_samples.npz", **final_samples)

    summary = {
        "phase_version": PHASE_VERSION,
        "dataset_dir": str(args.dataset_dir),
        "dataset_shape": [n, c, h, w],
        "channels": channels,
        "derived_variables": names,
        "sample_patches_requested": int(args.sample_patches),
        "sampling_source": sampling_source,
        "sampled_blocks": int(len(block_ids)),
        "sampled_patches": int(sum(row["patches"] for row in block_rows)),
        "reservoir_values_per_variable": int(args.reservoir_values),
        "note": "Bulk wind differences are vector-wind differences in m/s, not height-normalized shear rates.",
    }
    (args.output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Phase 2 complete: %s", args.output_dir)


if __name__ == "__main__":
    main()
