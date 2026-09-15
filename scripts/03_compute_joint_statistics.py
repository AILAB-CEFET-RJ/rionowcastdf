#!/usr/bin/env python3
"""Phase 3: joint ERA5/derived-predictor relationships with radar.

The script samples valid pixel-level observations from the CorrDiff Zarr dataset,
computes associations between predictors and the radar target in the reconstructed
post-log1p radar-legend domain, and produces conditional-bin statistics suitable
for the Phase 3 notebook.

Important interpretation notes
------------------------------
* Radar values are reconstructed with expm1(target). They are called
  "radar legend units" because the builder does not itself declare dBZ.
* Statistics describe the training-patch distribution. Overlapping patches can
  represent central spatial pixels multiple times.
* Associations are descriptive, not causal.

Outputs
-------
<output_dir>/
  analysis_summary.json
  predictor_catalog.parquet
  sampling_blocks.parquet
  association_metrics.parquet
  conditional_bins.parquet
  relationship_samples.npz
  phase3.log
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
    from scipy.stats import pearsonr, spearmanr
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Phase 3 requires scipy: pip install scipy") from exc

try:
    from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Phase 3 requires scikit-learn: pip install scikit-learn") from exc

PHASE_VERSION = "phase3-joint-v1"
RAW_CHANNELS = [
    "tcwv", "t2m", "u10", "v10", "t_850", "r_850", "u_850", "v_850",
    "t_500", "r_500", "u_500", "v_500",
]
DERIVED_NAMES = [
    "wind_speed_10", "wind_speed_850", "wind_speed_500",
    "delta_t_500_850", "delta_t_850_surface", "delta_r_500_850",
    "bulk_wind_diff_10_850", "bulk_wind_diff_850_500",
]
UNITS = {
    "tcwv": "kg m^-2", "t2m": "K", "u10": "m s^-1", "v10": "m s^-1",
    "t_850": "K", "r_850": "%", "u_850": "m s^-1", "v_850": "m s^-1",
    "t_500": "K", "r_500": "%", "u_500": "m s^-1", "v_500": "m s^-1",
    "wind_speed_10": "m s^-1", "wind_speed_850": "m s^-1", "wind_speed_500": "m s^-1",
    "delta_t_500_850": "K", "delta_t_850_surface": "K", "delta_r_500_850": "percentage points",
    "bulk_wind_diff_10_850": "m s^-1", "bulk_wind_diff_850_500": "m s^-1",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CorrDiff Phase 3 joint ERA5-radar statistics")
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("analysis_outputs/03_joint"))
    p.add_argument("--sample-patches", type=int, default=16384)
    p.add_argument("--block-size", type=int, default=0)
    p.add_argument("--sample-pixels", type=int, default=300000,
                   help="Maximum valid pixel observations retained across all predictors.")
    p.add_argument("--mi-sample", type=int, default=60000,
                   help="Maximum observations used for mutual information.")
    p.add_argument("--conditional-bins", type=int, default=10)
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
    logger = logging.getLogger("corrdiff.phase3")
    logger.setLevel(logging.INFO); logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    fh = logging.FileHandler(out / "phase3.log", encoding="utf-8"); fh.setFormatter(fmt); logger.addHandler(fh)
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
    ids = np.floor((np.arange(needed) + rng.random(needed)) * width).astype(np.int64)
    ids = np.unique(np.clip(ids, 0, total_blocks - 1))
    if ids.size < needed:
        remain = np.setdiff1d(np.arange(total_blocks), ids, assume_unique=True)
        ids = np.sort(np.concatenate([ids, rng.choice(remain, needed - ids.size, replace=False)]))
    return ids


def derive_from_x(x: np.ndarray, index: dict[str, int]) -> dict[str, np.ndarray]:
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
        "bulk_wind_diff_10_850": np.hypot(u850-u10, v850-v10),
        "bulk_wind_diff_850_500": np.hypot(u500-u850, v500-v850),
    }


def safe_corr(func, x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")
    try:
        result = func(x, y)
        return float(result.statistic if hasattr(result, "statistic") else result[0])
    except Exception:
        return float("nan")


def mi_reg(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, max_n: int) -> float:
    if x.size < 20 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")
    if x.size > max_n:
        idx = rng.choice(x.size, max_n, replace=False); x = x[idx]; y = y[idx]
    return float(mutual_info_regression(x.reshape(-1, 1), y, random_state=0)[0])


def mi_cls(x: np.ndarray, y_binary: np.ndarray, rng: np.random.Generator, max_n: int) -> float:
    if x.size < 20 or np.nanstd(x) == 0 or np.unique(y_binary).size < 2:
        return float("nan")
    if x.size > max_n:
        idx = rng.choice(x.size, max_n, replace=False); x = x[idx]; y_binary = y_binary[idx]
    return float(mutual_info_classif(x.reshape(-1, 1), y_binary.astype(int), random_state=0)[0])


def main() -> None:
    args = parse_args()
    prepare_output(args.output_dir, args.overwrite)
    logger = setup_logger(args.output_dir)
    rng = np.random.default_rng(args.seed)

    root = open_group(args.dataset_dir / "train.zarr")
    xds, yds, mds = root["input"], root["target"], root["mask"]
    n, c, h, w = map(int, xds.shape)
    channels = list(root.attrs.get("channels", []))
    if not channels:
        meta = json.loads((args.dataset_dir / "metadata.json").read_text(encoding="utf-8"))
        channels = list(meta.get("channels", []))
    missing = [name for name in RAW_CHANNELS if name not in channels]
    if missing:
        raise RuntimeError(f"Required channels missing: {missing}")
    index = {name: channels.index(name) for name in channels}

    chunk0 = int(getattr(xds, "chunks", (256,))[0])
    block_size = args.block_size or chunk0
    block_ids = select_blocks(n, block_size, args.sample_patches, rng)
    quota = max(1, math.ceil(args.sample_pixels / len(block_ids)))

    predictor_names = RAW_CHANNELS + DERIVED_NAMES
    rows_x: list[np.ndarray] = []
    rows_y: list[np.ndarray] = []
    block_rows: list[dict[str, Any]] = []

    for bid in block_ids:
        start = int(bid * block_size); end = min(n, start + block_size)
        x = np.asarray(xds[start:end], dtype=np.float32)
        y = np.asarray(yds[start:end, 0], dtype=np.float32)
        mask = np.asarray(mds[start:end, 0], dtype=np.float32) > 0.5
        radar = np.expm1(y).astype(np.float32, copy=False)
        derived = derive_from_x(x, index)
        raw_fields = {name: x[:, index[name], :, :] for name in RAW_CHANNELS}
        fields = {**raw_fields, **derived}

        valid = mask.copy()
        for arr in fields.values():
            valid &= np.isfinite(arr)
        valid &= np.isfinite(radar)
        flat_valid = np.flatnonzero(valid.reshape(-1))
        if flat_valid.size == 0:
            continue
        take = min(quota, flat_valid.size)
        chosen_flat = rng.choice(flat_valid, size=take, replace=False)
        matrix = np.column_stack([fields[name].reshape(-1)[chosen_flat] for name in predictor_names]).astype(np.float32)
        target = radar.reshape(-1)[chosen_flat].astype(np.float32)
        rows_x.append(matrix); rows_y.append(target)
        block_rows.append({"block_id": int(bid), "start_patch": start, "end_patch": end,
                           "patches": end-start, "valid_pixels": int(flat_valid.size), "retained_pixels": int(take)})

    if not rows_x:
        raise RuntimeError("No valid pixel observations were sampled")
    X = np.concatenate(rows_x, axis=0)
    Y = np.concatenate(rows_y, axis=0)
    if X.shape[0] > args.sample_pixels:
        take = rng.choice(X.shape[0], args.sample_pixels, replace=False)
        X, Y = X[take], Y[take]

    pd.DataFrame(block_rows).to_parquet(args.output_dir / "sampling_blocks.parquet", index=False)
    catalog = pd.DataFrame([
        {"predictor": name, "source": "raw" if name in RAW_CHANNELS else "derived", "unit": UNITS.get(name, "unknown")}
        for name in predictor_names
    ])
    catalog.to_parquet(args.output_dir / "predictor_catalog.parquet", index=False)

    event = Y > 0
    metrics: list[dict[str, Any]] = []
    for j, name in enumerate(predictor_names):
        x = X[:, j].astype(np.float64)
        y = Y.astype(np.float64)
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        ev = y > 0
        xp, yp = x[ev], y[ev]
        metrics.append({
            "predictor": name,
            "source": "raw" if name in RAW_CHANNELS else "derived",
            "unit": UNITS.get(name, "unknown"),
            "n_all": int(x.size),
            "pearson_r": safe_corr(pearsonr, x, y),
            "spearman_rho": safe_corr(spearmanr, x, y),
            "mutual_information_regression": mi_reg(x, y, rng, args.mi_sample),
            "positive_event_rate": float(ev.mean()) if ev.size else np.nan,
            "point_biserial_r_positive": safe_corr(pearsonr, x, ev.astype(float)),
            "mutual_information_positive_event": mi_cls(x, ev, rng, args.mi_sample),
            "n_positive": int(xp.size),
            "pearson_r_positive": safe_corr(pearsonr, xp, yp),
            "spearman_rho_positive": safe_corr(spearmanr, xp, yp),
            "mutual_information_regression_positive": mi_reg(xp, yp, rng, args.mi_sample),
        })
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_parquet(args.output_dir / "association_metrics.parquet", index=False)

    bin_rows: list[dict[str, Any]] = []
    for j, name in enumerate(predictor_names):
        x = X[:, j].astype(np.float64)
        y = Y.astype(np.float64)
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        try:
            cats, edges = pd.qcut(x, q=args.conditional_bins, retbins=True, duplicates="drop")
            codes = cats.codes
        except Exception:
            edges = np.linspace(np.nanmin(x), np.nanmax(x), args.conditional_bins + 1)
            codes = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges)-2)
        for b in range(max(codes)+1 if codes.size else 0):
            sel = codes == b
            if not np.any(sel):
                continue
            xb, yb = x[sel], y[sel]
            bin_rows.append({
                "predictor": name, "bin_index": int(b), "n": int(sel.sum()),
                "x_min": float(np.min(xb)), "x_max": float(np.max(xb)), "x_mean": float(np.mean(xb)),
                "radar_mean": float(np.mean(yb)), "radar_median": float(np.median(yb)),
                "radar_p95": float(np.quantile(yb, 0.95)),
                "positive_rate": float(np.mean(yb > 0)),
                "ge20_rate": float(np.mean(yb >= 20)),
                "ge30_rate": float(np.mean(yb >= 30)),
                "ge40_rate": float(np.mean(yb >= 40)),
            })
    pd.DataFrame(bin_rows).to_parquet(args.output_dir / "conditional_bins.parquet", index=False)

    np.savez_compressed(args.output_dir / "relationship_samples.npz",
                        predictors=X.astype(np.float32), radar=Y.astype(np.float32),
                        predictor_names=np.asarray(predictor_names, dtype="U64"))

    summary = {
        "phase_version": PHASE_VERSION,
        "dataset_dir": str(args.dataset_dir),
        "dataset_shape": [n, c, h, w],
        "sampled_blocks": int(len(block_ids)),
        "retained_pixel_observations": int(X.shape[0]),
        "predictor_count": int(X.shape[1]),
        "raw_predictor_count": len(RAW_CHANNELS),
        "derived_predictor_count": len(DERIVED_NAMES),
        "radar_domain": "expm1(stored_target): post-clip numerical radar-legend values; physical unit not assumed",
        "positive_event_rate_in_sample": float(event.mean()),
        "note": "Associations are descriptive and represent overlapping training patches.",
    }
    (args.output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Phase 3 complete: %s", args.output_dir)


if __name__ == "__main__":
    main()
