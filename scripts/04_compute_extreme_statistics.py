#!/usr/bin/env python3
"""Phase 4: atmospheric conditions associated with intense radar events.

This phase builds on Phase 3 relationship samples when available. It defines
several radar-event thresholds, compares predictor distributions between event
and non-event observations, and estimates event probability across predictor
quantile bins.

Event families
--------------
1. Fixed radar-legend thresholds: >=20, >=25, >=30, >=35, >=40, >=45, >=50.
2. Global radar quantiles: >P90, >P95, >P99 over all sampled valid radar pixels.
3. Positive-radar quantiles: >P90+, >P95+, >P99+ computed only where radar > 0.

Because the source builder does not declare the physical unit of the radar
legend, thresholds are reported as "radar legend units" rather than dBZ.

Outputs
-------
<output_dir>/
  analysis_summary.json
  predictor_catalog.parquet
  threshold_definitions.parquet
  event_prevalence.parquet
  conditional_predictor_stats.parquet
  effect_sizes.parquet
  event_rate_by_predictor_decile.parquet
  extreme_samples.npz
  phase4.log
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

PHASE_VERSION = "phase4-extremes-v1"
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
    p = argparse.ArgumentParser(description="CorrDiff Phase 4 extreme-event statistics")
    p.add_argument("--dataset-dir", type=Path, required=True,
                   help="Used only when Phase 3 samples are unavailable.")
    p.add_argument("--phase3-dir", type=Path, default=Path("analysis_outputs/03_joint"))
    p.add_argument("--output-dir", type=Path, default=Path("analysis_outputs/04_extremes"))
    p.add_argument("--sample-patches", type=int, default=16384)
    p.add_argument("--sample-pixels", type=int, default=300000)
    p.add_argument("--block-size", type=int, default=0)
    p.add_argument("--deciles", type=int, default=10)
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
    logger = logging.getLogger("corrdiff.phase4")
    logger.setLevel(logging.INFO); logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    fh = logging.FileHandler(out / "phase4.log", encoding="utf-8"); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger


def open_group(path: Path) -> Any:
    try:
        return zarr.open_group(str(path), mode="r")
    except Exception:
        return zarr.open(str(path), mode="r")


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


def select_blocks(n: int, block_size: int, sample_patches: int, rng: np.random.Generator) -> np.ndarray:
    total_blocks = math.ceil(n / block_size)
    needed = min(total_blocks, max(1, math.ceil(sample_patches / block_size)))
    if needed == total_blocks:
        return np.arange(total_blocks, dtype=np.int64)
    width = total_blocks / needed
    ids = np.floor((np.arange(needed) + rng.random(needed)) * width).astype(np.int64)
    ids = np.unique(np.clip(ids, 0, total_blocks - 1))
    if ids.size < needed:
        rem = np.setdiff1d(np.arange(total_blocks), ids, assume_unique=True)
        ids = np.sort(np.concatenate([ids, rng.choice(rem, needed-ids.size, replace=False)]))
    return ids


def load_or_sample(args: argparse.Namespace, logger: logging.Logger, rng: np.random.Generator):
    phase3_npz = args.phase3_dir / "relationship_samples.npz"
    phase3_catalog = args.phase3_dir / "predictor_catalog.parquet"
    if phase3_npz.exists():
        data = np.load(phase3_npz, allow_pickle=False)
        X = np.asarray(data["predictors"], dtype=np.float32)
        Y = np.asarray(data["radar"], dtype=np.float32)
        names = [str(v) for v in data["predictor_names"].tolist()]
        logger.info("Reusing Phase 3 relationship samples: %d observations", len(Y))
        if phase3_catalog.exists():
            catalog = pd.read_parquet(phase3_catalog)
        else:
            catalog = pd.DataFrame({"predictor": names,
                                    "source": ["raw" if n in RAW_CHANNELS else "derived" for n in names],
                                    "unit": [UNITS.get(n, "unknown") for n in names]})
        return X, Y, names, catalog, "phase3_relationship_samples"

    root = open_group(args.dataset_dir / "train.zarr")
    xds, yds, mds = root["input"], root["target"], root["mask"]
    n = int(xds.shape[0])
    channels = list(root.attrs.get("channels", []))
    if not channels:
        channels = list(json.loads((args.dataset_dir / "metadata.json").read_text(encoding="utf-8")).get("channels", []))
    missing = [n for n in RAW_CHANNELS if n not in channels]
    if missing:
        raise RuntimeError(f"Required channels missing: {missing}")
    index = {name: channels.index(name) for name in channels}
    block_size = args.block_size or int(getattr(xds, "chunks", (256,))[0])
    block_ids = select_blocks(n, block_size, args.sample_patches, rng)
    quota = max(1, math.ceil(args.sample_pixels / len(block_ids)))
    names = RAW_CHANNELS + DERIVED_NAMES
    xs, ys = [], []
    for bid in block_ids:
        start = int(bid*block_size); end = min(n, start+block_size)
        x = np.asarray(xds[start:end], dtype=np.float32)
        y = np.expm1(np.asarray(yds[start:end, 0], dtype=np.float32)).astype(np.float32)
        mask = np.asarray(mds[start:end, 0], dtype=np.float32) > 0.5
        fields = {name: x[:, index[name], :, :] for name in RAW_CHANNELS}
        fields.update(derive_from_x(x, index))
        valid = mask & np.isfinite(y)
        for arr in fields.values(): valid &= np.isfinite(arr)
        ids = np.flatnonzero(valid.reshape(-1))
        if ids.size == 0: continue
        take = min(quota, ids.size); sel = rng.choice(ids, take, replace=False)
        xs.append(np.column_stack([fields[name].reshape(-1)[sel] for name in names]).astype(np.float32))
        ys.append(y.reshape(-1)[sel].astype(np.float32))
    if not xs: raise RuntimeError("No valid observations sampled")
    X, Y = np.concatenate(xs), np.concatenate(ys)
    if len(Y) > args.sample_pixels:
        keep = rng.choice(len(Y), args.sample_pixels, replace=False); X, Y = X[keep], Y[keep]
    catalog = pd.DataFrame({"predictor": names,
                            "source": ["raw" if n in RAW_CHANNELS else "derived" for n in names],
                            "unit": [UNITS.get(n, "unknown") for n in names]})
    return X, Y, names, catalog, "direct_zarr_sample"


def pooled_smd(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2: return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    denom_n = a.size + b.size - 2
    if denom_n <= 0: return float("nan")
    pooled = math.sqrt(max(((a.size-1)*va + (b.size-1)*vb) / denom_n, 0.0))
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else float("nan")


def main() -> None:
    args = parse_args()
    prepare_output(args.output_dir, args.overwrite)
    logger = setup_logger(args.output_dir)
    rng = np.random.default_rng(args.seed)

    X, Y, names, catalog, source = load_or_sample(args, logger, rng)
    catalog.to_parquet(args.output_dir / "predictor_catalog.parquet", index=False)

    thresholds: list[dict[str, Any]] = []
    for v in [20, 25, 30, 35, 40, 45, 50]:
        thresholds.append({"event_id": f"fixed_ge_{v}", "family": "fixed", "quantile": np.nan,
                           "threshold": float(v), "operator": ">=", "label": f"Radar >= {v}"})
    for q in [0.90, 0.95, 0.99]:
        val = float(np.quantile(Y, q))
        thresholds.append({"event_id": f"global_p{int(q*100)}", "family": "global_quantile", "quantile": q,
                           "threshold": val, "operator": ">", "label": f"Radar > P{int(q*100)} global"})
    positive = Y[Y > 0]
    if positive.size:
        for q in [0.90, 0.95, 0.99]:
            val = float(np.quantile(positive, q))
            thresholds.append({"event_id": f"positive_p{int(q*100)}", "family": "positive_quantile", "quantile": q,
                               "threshold": val, "operator": ">", "label": f"Radar > P{int(q*100)} dos valores positivos"})

    threshold_df = pd.DataFrame(thresholds)
    prevalence_rows = []
    cond_rows = []
    effect_rows = []
    decile_rows = []

    for th in thresholds:
        event = Y >= th["threshold"] if th["operator"] == ">=" else Y > th["threshold"]
        prevalence_rows.append({**th, "n": int(Y.size), "event_count": int(event.sum()), "event_rate": float(event.mean()),
                                "non_event_count": int((~event).sum())})
        for j, name in enumerate(names):
            x = X[:, j].astype(np.float64)
            finite = np.isfinite(x)
            ev = event & finite; ne = (~event) & finite
            a, b = x[ev], x[ne]
            for group, vals in [("event", a), ("non_event", b)]:
                cond_rows.append({
                    "event_id": th["event_id"], "family": th["family"], "threshold": th["threshold"],
                    "predictor": name, "group": group, "n": int(vals.size),
                    "mean": float(np.mean(vals)) if vals.size else np.nan,
                    "std": float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan,
                    "p25": float(np.quantile(vals, .25)) if vals.size else np.nan,
                    "median": float(np.median(vals)) if vals.size else np.nan,
                    "p75": float(np.quantile(vals, .75)) if vals.size else np.nan,
                    "p95": float(np.quantile(vals, .95)) if vals.size else np.nan,
                })
            effect_rows.append({
                "event_id": th["event_id"], "family": th["family"], "threshold": th["threshold"],
                "predictor": name, "source": "raw" if name in RAW_CHANNELS else "derived",
                "unit": UNITS.get(name, "unknown"), "n_event": int(a.size), "n_non_event": int(b.size),
                "event_mean": float(np.mean(a)) if a.size else np.nan,
                "non_event_mean": float(np.mean(b)) if b.size else np.nan,
                "mean_difference": float(np.mean(a)-np.mean(b)) if a.size and b.size else np.nan,
                "event_median": float(np.median(a)) if a.size else np.nan,
                "non_event_median": float(np.median(b)) if b.size else np.nan,
                "median_difference": float(np.median(a)-np.median(b)) if a.size and b.size else np.nan,
                "standardized_mean_difference": pooled_smd(a, b),
            })

            # Predictor quantile-bin response curve.
            xf = x[finite]; ef = event[finite]
            if xf.size < args.deciles:
                continue
            try:
                cats = pd.qcut(xf, q=args.deciles, duplicates="drop")
                codes = cats.codes
            except Exception:
                continue
            for dec in range(max(codes)+1 if codes.size else 0):
                s = codes == dec
                if not np.any(s): continue
                decile_rows.append({
                    "event_id": th["event_id"], "predictor": name, "decile": int(dec+1), "n": int(s.sum()),
                    "predictor_mean": float(np.mean(xf[s])), "predictor_min": float(np.min(xf[s])),
                    "predictor_max": float(np.max(xf[s])), "event_rate": float(np.mean(ef[s])),
                })

    threshold_df.to_parquet(args.output_dir / "threshold_definitions.parquet", index=False)
    pd.DataFrame(prevalence_rows).to_parquet(args.output_dir / "event_prevalence.parquet", index=False)
    pd.DataFrame(cond_rows).to_parquet(args.output_dir / "conditional_predictor_stats.parquet", index=False)
    pd.DataFrame(effect_rows).to_parquet(args.output_dir / "effect_sizes.parquet", index=False)
    pd.DataFrame(decile_rows).to_parquet(args.output_dir / "event_rate_by_predictor_decile.parquet", index=False)
    np.savez_compressed(args.output_dir / "extreme_samples.npz", predictors=X.astype(np.float32), radar=Y.astype(np.float32),
                        predictor_names=np.asarray(names, dtype="U64"))

    degenerate = [r["event_id"] for r in prevalence_rows if r["event_rate"] in (0.0, 1.0)]
    summary = {
        "phase_version": PHASE_VERSION, "sample_source": source,
        "observations": int(len(Y)), "predictor_count": len(names),
        "positive_radar_rate": float(np.mean(Y > 0)),
        "threshold_count": len(thresholds), "degenerate_event_definitions": degenerate,
        "radar_domain": "post-log1p inverse using expm1; numerical radar-legend units, not assumed dBZ",
        "interpretation_note": "Global P90/P95 can equal zero in a highly sparse target; positive-only quantiles are included to characterize event intensity separately.",
    }
    (args.output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Phase 4 complete: %s", args.output_dir)


if __name__ == "__main__":
    main()
