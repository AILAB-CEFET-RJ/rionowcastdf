#!/usr/bin/env python3
"""CorrDiff Phase 4 v2 — stratified analysis of intense radar events.

Why v2?
-------
The original Phase 4 reused the 300k pixel sample from Phase 3. That sample
represented radar occurrence reasonably well, but substantially
underrepresented the intensity tail (>=25, >=30, >=40, >=45).

v2 reads the CorrDiff Zarr directly and creates a STRATIFIED pixel reservoir
over disjoint radar-intensity strata. Each stratum is sampled independently,
and inverse sampling weights are used so event/non-event statistics reconstruct
the population represented by the sampled Zarr blocks.

The fixed-threshold prevalence from Phase 0 is also loaded when available and
stored as an external exact reference for the full patch-weighted dataset.

Important
---------
- Radar values are numerical legend values after expm1(stored_target).
- The builder does not declare their physical unit; the script does not label
  them dBZ.
- Patch overlap remains present. Statistics describe the training distribution.
- Weighted conditional statistics are descriptive, not causal.

Default disjoint strata
-----------------------
zero
(0, 20)
[20, 25)
[25, 30)
[30, 35)
[35, 40)
[40, 45)
[45, 50)
>= 50

Outputs
-------
<output_dir>/
  analysis_summary.json
  predictor_catalog.parquet
  sampling_blocks.parquet
  stratum_sampling.parquet
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import zarr

PHASE_VERSION = "phase4-extremes-v2-stratified-weighted"

RAW_CHANNELS = [
    "tcwv", "t2m", "u10", "v10",
    "t_850", "r_850", "u_850", "v_850",
    "t_500", "r_500", "u_500", "v_500",
]
DERIVED_NAMES = [
    "wind_speed_10", "wind_speed_850", "wind_speed_500",
    "delta_t_500_850", "delta_t_850_surface", "delta_r_500_850",
    "bulk_wind_diff_10_850", "bulk_wind_diff_850_500",
]
ALL_PREDICTORS = RAW_CHANNELS + DERIVED_NAMES

UNITS = {
    "tcwv": "kg m^-2", "t2m": "K", "u10": "m s^-1", "v10": "m s^-1",
    "t_850": "K", "r_850": "%", "u_850": "m s^-1", "v_850": "m s^-1",
    "t_500": "K", "r_500": "%", "u_500": "m s^-1", "v_500": "m s^-1",
    "wind_speed_10": "m s^-1", "wind_speed_850": "m s^-1",
    "wind_speed_500": "m s^-1", "delta_t_500_850": "K",
    "delta_t_850_surface": "K", "delta_r_500_850": "percentage points",
    "bulk_wind_diff_10_850": "m s^-1",
    "bulk_wind_diff_850_500": "m s^-1",
}

STRATA = [
    ("zero", None, 0.0, True, True),
    ("gt0_lt20", 0.0, 20.0, False, False),
    ("ge20_lt25", 20.0, 25.0, True, False),
    ("ge25_lt30", 25.0, 30.0, True, False),
    ("ge30_lt35", 30.0, 35.0, True, False),
    ("ge35_lt40", 35.0, 40.0, True, False),
    ("ge40_lt45", 40.0, 45.0, True, False),
    ("ge45_lt50", 45.0, 50.0, True, False),
    ("ge50", 50.0, None, True, True),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CorrDiff Phase 4 v2: stratified weighted extreme-event analysis"
    )
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--phase0-dir", type=Path,
                   default=Path("analysis_outputs/00_quality"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("analysis_outputs/04_extremes"))
    p.add_argument(
        "--sample-patches", type=int, default=65536,
        help="Patches distributed systematically through the full Zarr.",
    )
    p.add_argument("--block-size", type=int, default=0)
    p.add_argument(
        "--stratum-size", type=int, default=20000,
        help="Maximum retained pixels per disjoint intensity stratum.",
    )
    p.add_argument("--deciles", type=int, default=10)
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
    logger = logging.getLogger("corrdiff.phase4.v2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(out / "phase4.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def open_group(path: Path) -> Any:
    try:
        return zarr.open_group(str(path), mode="r")
    except Exception:
        return zarr.open(str(path), mode="r")


def derive_from_x(
    x: np.ndarray, index: dict[str, int]
) -> dict[str, np.ndarray]:
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
    ids = np.floor(
        (np.arange(needed) + rng.random(needed)) * width
    ).astype(np.int64)
    ids = np.unique(
        np.clip(ids, 0, total_blocks - 1)
    )
    if ids.size < needed:
        remaining = np.setdiff1d(
            np.arange(total_blocks, dtype=np.int64),
            ids,
            assume_unique=True,
        )
        ids = np.sort(
            np.concatenate([
                ids,
                rng.choice(
                    remaining,
                    needed - ids.size,
                    replace=False,
                ),
            ])
        )
    return ids


def stratum_mask(y: np.ndarray, name: str) -> np.ndarray:
    if name == "zero":
        return y == 0
    if name == "gt0_lt20":
        return (y > 0) & (y < 20)
    if name == "ge20_lt25":
        return (y >= 20) & (y < 25)
    if name == "ge25_lt30":
        return (y >= 25) & (y < 30)
    if name == "ge30_lt35":
        return (y >= 30) & (y < 35)
    if name == "ge35_lt40":
        return (y >= 35) & (y < 40)
    if name == "ge40_lt45":
        return (y >= 40) & (y < 45)
    if name == "ge45_lt50":
        return (y >= 45) & (y < 50)
    if name == "ge50":
        return y >= 50
    raise KeyError(name)


@dataclass
class Reservoir:
    capacity: int
    n_features: int
    rng: np.random.Generator
    total_seen: int = 0
    X: np.ndarray | None = None
    Y: np.ndarray | None = None

    def update(
        self,
        candidate_ids: np.ndarray,
        flat_fields: list[np.ndarray],
        flat_y: np.ndarray,
    ) -> None:
        m = int(candidate_ids.size)
        if m == 0:
            return

        old_total = self.total_seen
        new_total = old_total + m
        target_size = min(self.capacity, new_total)

        if old_total == 0:
            k_new = target_size
            old_keep = np.empty(0, dtype=np.int64)
        else:
            # Number selected from new batch in a uniform sample of target_size
            # from old_total + m population items.
            k_new = int(
                self.rng.hypergeometric(
                    ngood=m,
                    nbad=old_total,
                    nsample=target_size,
                )
            )
            old_needed = target_size - k_new
            old_size = 0 if self.X is None else len(self.X)
            if old_needed > old_size:
                # This should not occur when the prior reservoir is valid.
                old_needed = old_size
                k_new = target_size - old_needed
            old_keep = (
                self.rng.choice(
                    old_size,
                    size=old_needed,
                    replace=False,
                )
                if old_needed > 0
                else np.empty(0, dtype=np.int64)
            )

        if k_new > 0:
            chosen_ids = self.rng.choice(
                candidate_ids,
                size=k_new,
                replace=False,
            )
            new_X = np.column_stack([
                field[chosen_ids] for field in flat_fields
            ]).astype(np.float32, copy=False)
            new_Y = flat_y[chosen_ids].astype(np.float32, copy=False)
        else:
            new_X = np.empty((0, self.n_features), dtype=np.float32)
            new_Y = np.empty(0, dtype=np.float32)

        if self.X is not None and old_keep.size:
            old_X = self.X[old_keep]
            old_Y = self.Y[old_keep]
            self.X = np.concatenate([old_X, new_X], axis=0)
            self.Y = np.concatenate([old_Y, new_Y], axis=0)
        else:
            self.X = new_X
            self.Y = new_Y

        self.total_seen = new_total


def weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(mask):
        return float("nan")
    return float(np.average(x[mask], weights=w[mask]))


def weighted_var(x: np.ndarray, w: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if np.sum(mask) < 2:
        return float("nan")
    xx = x[mask].astype(np.float64)
    ww = w[mask].astype(np.float64)
    mu = np.average(xx, weights=ww)
    sw = ww.sum()
    sw2 = np.square(ww).sum()
    denom = sw - sw2 / sw
    if denom <= 0:
        return float("nan")
    return float(np.sum(ww * np.square(xx - mu)) / denom)


def weighted_std(x: np.ndarray, w: np.ndarray) -> float:
    v = weighted_var(x, w)
    return math.sqrt(v) if np.isfinite(v) and v >= 0 else float("nan")


def weighted_quantile(
    values: np.ndarray,
    quantiles: np.ndarray | list[float] | float,
    weights: np.ndarray,
) -> np.ndarray:
    q = np.atleast_1d(np.asarray(quantiles, dtype=np.float64))
    mask = (
        np.isfinite(values) &
        np.isfinite(weights) &
        (weights > 0)
    )
    if not np.any(mask):
        return np.full(q.shape, np.nan, dtype=np.float64)

    v = values[mask].astype(np.float64)
    w = weights[mask].astype(np.float64)
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cumulative = np.cumsum(w)
    total = cumulative[-1]
    targets = np.clip(q, 0, 1) * total
    idx = np.searchsorted(cumulative, targets, side="left")
    idx = np.clip(idx, 0, len(v) - 1)
    return v[idx]


def effective_sample_size(w: np.ndarray) -> float:
    w = w[np.isfinite(w) & (w > 0)].astype(np.float64)
    if w.size == 0:
        return 0.0
    s1 = w.sum()
    s2 = np.square(w).sum()
    return float((s1 * s1) / s2) if s2 > 0 else 0.0


def weighted_smd(
    event_x: np.ndarray,
    event_w: np.ndarray,
    non_x: np.ndarray,
    non_w: np.ndarray,
) -> float:
    ma = weighted_mean(event_x, event_w)
    mb = weighted_mean(non_x, non_w)
    va = weighted_var(event_x, event_w)
    vb = weighted_var(non_x, non_w)
    if not all(np.isfinite(v) for v in [ma, mb, va, vb]):
        return float("nan")
    pooled = math.sqrt(max((va + vb) / 2.0, 0.0))
    return float((ma - mb) / pooled) if pooled > 0 else float("nan")


def load_phase0_reference(path: Path) -> dict[str, float]:
    file = path / "target_event_rates_global.parquet"
    if not file.exists():
        return {}
    df = pd.read_parquet(file)
    out: dict[str, float] = {}
    if "threshold" in df.columns and "event_pixel_ratio" in df.columns:
        for _, row in df.iterrows():
            out[str(row["threshold"])] = float(row["event_pixel_ratio"])
    return out


def main() -> None:
    args = parse_args()
    prepare_output(args.output_dir, args.overwrite)
    logger = setup_logger(args.output_dir)
    rng = np.random.default_rng(args.seed)

    root = open_group(args.dataset_dir / "train.zarr")
    xds, yds, mds = root["input"], root["target"], root["mask"]
    n = int(xds.shape[0])

    channels = list(root.attrs.get("channels", []))
    if not channels:
        metadata = args.dataset_dir / "metadata.json"
        channels = list(
            json.loads(
                metadata.read_text(encoding="utf-8")
            ).get("channels", [])
        )

    missing = [name for name in RAW_CHANNELS if name not in channels]
    if missing:
        raise RuntimeError(f"Required channels missing: {missing}")
    channel_index = {
        name: channels.index(name) for name in channels
    }

    block_size = args.block_size or int(
        getattr(xds, "chunks", (256,))[0]
    )
    block_ids = select_blocks(
        n, block_size, args.sample_patches, rng
    )

    logger.info(
        "Sampling %d blocks (~%d patches) with stratified reservoirs",
        len(block_ids),
        len(block_ids) * block_size,
    )

    reservoirs = {
        name: Reservoir(
            capacity=args.stratum_size,
            n_features=len(ALL_PREDICTORS),
            rng=rng,
        )
        for name, *_ in STRATA
    }

    block_rows: list[dict[str, Any]] = []
    total_valid_seen = 0

    for block_number, block_id in enumerate(block_ids, start=1):
        start = int(block_id * block_size)
        end = min(n, start + block_size)

        x = np.asarray(xds[start:end], dtype=np.float32)
        target = np.expm1(
            np.asarray(yds[start:end, 0], dtype=np.float32)
        ).astype(np.float32)
        mask = np.asarray(
            mds[start:end, 0], dtype=np.float32
        ) > 0.5

        fields: dict[str, np.ndarray] = {
            name: x[:, channel_index[name], :, :]
            for name in RAW_CHANNELS
        }
        fields.update(derive_from_x(x, channel_index))

        valid = mask & np.isfinite(target)
        for arr in fields.values():
            valid &= np.isfinite(arr)

        flat_y = target.reshape(-1)
        flat_valid = valid.reshape(-1)
        flat_fields = [
            fields[name].reshape(-1) for name in ALL_PREDICTORS
        ]

        valid_count = int(flat_valid.sum())
        total_valid_seen += valid_count
        row = {
            "block_id": int(block_id),
            "start_patch": start,
            "end_patch": end,
            "patches": end - start,
            "valid_pixels": valid_count,
        }

        for stratum_name, *_ in STRATA:
            ids = np.flatnonzero(
                flat_valid & stratum_mask(flat_y, stratum_name)
            )
            row[f"count_{stratum_name}"] = int(ids.size)
            reservoirs[stratum_name].update(
                ids, flat_fields, flat_y
            )

        block_rows.append(row)

        if block_number % 25 == 0 or block_number == len(block_ids):
            logger.info(
                "Processed %d/%d blocks",
                block_number, len(block_ids),
            )

    pd.DataFrame(block_rows).to_parquet(
        args.output_dir / "sampling_blocks.parquet",
        index=False,
    )

    # Concatenate stratum reservoirs and assign inverse sampling weights.
    X_parts: list[np.ndarray] = []
    Y_parts: list[np.ndarray] = []
    W_parts: list[np.ndarray] = []
    S_parts: list[np.ndarray] = []
    stratum_rows: list[dict[str, Any]] = []

    for stratum_name, lower, upper, lower_inclusive, upper_inclusive in STRATA:
        reservoir = reservoirs[stratum_name]
        sample_count = 0 if reservoir.X is None else len(reservoir.X)
        population_count = int(reservoir.total_seen)
        weight = (
            population_count / sample_count
            if sample_count > 0 else np.nan
        )
        sampling_fraction = (
            sample_count / population_count
            if population_count > 0 else np.nan
        )

        stratum_rows.append({
            "stratum": stratum_name,
            "lower": lower,
            "upper": upper,
            "population_count_in_sampled_blocks": population_count,
            "sample_count": sample_count,
            "sampling_fraction": sampling_fraction,
            "inverse_sampling_weight": weight,
        })

        if sample_count:
            X_parts.append(reservoir.X.astype(np.float32))
            Y_parts.append(reservoir.Y.astype(np.float32))
            W_parts.append(
                np.full(sample_count, weight, dtype=np.float64)
            )
            S_parts.append(
                np.full(sample_count, stratum_name, dtype="U32")
            )

    stratum_df = pd.DataFrame(stratum_rows)
    stratum_df.to_parquet(
        args.output_dir / "stratum_sampling.parquet",
        index=False,
    )

    if not X_parts:
        raise RuntimeError("No valid pixels retained in stratified sample")

    X = np.concatenate(X_parts, axis=0)
    Y = np.concatenate(Y_parts, axis=0)
    W = np.concatenate(W_parts, axis=0)
    S = np.concatenate(S_parts, axis=0)

    catalog = pd.DataFrame({
        "predictor": ALL_PREDICTORS,
        "source": [
            "raw" if name in RAW_CHANNELS else "derived"
            for name in ALL_PREDICTORS
        ],
        "unit": [UNITS.get(name, "unknown") for name in ALL_PREDICTORS],
    })
    catalog.to_parquet(
        args.output_dir / "predictor_catalog.parquet",
        index=False,
    )

    # Weighted radar quantiles reconstruct the sampled-block population.
    global_q = weighted_quantile(
        Y, [0.90, 0.95, 0.99], W
    )
    positive_mask = Y > 0
    positive_q = weighted_quantile(
        Y[positive_mask],
        [0.90, 0.95, 0.99],
        W[positive_mask],
    )

    thresholds: list[dict[str, Any]] = []
    for value in [20, 25, 30, 35, 40, 45, 50]:
        thresholds.append({
            "event_id": f"fixed_ge_{value}",
            "family": "fixed",
            "quantile": np.nan,
            "threshold": float(value),
            "operator": ">=",
            "label": f"Radar >= {value}",
        })

    for q, value in zip([0.90, 0.95, 0.99], global_q):
        thresholds.append({
            "event_id": f"global_p{int(q * 100)}",
            "family": "global_quantile",
            "quantile": q,
            "threshold": float(value),
            "operator": ">",
            "label": f"Radar > P{int(q * 100)} global (weighted)",
        })

    for q, value in zip([0.90, 0.95, 0.99], positive_q):
        thresholds.append({
            "event_id": f"positive_p{int(q * 100)}",
            "family": "positive_quantile",
            "quantile": q,
            "threshold": float(value),
            "operator": ">",
            "label": f"Radar > P{int(q * 100)} positive-only (weighted)",
        })

    threshold_df = pd.DataFrame(thresholds)
    threshold_df.to_parquet(
        args.output_dir / "threshold_definitions.parquet",
        index=False,
    )

    phase0_reference = load_phase0_reference(args.phase0_dir)

    prevalence_rows: list[dict[str, Any]] = []
    conditional_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    decile_rows: list[dict[str, Any]] = []

    total_weight = float(W.sum())

    for th in thresholds:
        if th["operator"] == ">=":
            event = Y >= th["threshold"]
        else:
            event = Y > th["threshold"]

        event_weight = float(W[event].sum())
        non_event_weight = float(W[~event].sum())
        weighted_event_rate = (
            event_weight / total_weight
            if total_weight > 0 else np.nan
        )

        reference_key = None
        if th["family"] == "fixed":
            reference_key = (
                "gt_0" if th["threshold"] == 0
                else f"ge_{int(th['threshold'])}"
            )
        reference_rate = (
            phase0_reference.get(reference_key, np.nan)
            if reference_key else np.nan
        )

        reported_rate = (
            reference_rate
            if np.isfinite(reference_rate)
            else weighted_event_rate
        )
        rate_source = (
            "phase0_exact_full_scan"
            if np.isfinite(reference_rate)
            else "phase4_weighted_sampled_blocks"
        )

        prevalence_rows.append({
            **th,
            "sample_n": int(len(Y)),
            "sample_event_count_unweighted": int(event.sum()),
            "sample_non_event_count_unweighted": int((~event).sum()),
            "estimated_population_event_count_in_sampled_blocks": event_weight,
            "estimated_population_non_event_count_in_sampled_blocks": non_event_weight,
            "weighted_event_rate_sampled_blocks": weighted_event_rate,
            "reference_event_rate_phase0": reference_rate,
            "event_rate": reported_rate,
            "event_rate_source": rate_source,
            "effective_sample_size_all": effective_sample_size(W),
            "effective_sample_size_event": effective_sample_size(W[event]),
        })

        for j, name in enumerate(ALL_PREDICTORS):
            values = X[:, j].astype(np.float64)
            finite = np.isfinite(values)
            ev = event & finite
            ne = (~event) & finite

            for group_name, selector in [
                ("event", ev),
                ("non_event", ne),
            ]:
                xv = values[selector]
                wv = W[selector]
                q25, q50, q75, q95 = weighted_quantile(
                    xv, [0.25, 0.50, 0.75, 0.95], wv
                )
                conditional_rows.append({
                    "event_id": th["event_id"],
                    "family": th["family"],
                    "threshold": th["threshold"],
                    "predictor": name,
                    "group": group_name,
                    "sample_n": int(xv.size),
                    "effective_sample_size": effective_sample_size(wv),
                    "estimated_population_weight": float(wv.sum()),
                    "mean": weighted_mean(xv, wv),
                    "std": weighted_std(xv, wv),
                    "p25": float(q25),
                    "median": float(q50),
                    "p75": float(q75),
                    "p95": float(q95),
                })

            a, wa = values[ev], W[ev]
            b, wb = values[ne], W[ne]

            event_mean = weighted_mean(a, wa)
            non_event_mean = weighted_mean(b, wb)
            event_median = weighted_quantile(
                a, 0.5, wa
            )[0]
            non_event_median = weighted_quantile(
                b, 0.5, wb
            )[0]

            effect_rows.append({
                "event_id": th["event_id"],
                "family": th["family"],
                "threshold": th["threshold"],
                "predictor": name,
                "source": (
                    "raw" if name in RAW_CHANNELS else "derived"
                ),
                "unit": UNITS.get(name, "unknown"),
                "n_event_sample": int(a.size),
                "n_non_event_sample": int(b.size),
                "effective_n_event": effective_sample_size(wa),
                "effective_n_non_event": effective_sample_size(wb),
                "event_mean": event_mean,
                "non_event_mean": non_event_mean,
                "mean_difference": event_mean - non_event_mean,
                "event_median": float(event_median),
                "non_event_median": float(non_event_median),
                "median_difference": float(
                    event_median - non_event_median
                ),
                "standardized_mean_difference": weighted_smd(
                    a, wa, b, wb
                ),
                "statistics_weighted": True,
            })

            # Weighted predictor deciles.
            x_all = values[finite]
            w_all = W[finite]
            event_all = event[finite]

            edges = weighted_quantile(
                x_all,
                np.linspace(0, 1, args.deciles + 1),
                w_all,
            )
            edges = np.unique(edges[np.isfinite(edges)])
            if edges.size < 2:
                continue

            codes = np.searchsorted(
                edges[1:-1], x_all, side="right"
            )
            n_bins = int(codes.max()) + 1 if codes.size else 0

            for decile in range(n_bins):
                sel = codes == decile
                if not np.any(sel):
                    continue
                ww = w_all[sel]
                xx = x_all[sel]
                ee = event_all[sel]
                denominator = float(ww.sum())
                event_rate = (
                    float(ww[ee].sum() / denominator)
                    if denominator > 0 else np.nan
                )
                decile_rows.append({
                    "event_id": th["event_id"],
                    "predictor": name,
                    "decile": decile + 1,
                    "sample_n": int(sel.sum()),
                    "effective_sample_size": effective_sample_size(ww),
                    "estimated_population_weight": denominator,
                    "predictor_mean": weighted_mean(xx, ww),
                    "predictor_min": float(np.min(xx)),
                    "predictor_max": float(np.max(xx)),
                    "event_rate": event_rate,
                    "weighted": True,
                })

    prevalence_df = pd.DataFrame(prevalence_rows)
    prevalence_df.to_parquet(
        args.output_dir / "event_prevalence.parquet",
        index=False,
    )
    pd.DataFrame(conditional_rows).to_parquet(
        args.output_dir / "conditional_predictor_stats.parquet",
        index=False,
    )
    pd.DataFrame(effect_rows).to_parquet(
        args.output_dir / "effect_sizes.parquet",
        index=False,
    )
    pd.DataFrame(decile_rows).to_parquet(
        args.output_dir / "event_rate_by_predictor_decile.parquet",
        index=False,
    )

    np.savez_compressed(
        args.output_dir / "extreme_samples.npz",
        predictors=X.astype(np.float32),
        radar=Y.astype(np.float32),
        sample_weight=W.astype(np.float64),
        stratum=S,
        predictor_names=np.asarray(
            ALL_PREDICTORS, dtype="U64"
        ),
    )

    degenerate = prevalence_df.loc[
        prevalence_df["sample_event_count_unweighted"].isin([0, len(Y)]),
        "event_id",
    ].tolist()

    max_reference_abs_diff = np.nan
    fixed = prevalence_df[
        prevalence_df["family"] == "fixed"
    ].copy()
    if (
        "reference_event_rate_phase0" in fixed.columns and
        fixed["reference_event_rate_phase0"].notna().any()
    ):
        diff = (
            fixed["weighted_event_rate_sampled_blocks"] -
            fixed["reference_event_rate_phase0"]
        ).abs()
        if diff.notna().any():
            max_reference_abs_diff = float(diff.max())

    summary = {
        "phase_version": PHASE_VERSION,
        "sample_source": "direct_zarr_stratified_reservoir",
        "sampled_blocks": int(len(block_ids)),
        "sampled_patches": int(
            sum(row["patches"] for row in block_rows)
        ),
        "valid_population_pixels_seen_in_sampled_blocks": int(total_valid_seen),
        "retained_stratified_sample_rows": int(len(Y)),
        "stratum_capacity": int(args.stratum_size),
        "predictor_count": len(ALL_PREDICTORS),
        "positive_radar_rate_weighted_sampled_blocks": float(
            W[Y > 0].sum() / W.sum()
        ),
        "positive_radar_rate_unweighted_retained_sample": float(
            np.mean(Y > 0)
        ),
        "threshold_count": int(len(thresholds)),
        "degenerate_event_definitions": degenerate,
        "phase0_reference_available": bool(phase0_reference),
        "max_abs_fixed_threshold_rate_diff_vs_phase0": max_reference_abs_diff,
        "weighting_method": (
            "inverse stratum sampling fraction; each retained pixel receives "
            "population_count_in_sampled_blocks/sample_count for its stratum"
        ),
        "radar_domain": (
            "expm1(stored_target): post-clip numerical radar-legend values; "
            "physical unit not assumed"
        ),
        "interpretation_note": (
            "Conditional means, quantiles, SMD and predictor-decile event rates "
            "are inverse-sampling-weighted. Fixed-threshold full-dataset "
            "prevalence uses Phase 0 exact rates when available."
        ),
    }

    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info(
        "Phase 4 v2 complete. Retained %d stratified rows from %d valid pixels.",
        len(Y), total_valid_seen,
    )
    logger.info(
        "Max |weighted fixed-event rate - Phase0 exact| = %s",
        max_reference_abs_diff,
    )
    logger.info("Output: %s", args.output_dir)


if __name__ == "__main__":
    main()
