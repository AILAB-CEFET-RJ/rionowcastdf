#!/usr/bin/env python3
"""
CorrDiff - Fase 12
Análise multivariada
====================

Análise multivariada em nível de patch com amostragem estratificada,
diagnóstico de colinearidade/PCA e triagem temporal forward-chaining.

Esta fase não substitui:
- Fase 15: split formal;
- Fase 16: baselines finais;
- Fase 17: ablações formais.
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

try:
    import zarr
except ImportError as exc:
    raise SystemExit("Fase 12 requer zarr: pip install zarr") from exc

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import average_precision_score, roc_auc_score
except ImportError as exc:
    raise SystemExit(
        "Fase 12 requer scikit-learn: pip install scikit-learn"
    ) from exc


PHASE_VERSION = "phase12-multivariate-v1-forward-temporal-weighted"

CANONICAL_RAW_CHANNELS = [
    "tcwv", "t2m", "u10", "v10",
    "t_850", "r_850", "u_850", "v_850",
    "t_500", "r_500", "u_500", "v_500",
]

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

EVENT_THRESHOLDS = {
    "gt_0": 0.0,
    "ge_20": 20.0,
    "ge_30": 30.0,
    "ge_40": 40.0,
    "ge_45": 45.0,
}

BINARY_TARGETS = list(EVENT_THRESHOLDS.keys())

CONTINUOUS_TARGETS = [
    "max_target_log1p",
    "positive_pixel_fraction",
    "event_pixel_fraction_ge_30",
    "event_pixel_fraction_ge_40",
    "event_pixel_fraction_ge_45",
]

STRATUM_NAMES = {
    0: "dry",
    1: "gt0_lt20",
    2: "ge20_lt30",
    3: "ge30_lt40",
    4: "ge40_lt45",
    5: "ge45",
}

SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]

# Screening temporal; Phase 15 continua responsável pelo split formal.
DEFAULT_FORWARD_FOLDS = [
    ("F1", 2014, 2015, 2016),
    ("F2", 2016, 2017, 2018),
    ("F3", 2018, 2019, 2020),
    ("F4", 2020, 2021, 2022),
    ("F5", 2022, 2023, 2024),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CorrDiff Fase 12: análise multivariada."
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("datasets/corrdiff_2011_2024"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs/12_multivariate"),
    )
    p.add_argument("--sample-per-stratum", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--local-timezone", default="America/Sao_Paulo")
    p.add_argument("--channel-names", default=None)
    p.add_argument("--logistic-c", type=float, default=1.0)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--hist-learning-rate", type=float, default=0.05)
    p.add_argument("--hist-max-iter", type=int, default=200)
    p.add_argument("--hist-max-leaf-nodes", type=int, default=15)
    p.add_argument("--save-feature-table", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def setup_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {path}; use --overwrite"
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def setup_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("corrdiff.phase12")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(path / "phase12.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def open_zarr_group(path: Path):
    try:
        return zarr.open_group(str(path), mode="r")
    except Exception:
        return zarr.open(str(path), mode="r")


def infer_datetime_unit(values: np.ndarray) -> str:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "ns"
    magnitude = float(np.nanmedian(np.abs(finite.astype(np.float64))))
    if magnitude >= 1e17:
        return "ns"
    if magnitude >= 1e14:
        return "us"
    if magnitude >= 1e11:
        return "ms"
    return "s"


def to_datetime_utc(values: np.ndarray) -> pd.DatetimeIndex:
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.datetime64):
        return pd.DatetimeIndex(pd.to_datetime(arr, utc=True))
    return pd.DatetimeIndex(
        pd.to_datetime(arr, unit=infer_datetime_unit(arr), utc=True)
    )


def season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def build_patch_time_metadata(
    timestamps_raw: np.ndarray,
    timezone: str,
) -> pd.DataFrame:
    utc = to_datetime_utc(timestamps_raw)
    local = utc.tz_convert(timezone)
    months = np.asarray(local.month, dtype=np.int8)
    return pd.DataFrame({
        "timestamp_utc": utc,
        "year_utc": np.asarray(utc.year, dtype=np.int16),
        "month_local": months,
        "hour_local": np.asarray(local.hour, dtype=np.int8),
        "season_code": pd.Categorical(
            [season_from_month(int(m)) for m in months],
            categories=SEASON_ORDER,
            ordered=True,
        ),
    })


def resolve_raw_channel_names(
    input_array,
    user_names: str | None,
) -> tuple[list[str], str]:
    n_channels = int(input_array.shape[1])

    if user_names:
        names = [x.strip() for x in user_names.split(",") if x.strip()]
        if len(names) != n_channels:
            raise ValueError(
                f"--channel-names supplied {len(names)} names, but input "
                f"has {n_channels} channels."
            )
        return names, "cli"

    attrs = dict(getattr(input_array, "attrs", {}))
    for key in (
        "channel_names",
        "channels",
        "variables",
        "input_variables",
    ):
        value = attrs.get(key)
        if isinstance(value, (list, tuple)) and len(value) == n_channels:
            return [str(x) for x in value], f"zarr_attr:{key}"

    if n_channels == len(CANONICAL_RAW_CHANNELS):
        return list(CANONICAL_RAW_CHANNELS), "canonical_12_channel_fallback"

    raise RuntimeError(
        "Could not infer input channel names. Pass --channel-names explicitly."
    )


def validate_required_raw_channels(names: list[str]) -> None:
    missing = sorted(set(CANONICAL_RAW_CHANNELS) - set(names))
    if missing:
        raise RuntimeError(
            "Derived predictors require canonical raw fields. "
            f"Missing: {missing}"
        )


def predictor_family(name: str) -> str:
    if name in {"tcwv", "r_850", "r_500", "delta_r_500_850"}:
        return "moisture"
    if name in {
        "t2m",
        "t_850",
        "t_500",
        "delta_t_500_850",
        "delta_t_850_surface",
    }:
        return "thermal"
    return "dynamics"


def predictor_metadata(
    raw_names: list[str],
    source: str,
) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(raw_names):
        rows.append({
            "predictor": name,
            "source": "raw",
            "input_channel_index": i,
            "channel_name_source": source,
            "definition": name,
            "family": predictor_family(name),
        })

    defs = {
        "wind_speed_10": "sqrt(u10^2 + v10^2)",
        "wind_speed_850": "sqrt(u_850^2 + v_850^2)",
        "wind_speed_500": "sqrt(u_500^2 + v_500^2)",
        "delta_t_500_850": "t_500 - t_850",
        "delta_t_850_surface": "t_850 - t2m",
        "delta_r_500_850": "r_500 - r_850",
        "bulk_wind_diff_10_850": (
            "sqrt((u_850-u10)^2 + (v_850-v10)^2)"
        ),
        "bulk_wind_diff_850_500": (
            "sqrt((u_500-u_850)^2 + (v_500-v_850)^2)"
        ),
    }

    for name in DERIVED_CHANNELS:
        rows.append({
            "predictor": name,
            "source": "derived",
            "input_channel_index": np.nan,
            "channel_name_source": source,
            "definition": defs[name],
            "family": predictor_family(name),
        })
    return pd.DataFrame(rows)


def stored_threshold(dbz: float) -> np.float32:
    return np.float32(np.log1p(np.float32(dbz)))


STORED_THRESHOLDS = {
    event_id: stored_threshold(dbz)
    for event_id, dbz in EVENT_THRESHOLDS.items()
    if dbz > 0
}


def event_mask(stored_target: np.ndarray, event_id: str) -> np.ndarray:
    if event_id == "gt_0":
        return stored_target > np.float32(0.0)
    return stored_target >= STORED_THRESHOLDS[event_id]


def classify_patch_strata(
    target,
    logger: logging.Logger,
) -> tuple[np.ndarray, pd.DataFrame]:
    n = int(target.shape[0])
    strata = np.empty(n, dtype=np.uint8)
    chunk0 = int(target.chunks[0]) if getattr(target, "chunks", None) else 256

    t20 = STORED_THRESHOLDS["ge_20"]
    t30 = STORED_THRESHOLDS["ge_30"]
    t40 = STORED_THRESHOLDS["ge_40"]
    t45 = STORED_THRESHOLDS["ge_45"]

    logger.info("=" * 80)
    logger.info("PHASE 12 - PASS A - TARGET STRATA")
    logger.info("Target patches : %d", n)
    logger.info("=" * 80)

    for start in range(0, n, chunk0):
        end = min(n, start + chunk0)
        block = np.asarray(target[start:end], dtype=np.float32)
        mx = np.max(block.reshape(end - start, -1), axis=1)

        s = np.zeros(end - start, dtype=np.uint8)
        s[mx > 0] = 1
        s[mx >= t20] = 2
        s[mx >= t30] = 3
        s[mx >= t40] = 4
        s[mx >= t45] = 5
        strata[start:end] = s

        if end == n or end % max(chunk0 * 200, 1) == 0:
            logger.info("PASS A: %d/%d patches", end, n)

    counts = np.bincount(strata, minlength=6)
    rows = [
        {
            "stratum_id": sid,
            "stratum": STRATUM_NAMES[sid],
            "patch_count": int(counts[sid]),
            "patch_ratio": float(counts[sid] / n),
        }
        for sid in range(6)
    ]
    return strata, pd.DataFrame(rows)


def build_sample_manifest(
    strata: np.ndarray,
    time_meta: pd.DataFrame,
    sample_per_stratum: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for sid in range(6):
        candidates = np.flatnonzero(strata == sid)
        n_total = int(len(candidates))
        n_sample = min(sample_per_stratum, n_total)
        if n_sample == 0:
            continue

        if n_sample == n_total:
            chosen = candidates
        else:
            chosen = np.sort(
                rng.choice(candidates, size=n_sample, replace=False)
            )

        weight = n_total / n_sample

        for idx in chosen:
            rows.append({
                "patch_index": int(idx),
                "stratum_id": sid,
                "stratum": STRATUM_NAMES[sid],
                "stratum_patch_count": n_total,
                "stratum_sample_count": n_sample,
                "sampling_fraction": n_sample / n_total,
                "sampling_weight": float(weight),
            })

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise RuntimeError("No patches sampled.")

    idx = manifest["patch_index"].to_numpy(dtype=np.int64)
    tm = time_meta.iloc[idx].reset_index(drop=True)
    return pd.concat(
        [manifest.reset_index(drop=True), tm],
        axis=1,
    )


def derive_predictors(
    raw: np.ndarray,
    raw_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    index = {name: i for i, name in enumerate(raw_names)}

    def v(name: str) -> np.ndarray:
        return raw[:, index[name]]

    derived = [
        np.hypot(v("u10"), v("v10")),
        np.hypot(v("u_850"), v("v_850")),
        np.hypot(v("u_500"), v("v_500")),
        v("t_500") - v("t_850"),
        v("t_850") - v("t2m"),
        v("r_500") - v("r_850"),
        np.hypot(v("u_850") - v("u10"), v("v_850") - v("v10")),
        np.hypot(v("u_500") - v("u_850"), v("v_500") - v("v_850")),
    ]

    fields = np.concatenate(
        [
            raw.astype(np.float32, copy=False),
            np.stack(derived, axis=1).astype(np.float32, copy=False),
        ],
        axis=1,
    )
    return fields, raw_names + DERIVED_CHANNELS


def extract_patch_features(
    predictors: np.ndarray,
    predictor_names: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    means = predictors.mean(axis=(2, 3), dtype=np.float64)
    stds = predictors.std(axis=(2, 3), ddof=0, dtype=np.float64)

    mean_names = [f"{name}__mean" for name in predictor_names]
    mean_std_names = mean_names + [
        f"{name}__std" for name in predictor_names
    ]
    return (
        means,
        np.concatenate([means, stds], axis=1),
        mean_names,
        mean_std_names,
    )


def extract_targets(
    target_block: np.ndarray,
) -> dict[str, np.ndarray]:
    t = np.asarray(target_block, dtype=np.float32)
    if t.ndim == 4 and t.shape[1] == 1:
        t = t[:, 0]
    if t.ndim != 3:
        raise RuntimeError(f"Unexpected target shape: {target_block.shape}")

    flat = t.reshape(t.shape[0], -1)
    out: dict[str, np.ndarray] = {}

    for event_id in BINARY_TARGETS:
        out[event_id] = (
            event_mask(t, event_id)
            .reshape(t.shape[0], -1)
            .any(axis=1)
            .astype(np.int8)
        )

    out["max_target_log1p"] = flat.max(axis=1).astype(np.float64)
    out["positive_pixel_fraction"] = (
        (flat > 0).mean(axis=1).astype(np.float64)
    )
    out["event_pixel_fraction_ge_30"] = (
        (flat >= STORED_THRESHOLDS["ge_30"])
        .mean(axis=1)
        .astype(np.float64)
    )
    out["event_pixel_fraction_ge_40"] = (
        (flat >= STORED_THRESHOLDS["ge_40"])
        .mean(axis=1)
        .astype(np.float64)
    )
    out["event_pixel_fraction_ge_45"] = (
        (flat >= STORED_THRESHOLDS["ge_45"])
        .mean(axis=1)
        .astype(np.float64)
    )
    return out


def weighted_mean_std(
    x: np.ndarray,
    w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    sw = w.sum()
    mean = np.sum(x * w[:, None], axis=0) / sw
    var = np.sum(w[:, None] * (x - mean) ** 2, axis=0) / sw
    std = np.sqrt(np.maximum(var, 0.0))
    std = np.where(std > 1e-12, std, 1.0)
    return mean, std


def standardize_train_eval(
    x_train: np.ndarray,
    x_eval: np.ndarray,
    w_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean, std = weighted_mean_std(x_train, w_train)
    return (x_train - mean) / std, (x_eval - mean) / std


def weighted_corr_matrix(
    x: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    mean, std = weighted_mean_std(x, w)
    z = (x - mean) / std
    cov = (z * w[:, None]).T @ z / w.sum()
    cov = np.clip(cov, -1.0, 1.0)
    np.fill_diagonal(cov, 1.0)
    return cov


def weighted_pca(
    x: np.ndarray,
    w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    corr = weighted_corr_matrix(x, w)
    eigvals, eigvecs = np.linalg.eigh(corr)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]
    total = eigvals.sum()
    ratio = eigvals / total if total > 0 else np.full_like(eigvals, np.nan)
    return eigvals, ratio, eigvecs


def build_forward_fold_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold_id, train_end, eval_start, eval_end in DEFAULT_FORWARD_FOLDS:
        train_mask = df["year_utc"] <= train_end
        eval_mask = (
            (df["year_utc"] >= eval_start)
            & (df["year_utc"] <= eval_end)
        )
        rows.append({
            "fold_id": fold_id,
            "train_start_year": int(df.loc[train_mask, "year_utc"].min())
            if train_mask.any() else np.nan,
            "train_end_year": train_end,
            "eval_start_year": eval_start,
            "eval_end_year": eval_end,
            "n_train_rows": int(train_mask.sum()),
            "n_eval_rows": int(eval_mask.sum()),
            "weighted_train_mass": float(
                df.loc[train_mask, "sampling_weight"].sum()
            ),
            "weighted_eval_mass": float(
                df.loc[eval_mask, "sampling_weight"].sum()
            ),
            "n_train_timestamps": int(
                df.loc[train_mask, "timestamp_utc"].nunique()
            ),
            "n_eval_timestamps": int(
                df.loc[eval_mask, "timestamp_utc"].nunique()
            ),
        })
    return pd.DataFrame(rows)


def month_hour_climatology(
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    eval_df: pd.DataFrame,
    w_train: np.ndarray,
) -> tuple[np.ndarray, float]:
    tmp = pd.DataFrame({
        "month_local": train_df["month_local"].to_numpy(),
        "hour_local": train_df["hour_local"].to_numpy(),
        "y": np.asarray(y_train, dtype=np.float64),
        "w": np.asarray(w_train, dtype=np.float64),
    })
    tmp["wy"] = tmp["w"] * tmp["y"]

    grouped = (
        tmp.groupby(["month_local", "hour_local"], observed=True)
        .agg(sum_w=("w", "sum"), sum_wy=("wy", "sum"))
    )
    grouped["climatology"] = grouped["sum_wy"] / grouped["sum_w"]

    global_mean = float(
        np.average(
            np.asarray(y_train, dtype=np.float64),
            weights=np.asarray(w_train, dtype=np.float64),
        )
    )

    keys = pd.MultiIndex.from_arrays([
        eval_df["month_local"].to_numpy(),
        eval_df["hour_local"].to_numpy(),
    ])
    pred = grouped["climatology"].reindex(keys).to_numpy()
    pred = np.where(np.isfinite(pred), pred, global_mean)
    return pred.astype(np.float64), global_mean


def weighted_brier(y, p, w) -> float:
    return float(np.average((p - y) ** 2, weights=w))


def weighted_mae(y, pred, w) -> float:
    return float(np.average(np.abs(pred - y), weights=w))


def weighted_rmse(y, pred, w) -> float:
    return float(math.sqrt(np.average((pred - y) ** 2, weights=w)))


def weighted_bias(y, pred, w) -> float:
    return float(np.average(pred - y, weights=w))


def weighted_ece(y, p, w, n_bins: int = 10) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    w = np.asarray(w, dtype=float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(
        np.digitize(p, edges[1:-1], right=False),
        0,
        n_bins - 1,
    )
    total = w.sum()
    if total <= 0:
        return np.nan

    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if not np.any(m):
            continue
        wb = w[m]
        mass = wb.sum()
        conf = np.average(p[m], weights=wb)
        obs = np.average(y[m], weights=wb)
        ece += (mass / total) * abs(conf - obs)
    return float(ece)


def safe_binary_rank_metrics(y, p, w) -> tuple[float, float]:
    if np.unique(y).size < 2:
        return np.nan, np.nan
    try:
        auc = float(roc_auc_score(y, p, sample_weight=w))
    except Exception:
        auc = np.nan
    try:
        ap = float(average_precision_score(y, p, sample_weight=w))
    except Exception:
        ap = np.nan
    return auc, ap


def evaluate_binary_predictions(
    *,
    fold_id: str,
    event_id: str,
    model: str,
    representation: str,
    y: np.ndarray,
    p: np.ndarray,
    w: np.ndarray,
    climatology_p: np.ndarray,
) -> dict[str, Any]:
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    y = np.asarray(y, dtype=int)
    w = np.asarray(w, dtype=float)

    brier = weighted_brier(y, p, w)
    ref = weighted_brier(y, climatology_p, w)
    auc, ap = safe_binary_rank_metrics(y, p, w)

    return {
        "fold_id": fold_id,
        "event_id": event_id,
        "model": model,
        "representation": representation,
        "n_eval_rows": int(len(y)),
        "weighted_eval_mass": float(w.sum()),
        "weighted_event_rate": float(np.average(y, weights=w)),
        "brier": brier,
        "brier_reference_climatology": ref,
        "brier_skill_vs_climatology": (
            1.0 - brier / ref if ref > 0 else np.nan
        ),
        "roc_auc": auc,
        "average_precision": ap,
        "ece_10bins": weighted_ece(y, p, w, 10),
    }


def evaluate_continuous_predictions(
    *,
    fold_id: str,
    target: str,
    model: str,
    representation: str,
    y: np.ndarray,
    pred: np.ndarray,
    w: np.ndarray,
    climatology_pred: np.ndarray,
) -> dict[str, Any]:
    rmse = weighted_rmse(y, pred, w)
    mae = weighted_mae(y, pred, w)
    ref_rmse = weighted_rmse(y, climatology_pred, w)
    ref_mae = weighted_mae(y, climatology_pred, w)

    return {
        "fold_id": fold_id,
        "target": target,
        "model": model,
        "representation": representation,
        "n_eval_rows": int(len(y)),
        "weighted_eval_mass": float(w.sum()),
        "weighted_target_mean": float(np.average(y, weights=w)),
        "rmse": rmse,
        "mae": mae,
        "bias": weighted_bias(y, pred, w),
        "reference_climatology_rmse": ref_rmse,
        "reference_climatology_mae": ref_mae,
        "rmse_skill_vs_climatology": (
            1.0 - rmse / ref_rmse if ref_rmse > 0 else np.nan
        ),
        "mae_skill_vs_climatology": (
            1.0 - mae / ref_mae if ref_mae > 0 else np.nan
        ),
    }


def aggregate_coefficient_stability(
    coef_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    group_cols = [
        "target_type",
        "target",
        "model",
        "representation",
        "feature",
    ]

    for keys, g in coef_df.groupby(group_cols, observed=True):
        vals = g["coefficient"].to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            continue

        frac_pos = float(np.mean(finite > 0))
        frac_neg = float(np.mean(finite < 0))

        row = dict(zip(group_cols, keys))
        row.update({
            "n_folds": int(len(finite)),
            "coefficient_mean": float(np.mean(finite)),
            "coefficient_std": float(np.std(finite, ddof=0)),
            "coefficient_abs_mean": float(np.mean(np.abs(finite))),
            "fraction_positive": frac_pos,
            "fraction_negative": frac_neg,
            "sign_consistency": max(frac_pos, frac_neg),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    setup_output(args.output_dir, args.overwrite)
    logger = setup_logger(args.output_dir)

    zarr_path = args.dataset_dir / "train.zarr"
    if not zarr_path.exists():
        raise FileNotFoundError(f"train.zarr not found: {zarr_path}")

    root = open_zarr_group(zarr_path)
    input_arr = root["input"]
    target_arr = root["target"]
    timestamps_raw = np.asarray(root["timestamps"][:])

    n = int(target_arr.shape[0])
    if int(input_arr.shape[0]) != n or len(timestamps_raw) != n:
        raise RuntimeError(
            "input, target and timestamps first dimension must match."
        )
    if tuple(input_arr.shape[-2:]) != tuple(target_arr.shape[-2:]):
        raise RuntimeError(
            "Phase 12 requires aligned input/target patch shapes."
        )

    raw_names, channel_name_source = resolve_raw_channel_names(
        input_arr,
        args.channel_names,
    )
    validate_required_raw_channels(raw_names)
    predictor_names = raw_names + DERIVED_CHANNELS

    predictor_metadata(
        raw_names,
        channel_name_source,
    ).to_parquet(
        args.output_dir / "input_channel_metadata.parquet",
        index=False,
    )

    logger.info("=" * 80)
    logger.info("CORRDIFF - PHASE 12 - MULTIVARIATE ANALYSIS")
    logger.info("Input shape          : %s", tuple(input_arr.shape))
    logger.info("Target shape         : %s", tuple(target_arr.shape))
    logger.info("Sample/stratum       : %d", args.sample_per_stratum)
    logger.info("=" * 80)

    strata, strata_counts = classify_patch_strata(target_arr, logger)
    strata_counts.to_parquet(
        args.output_dir / "patch_strata_counts.parquet",
        index=False,
    )

    time_meta = build_patch_time_metadata(
        timestamps_raw,
        args.local_timezone,
    )
    manifest = build_sample_manifest(
        strata,
        time_meta,
        args.sample_per_stratum,
        args.seed,
    ).sort_values("patch_index").reset_index(drop=True)

    manifest.to_parquet(
        args.output_dir / "sampled_patch_manifest.parquet",
        index=False,
    )

    logger.info("Sampled patches       : %d", len(manifest))
    logger.info(
        "Sample ratio          : %.4f%%",
        100.0 * len(manifest) / n,
    )

    chunk0 = (
        int(input_arr.chunks[0])
        if getattr(input_arr, "chunks", None)
        else 256
    )
    patch_indices = manifest["patch_index"].to_numpy(dtype=np.int64)
    chunk_ids = patch_indices // chunk0
    unique_chunk_ids = np.unique(chunk_ids)

    means_parts: list[np.ndarray] = []
    means_std_parts: list[np.ndarray] = []
    target_parts: dict[str, list[np.ndarray]] = {
        name: [] for name in BINARY_TARGETS + CONTINUOUS_TARGETS
    }

    logger.info("=" * 80)
    logger.info("PHASE 12 - PASS B - PATCH FEATURES")
    logger.info("Chunks containing sample: %d", len(unique_chunk_ids))
    logger.info("=" * 80)

    processed = 0
    feature_names_means = None
    feature_names_means_std = None

    for ci, chunk_id in enumerate(unique_chunk_ids, start=1):
        pos = np.flatnonzero(chunk_ids == chunk_id)
        chunk_start = int(chunk_id * chunk0)
        chunk_end = min(n, chunk_start + chunk0)

        input_block = np.asarray(
            input_arr[chunk_start:chunk_end],
            dtype=np.float32,
        )
        target_block = np.asarray(
            target_arr[chunk_start:chunk_end],
            dtype=np.float32,
        )
        local_indices = (patch_indices[pos] - chunk_start).astype(np.int64)

        raw = input_block[local_indices]
        tgt = target_block[local_indices]

        predictors, names_check = derive_predictors(raw, raw_names)
        if names_check != predictor_names:
            raise RuntimeError("Predictor ordering changed unexpectedly.")
        if not np.isfinite(predictors).all():
            raise RuntimeError("Non-finite predictor in selected sample.")

        x_mean, x_mean_std, mean_names, mean_std_names = (
            extract_patch_features(predictors, predictor_names)
        )

        if feature_names_means is None:
            feature_names_means = mean_names
            feature_names_means_std = mean_std_names

        means_parts.append(x_mean)
        means_std_parts.append(x_mean_std)

        ydict = extract_targets(tgt)
        for key, value in ydict.items():
            target_parts[key].append(value)

        processed += len(pos)
        if ci == len(unique_chunk_ids) or ci % 200 == 0:
            logger.info(
                "PASS B: chunks %d/%d | patches %d/%d",
                ci,
                len(unique_chunk_ids),
                processed,
                len(manifest),
            )

    x_means = np.concatenate(means_parts, axis=0)
    x_means_std = np.concatenate(means_std_parts, axis=0)
    targets = {
        key: np.concatenate(parts, axis=0)
        for key, parts in target_parts.items()
    }

    if len(x_means) != len(manifest):
        raise RuntimeError("Feature rows do not match manifest.")
    assert feature_names_means is not None
    assert feature_names_means_std is not None

    if args.save_feature_table:
        ft = manifest[
            [
                "patch_index",
                "timestamp_utc",
                "year_utc",
                "month_local",
                "hour_local",
                "season_code",
                "sampling_weight",
                "stratum_id",
                "stratum",
            ]
        ].copy()
        for j, name in enumerate(feature_names_means_std):
            ft[name] = x_means_std[:, j]
        for key, value in targets.items():
            ft[key] = value
        ft.to_parquet(
            args.output_dir / "patch_feature_table.parquet",
            index=False,
        )

    weights_all = manifest["sampling_weight"].to_numpy(dtype=np.float64)

    # Collinearity / PCA on patch means.
    corr = weighted_corr_matrix(x_means, weights_all)
    corr_rows = []
    for i, fi in enumerate(feature_names_means):
        for j, fj in enumerate(feature_names_means):
            corr_rows.append({
                "feature_i": fi,
                "feature_j": fj,
                "weighted_correlation": float(corr[i, j]),
            })
    pd.DataFrame(corr_rows).to_parquet(
        args.output_dir / "weighted_predictor_correlation.parquet",
        index=False,
    )

    eigvals, ratios, eigvecs = weighted_pca(x_means, weights_all)
    cumulative = np.cumsum(ratios)

    pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(len(eigvals))],
        "component_index": np.arange(1, len(eigvals) + 1),
        "eigenvalue": eigvals,
        "explained_variance_ratio": ratios,
        "cumulative_explained_variance_ratio": cumulative,
    }).to_parquet(
        args.output_dir / "pca_explained_variance.parquet",
        index=False,
    )

    load_rows = []
    for pc in range(eigvecs.shape[1]):
        for j, feature in enumerate(feature_names_means):
            load_rows.append({
                "component": f"PC{pc+1}",
                "component_index": pc + 1,
                "feature": feature,
                "loading": float(eigvecs[j, pc]),
            })
    pd.DataFrame(load_rows).to_parquet(
        args.output_dir / "pca_loadings.parquet",
        index=False,
    )

    model_df = manifest[
        [
            "timestamp_utc",
            "year_utc",
            "month_local",
            "hour_local",
            "season_code",
            "sampling_weight",
        ]
    ].reset_index(drop=True)

    fold_table = build_forward_fold_table(model_df)
    fold_table.to_parquet(
        args.output_dir / "temporal_fold_definition.parquet",
        index=False,
    )
    logger.info("%s", fold_table.to_string(index=False))

    representations = {
        "means_only": (x_means, feature_names_means),
        "means_plus_std": (x_means_std, feature_names_means_std),
    }

    binary_metrics = []
    continuous_metrics = []
    logistic_coef_rows = []
    ridge_coef_rows = []
    status_rows = []

    for fold_id, train_end, eval_start, eval_end in DEFAULT_FORWARD_FOLDS:
        years = model_df["year_utc"].to_numpy()
        train_mask = years <= train_end
        eval_mask = (years >= eval_start) & (years <= eval_end)

        if not np.any(train_mask) or not np.any(eval_mask):
            logger.warning("Skipping %s due empty train/eval.", fold_id)
            continue

        train_df = model_df.loc[train_mask].reset_index(drop=True)
        eval_df = model_df.loc[eval_mask].reset_index(drop=True)
        w_train = model_df.loc[
            train_mask, "sampling_weight"
        ].to_numpy(dtype=np.float64)
        w_eval = model_df.loc[
            eval_mask, "sampling_weight"
        ].to_numpy(dtype=np.float64)

        logger.info(
            "%s | train=%d | eval=%d",
            fold_id,
            int(train_mask.sum()),
            int(eval_mask.sum()),
        )

        # Binary.
        for event_id in BINARY_TARGETS:
            y_all = np.asarray(targets[event_id], dtype=int)
            y_train = y_all[train_mask]
            y_eval = y_all[eval_mask]

            clim_eval, train_prevalence = month_hour_climatology(
                train_df, y_train, eval_df, w_train
            )
            clim_eval = np.clip(clim_eval, 1e-7, 1 - 1e-7)

            binary_metrics.append(
                evaluate_binary_predictions(
                    fold_id=fold_id,
                    event_id=event_id,
                    model="climatology_month_hour",
                    representation="calendar_reference",
                    y=y_eval,
                    p=clim_eval,
                    w=w_eval,
                    climatology_p=clim_eval,
                )
            )

            if np.unique(y_train).size < 2:
                status_rows.append({
                    "fold_id": fold_id,
                    "target_type": "binary",
                    "target": event_id,
                    "model": "all_binary_models",
                    "representation": "all",
                    "status": "skipped",
                    "detail": "training target has one class",
                })
                continue

            for representation, (x_all, feature_names) in representations.items():
                x_train = x_all[train_mask]
                x_eval = x_all[eval_mask]
                z_train, z_eval = standardize_train_eval(
                    x_train, x_eval, w_train
                )

                try:
                    model = LogisticRegression(
                        C=args.logistic_c,
                        solver="lbfgs",
                        max_iter=1000,
                        random_state=args.seed,
                    )
                    model.fit(
                        z_train,
                        y_train,
                        sample_weight=w_train,
                    )
                    p_eval = model.predict_proba(z_eval)[:, 1]

                    binary_metrics.append(
                        evaluate_binary_predictions(
                            fold_id=fold_id,
                            event_id=event_id,
                            model="logistic_l2",
                            representation=representation,
                            y=y_eval,
                            p=p_eval,
                            w=w_eval,
                            climatology_p=clim_eval,
                        )
                    )

                    for feature, coef in zip(
                        feature_names, model.coef_[0]
                    ):
                        logistic_coef_rows.append({
                            "fold_id": fold_id,
                            "target_type": "binary",
                            "target": event_id,
                            "model": "logistic_l2",
                            "representation": representation,
                            "feature": feature,
                            "coefficient": float(coef),
                        })

                    status_rows.append({
                        "fold_id": fold_id,
                        "target_type": "binary",
                        "target": event_id,
                        "model": "logistic_l2",
                        "representation": representation,
                        "status": "ok",
                        "detail": (
                            f"C={args.logistic_c}; "
                            f"train_prevalence={train_prevalence:.8f}"
                        ),
                    })
                except Exception as exc:
                    status_rows.append({
                        "fold_id": fold_id,
                        "target_type": "binary",
                        "target": event_id,
                        "model": "logistic_l2",
                        "representation": representation,
                        "status": "error",
                        "detail": repr(exc),
                    })

            # Nonlinear screen on richest representation only.
            try:
                hgb = HistGradientBoostingClassifier(
                    learning_rate=args.hist_learning_rate,
                    max_iter=args.hist_max_iter,
                    max_leaf_nodes=args.hist_max_leaf_nodes,
                    l2_regularization=1.0,
                    random_state=args.seed,
                )
                hgb.fit(
                    x_means_std[train_mask],
                    y_train,
                    sample_weight=w_train,
                )
                p_eval = hgb.predict_proba(
                    x_means_std[eval_mask]
                )[:, 1]

                binary_metrics.append(
                    evaluate_binary_predictions(
                        fold_id=fold_id,
                        event_id=event_id,
                        model="hist_gradient_boosting",
                        representation="means_plus_std",
                        y=y_eval,
                        p=p_eval,
                        w=w_eval,
                        climatology_p=clim_eval,
                    )
                )

                status_rows.append({
                    "fold_id": fold_id,
                    "target_type": "binary",
                    "target": event_id,
                    "model": "hist_gradient_boosting",
                    "representation": "means_plus_std",
                    "status": "ok",
                    "detail": (
                        f"learning_rate={args.hist_learning_rate}; "
                        f"max_iter={args.hist_max_iter}; "
                        f"max_leaf_nodes={args.hist_max_leaf_nodes}"
                    ),
                })
            except Exception as exc:
                status_rows.append({
                    "fold_id": fold_id,
                    "target_type": "binary",
                    "target": event_id,
                    "model": "hist_gradient_boosting",
                    "representation": "means_plus_std",
                    "status": "error",
                    "detail": repr(exc),
                })

        # Continuous.
        for target_name in CONTINUOUS_TARGETS:
            y_all = np.asarray(targets[target_name], dtype=np.float64)
            y_train = y_all[train_mask]
            y_eval = y_all[eval_mask]

            clim_eval, train_mean = month_hour_climatology(
                train_df, y_train, eval_df, w_train
            )

            continuous_metrics.append(
                evaluate_continuous_predictions(
                    fold_id=fold_id,
                    target=target_name,
                    model="climatology_month_hour",
                    representation="calendar_reference",
                    y=y_eval,
                    pred=clim_eval,
                    w=w_eval,
                    climatology_pred=clim_eval,
                )
            )

            for representation, (x_all, feature_names) in representations.items():
                x_train = x_all[train_mask]
                x_eval = x_all[eval_mask]
                z_train, z_eval = standardize_train_eval(
                    x_train, x_eval, w_train
                )

                try:
                    model = Ridge(
                        alpha=args.ridge_alpha,
                        fit_intercept=True,
                    )
                    model.fit(
                        z_train,
                        y_train,
                        sample_weight=w_train,
                    )
                    pred = model.predict(z_eval)

                    continuous_metrics.append(
                        evaluate_continuous_predictions(
                            fold_id=fold_id,
                            target=target_name,
                            model="ridge_l2",
                            representation=representation,
                            y=y_eval,
                            pred=pred,
                            w=w_eval,
                            climatology_pred=clim_eval,
                        )
                    )

                    for feature, coef in zip(
                        feature_names, model.coef_
                    ):
                        ridge_coef_rows.append({
                            "fold_id": fold_id,
                            "target_type": "continuous",
                            "target": target_name,
                            "model": "ridge_l2",
                            "representation": representation,
                            "feature": feature,
                            "coefficient": float(coef),
                        })

                    status_rows.append({
                        "fold_id": fold_id,
                        "target_type": "continuous",
                        "target": target_name,
                        "model": "ridge_l2",
                        "representation": representation,
                        "status": "ok",
                        "detail": (
                            f"alpha={args.ridge_alpha}; "
                            f"train_target_mean={train_mean:.8f}"
                        ),
                    })
                except Exception as exc:
                    status_rows.append({
                        "fold_id": fold_id,
                        "target_type": "continuous",
                        "target": target_name,
                        "model": "ridge_l2",
                        "representation": representation,
                        "status": "error",
                        "detail": repr(exc),
                    })

    binary_df = pd.DataFrame(binary_metrics)
    continuous_df = pd.DataFrame(continuous_metrics)
    logistic_df = pd.DataFrame(logistic_coef_rows)
    ridge_df = pd.DataFrame(ridge_coef_rows)
    status_df = pd.DataFrame(status_rows)

    binary_df.to_parquet(
        args.output_dir / "binary_model_metrics.parquet",
        index=False,
    )
    continuous_df.to_parquet(
        args.output_dir / "continuous_model_metrics.parquet",
        index=False,
    )
    logistic_df.to_parquet(
        args.output_dir / "logistic_coefficients.parquet",
        index=False,
    )
    ridge_df.to_parquet(
        args.output_dir / "ridge_coefficients.parquet",
        index=False,
    )
    status_df.to_parquet(
        args.output_dir / "model_status.parquet",
        index=False,
    )

    all_coef = pd.concat(
        [logistic_df, ridge_df],
        ignore_index=True,
    )
    stability = aggregate_coefficient_stability(all_coef)
    stability.to_parquet(
        args.output_dir / "coefficient_stability.parquet",
        index=False,
    )

    sample_ratio = len(manifest) / n
    warnings = []
    if sample_ratio < 0.05:
        warnings.append(
            "The multivariate analysis uses a stratified sample containing "
            "<5% of all patches. Inverse stratum weights are used."
        )
    if channel_name_source == "canonical_12_channel_fallback":
        warnings.append(
            "Input channel names were inferred from the canonical 12-channel "
            "builder order. Verify this order before publication."
        )

    valid_eig = eigvals[eigvals > 1e-12]
    condition_number = (
        float(valid_eig.max() / valid_eig.min())
        if valid_eig.size else np.nan
    )

    summary = {
        "phase_version": PHASE_VERSION,
        "dataset_dir": str(args.dataset_dir),
        "input_shape": [int(x) for x in input_arr.shape],
        "target_shape": [int(x) for x in target_arr.shape],
        "total_patches": n,
        "sampled_patches": int(len(manifest)),
        "sample_ratio": float(sample_ratio),
        "sample_per_stratum": int(args.sample_per_stratum),
        "seed": int(args.seed),
        "raw_channels": raw_names,
        "derived_channels": DERIVED_CHANNELS,
        "channel_name_source": channel_name_source,
        "representations": {
            "means_only": len(feature_names_means),
            "means_plus_std": len(feature_names_means_std),
        },
        "binary_targets": BINARY_TARGETS,
        "continuous_targets": CONTINUOUS_TARGETS,
        "forward_folds": [
            {
                "fold_id": f,
                "train_end_year": te,
                "eval_start_year": es,
                "eval_end_year": ee,
            }
            for f, te, es, ee in DEFAULT_FORWARD_FOLDS
        ],
        "pca_means_only": {
            "components_for_90pct": int(
                np.searchsorted(np.cumsum(ratios), 0.90) + 1
            ),
            "components_for_95pct": int(
                np.searchsorted(np.cumsum(ratios), 0.95) + 1
            ),
            "weighted_correlation_condition_number": condition_number,
        },
        "models": {
            "binary": [
                "climatology_month_hour",
                "logistic_l2",
                "hist_gradient_boosting",
            ],
            "continuous": [
                "climatology_month_hour",
                "ridge_l2",
            ],
        },
        "scope": (
            "patch-level multivariate screening on a weighted stratified "
            "sample; forward folds are for temporal stability, not final "
            "operational skill"
        ),
        "warnings": warnings,
        "methodological_notes": [
            (
                "All overlapping patches from the same timestamp fall in the "
                "same temporal period, but adjacent timestamps can belong to "
                "the same weather system."
            ),
            (
                "Forward folds are provisional. Phase 15 remains responsible "
                "for the formal train/validation/test split."
            ),
            (
                "The month x local-hour climatology is fit using training rows "
                "only in each fold."
            ),
            (
                "Logistic and ridge coefficients use weighted standardized "
                "features. Collinearity can redistribute coefficient weight "
                "among correlated predictors."
            ),
            (
                "PCA uses the weighted correlation matrix of the 20 patch-mean "
                "predictors and diagnoses redundancy, not supervised importance."
            ),
            (
                "HistGradientBoosting is only a nonlinear screening model; "
                "hyperparameters are fixed rather than tuned."
            ),
            (
                "means_plus_std adds within-patch variability but is not the "
                "formal ablation study planned for Phase 17."
            ),
            (
                "Metrics are weighted to approximate the original patch "
                "distribution. Weighted patch mass is not an independent "
                "sample size."
            ),
            (
                "Pixel fractions remain properties of overlapping patches, "
                "not de-duplicated full radar fields."
            ),
            (
                "All multivariate results are descriptive/predictive screening "
                "and do not establish causality."
            ),
        ],
    }

    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("=" * 80)
    logger.info("PHASE 12 COMPLETE")
    logger.info("Output: %s", args.output_dir)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
