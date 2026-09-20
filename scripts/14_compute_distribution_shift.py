#!/usr/bin/env python3
"""
CorrDiff - Fase 14
Distribution Shift e Estabilidade Temporal
===========================================

Objetivo
--------
Separar formalmente quatro fontes de mudança temporal:

1. coverage / disponibilidade do radar;
2. shift marginal dos predictors ERA5;
3. shift de composição dos regimes meteorológicos;
4. shift condicional do radar dentro dos regimes.

A fase também decompõe a mudança da taxa de eventos radar em:
- mudança de composição P(regime);
- mudança within-regime P(event | regime).

Inputs esperados
----------------
analysis_outputs/06_diurnal/
├── predictor_timestamp_means.parquet
└── timestamp_event_metrics.parquet

analysis_outputs/13_regimes/
└── regime_assignments.parquet

Outputs
-------
analysis_outputs/14_distribution_shift/
├── analysis_summary.json
├── comparison_windows.parquet
├── coverage_by_year.parquet
├── coverage_by_month.parquet
├── missing_gap_runs.parquet
├── predictor_shift_metrics.parquet
├── predictor_shift_anomaly_metrics.parquet
├── regime_composition_shift.parquet
├── regime_prevalence_delta.parquet
├── radar_event_shift.parquet
├── radar_continuous_shift.parquet
├── conditional_event_shift_by_regime.parquet
├── event_rate_decomposition.parquet
└── phase14.log
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


PHASE_VERSION = "phase14-distribution-shift-v1-coverage-regime-conditional"

RAW_CANONICAL = [
    "tcwv",
    "t2m",
    "u10",
    "v10",
    "t_850",
    "r_850",
    "u_850",
    "v_850",
    "t_500",
    "r_500",
    "u_500",
    "v_500",
]

DERIVED_CANONICAL = [
    "wind_speed_10",
    "wind_speed_850",
    "wind_speed_500",
    "delta_t_500_850",
    "delta_t_850_surface",
    "delta_r_500_850",
    "bulk_wind_diff_10_850",
    "bulk_wind_diff_850_500",
]

ALL_PREDICTORS = RAW_CANONICAL + DERIVED_CANONICAL

ALIASES = {
    "tcwv": ["tcwv"],
    "t2m": ["t2m"],
    "u10": ["u10"],
    "v10": ["v10"],
    "t_850": ["t_850", "t850"],
    "r_850": ["r_850", "r850"],
    "u_850": ["u_850", "u850"],
    "v_850": ["v_850", "v850"],
    "t_500": ["t_500", "t500"],
    "r_500": ["r_500", "r500"],
    "u_500": ["u_500", "u500"],
    "v_500": ["v_500", "v500"],
    "wind_speed_10": ["wind_speed_10"],
    "wind_speed_850": ["wind_speed_850"],
    "wind_speed_500": ["wind_speed_500"],
    "delta_t_500_850": ["delta_t_500_850"],
    "delta_t_850_surface": ["delta_t_850_surface"],
    "delta_r_500_850": ["delta_r_500_850"],
    "bulk_wind_diff_10_850": ["bulk_wind_diff_10_850"],
    "bulk_wind_diff_850_500": ["bulk_wind_diff_850_500"],
}

EVENT_THRESHOLDS = {
    "gt_0": 0.0,
    "ge_20": 20.0,
    "ge_30": 30.0,
    "ge_40": 40.0,
    "ge_45": 45.0,
}

REGIME_SPACES = ["absolute", "anomaly_month_hour"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CorrDiff Fase 14: distribution shift."
    )
    p.add_argument(
        "--phase6-dir",
        type=Path,
        default=Path("analysis_outputs/06_diurnal"),
    )
    p.add_argument(
        "--phase13-dir",
        type=Path,
        default=Path("analysis_outputs/13_regimes"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs/14_distribution_shift"),
    )
    p.add_argument(
        "--dataset-start-utc",
        default="2011-01-01 00:00:00+00:00",
    )
    p.add_argument(
        "--dataset-end-utc",
        default="2024-12-31 23:00:00+00:00",
    )
    p.add_argument(
        "--local-timezone",
        default="America/Sao_Paulo",
    )
    p.add_argument(
        "--bootstrap-reps",
        type=int,
        default=500,
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--psi-bins",
        type=int,
        default=10,
    )
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
    logger = logging.getLogger("corrdiff.phase14")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(path / "phase14.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def resolve_timestamp_column(df: pd.DataFrame) -> str:
    for c in ["timestamp_utc", "timestamp", "time"]:
        if c in df.columns:
            return c
    raise RuntimeError(
        "Could not find timestamp column. Expected timestamp_utc/timestamp/time."
    )


def ensure_time_columns(
    df: pd.DataFrame,
    local_timezone: str,
) -> pd.DataFrame:
    out = df.copy()
    ts_col = resolve_timestamp_column(out)
    ts = pd.to_datetime(out[ts_col], utc=True)
    out["timestamp_utc"] = ts
    local = ts.dt.tz_convert(local_timezone)

    out["year_utc"] = ts.dt.year.astype(np.int16)
    out["month_utc"] = ts.dt.month.astype(np.int8)
    out["date_utc"] = ts.dt.floor("D")
    out["month_local"] = local.dt.month.astype(np.int8)
    out["hour_local"] = local.dt.hour.astype(np.int8)
    return out


def canonicalize_predictors(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    available = []

    for canonical in ALL_PREDICTORS:
        found = None
        for alias in ALIASES[canonical]:
            if alias in out.columns:
                found = alias
                break
        if found is not None:
            if found != canonical:
                out[canonical] = out[found]
            available.append(canonical)

    missing_raw = sorted(set(RAW_CANONICAL) - set(available))
    if missing_raw:
        raise RuntimeError(
            f"Missing required raw predictors: {missing_raw}"
        )
    return out, available


def prepare_radar(
    radar: pd.DataFrame,
    local_timezone: str,
) -> pd.DataFrame:
    out = ensure_time_columns(radar, local_timezone)

    if "max_dbz" not in out.columns:
        raise RuntimeError(
            "Phase 14 requires max_dbz in Phase 6 timestamp_event_metrics."
        )

    max_dbz = out["max_dbz"].to_numpy(dtype=float)
    for event_id, threshold in EVENT_THRESHOLDS.items():
        if event_id == "gt_0":
            out[f"event_{event_id}"] = (max_dbz > 0).astype(np.int8)
        else:
            out[f"event_{event_id}"] = (
                max_dbz >= threshold
            ).astype(np.int8)

    if (
        "positive_pixel_fraction" not in out.columns
        and "event_pixel_fraction_gt_0" in out.columns
    ):
        out["positive_pixel_fraction"] = out[
            "event_pixel_fraction_gt_0"
        ]

    return out


def comparison_windows() -> pd.DataFrame:
    rows = [
        {
            "comparison_id": "Y2021_vs_pre2021",
            "reference_start_year": 2011,
            "reference_end_year": 2020,
            "evaluation_start_year": 2021,
            "evaluation_end_year": 2021,
            "purpose": "year-specific shift",
        },
        {
            "comparison_id": "Y2022_vs_pre2021",
            "reference_start_year": 2011,
            "reference_end_year": 2020,
            "evaluation_start_year": 2022,
            "evaluation_end_year": 2022,
            "purpose": "year-specific shift",
        },
        {
            "comparison_id": "Y2023_vs_pre2021",
            "reference_start_year": 2011,
            "reference_end_year": 2020,
            "evaluation_start_year": 2023,
            "evaluation_end_year": 2023,
            "purpose": "year-specific shift",
        },
        {
            "comparison_id": "Y2024_vs_pre2021",
            "reference_start_year": 2011,
            "reference_end_year": 2020,
            "evaluation_start_year": 2024,
            "evaluation_end_year": 2024,
            "purpose": "year-specific shift",
        },
        {
            "comparison_id": "RECENT_2023_2024_vs_pre2023",
            "reference_start_year": 2011,
            "reference_end_year": 2022,
            "evaluation_start_year": 2023,
            "evaluation_end_year": 2024,
            "purpose": "F5-oriented shift",
        },
    ]
    return pd.DataFrame(rows)


def masks_for_comparison(
    years: np.ndarray,
    row: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    ref = (
        (years >= int(row["reference_start_year"]))
        & (years <= int(row["reference_end_year"]))
    )
    ev = (
        (years >= int(row["evaluation_start_year"]))
        & (years <= int(row["evaluation_end_year"]))
    )
    return ref, ev


def empirical_ks(x: np.ndarray, y: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    y = np.sort(np.asarray(y, dtype=float))
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan

    values = np.sort(np.unique(np.concatenate([x, y])))
    cdf_x = np.searchsorted(x, values, side="right") / len(x)
    cdf_y = np.searchsorted(y, values, side="right") / len(y)
    return float(np.max(np.abs(cdf_x - cdf_y)))


def psi_from_reference_bins(
    reference: np.ndarray,
    evaluation: np.ndarray,
    bins: int,
) -> float:
    ref = np.asarray(reference, dtype=float)
    ev = np.asarray(evaluation, dtype=float)
    ref = ref[np.isfinite(ref)]
    ev = ev[np.isfinite(ev)]
    if len(ref) == 0 or len(ev) == 0:
        return np.nan

    qs = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(ref, qs))

    if len(edges) < 3:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    ev_counts, _ = np.histogram(ev, bins=edges)

    eps = 1e-6
    p = ref_counts / max(ref_counts.sum(), 1)
    q = ev_counts / max(ev_counts.sum(), 1)
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    p = p / p.sum()
    q = q / q.sum()

    return float(np.sum((q - p) * np.log(q / p)))


def standardized_quantile_shift(
    ref: np.ndarray,
    ev: np.ndarray,
    q: float,
) -> float:
    ref = np.asarray(ref, dtype=float)
    ev = np.asarray(ev, dtype=float)
    ref = ref[np.isfinite(ref)]
    ev = ev[np.isfinite(ev)]
    if len(ref) == 0 or len(ev) == 0:
        return np.nan
    std = float(np.std(ref, ddof=0))
    if std < 1e-12:
        return np.nan
    return float(
        (np.quantile(ev, q) - np.quantile(ref, q)) / std
    )


def predictor_shift_rows(
    frame: pd.DataFrame,
    predictors: list[str],
    comparisons: pd.DataFrame,
    representation: str,
    psi_bins: int,
) -> pd.DataFrame:
    rows = []
    years = frame["year_utc"].to_numpy()

    for _, comp in comparisons.iterrows():
        ref_mask, ev_mask = masks_for_comparison(years, comp)

        for predictor in predictors:
            ref = frame.loc[ref_mask, predictor].to_numpy(dtype=float)
            ev = frame.loc[ev_mask, predictor].to_numpy(dtype=float)
            ref = ref[np.isfinite(ref)]
            ev = ev[np.isfinite(ev)]

            if len(ref) == 0 or len(ev) == 0:
                continue

            ref_mean = float(np.mean(ref))
            ev_mean = float(np.mean(ev))
            ref_std = float(np.std(ref, ddof=0))
            ev_std = float(np.std(ev, ddof=0))

            smd = (
                (ev_mean - ref_mean) / ref_std
                if ref_std > 1e-12
                else np.nan
            )
            std_ratio = (
                ev_std / ref_std
                if ref_std > 1e-12
                else np.nan
            )

            rows.append({
                "comparison_id": comp["comparison_id"],
                "representation": representation,
                "predictor": predictor,
                "n_reference": int(len(ref)),
                "n_evaluation": int(len(ev)),
                "reference_mean": ref_mean,
                "evaluation_mean": ev_mean,
                "reference_std": ref_std,
                "evaluation_std": ev_std,
                "standardized_mean_difference": float(smd),
                "std_ratio_eval_over_reference": float(std_ratio),
                "ks_statistic": empirical_ks(ref, ev),
                "psi": psi_from_reference_bins(
                    ref,
                    ev,
                    psi_bins,
                ),
                "p10_shift_in_reference_sigma": standardized_quantile_shift(
                    ref, ev, 0.10
                ),
                "p50_shift_in_reference_sigma": standardized_quantile_shift(
                    ref, ev, 0.50
                ),
                "p90_shift_in_reference_sigma": standardized_quantile_shift(
                    ref, ev, 0.90
                ),
            })

    return pd.DataFrame(rows)


def build_reference_climatology_anomalies(
    frame: pd.DataFrame,
    predictors: list[str],
    comparison: pd.Series,
) -> pd.DataFrame:
    years = frame["year_utc"].to_numpy()
    ref_mask, _ = masks_for_comparison(years, comparison)

    ref = frame.loc[
        ref_mask,
        ["month_local", "hour_local"] + predictors,
    ].copy()

    clim = (
        ref.groupby(
            ["month_local", "hour_local"],
            observed=True,
        )[predictors]
        .mean()
    )

    keys = pd.MultiIndex.from_arrays([
        frame["month_local"].to_numpy(),
        frame["hour_local"].to_numpy(),
    ])
    mapped = clim.reindex(keys)

    out = frame[
        [
            "timestamp_utc",
            "year_utc",
            "month_local",
            "hour_local",
        ]
    ].copy()

    global_ref_means = ref[predictors].mean()

    for predictor in predictors:
        baseline = mapped[predictor].to_numpy(dtype=float)
        baseline = np.where(
            np.isfinite(baseline),
            baseline,
            float(global_ref_means[predictor]),
        )
        out[predictor] = (
            frame[predictor].to_numpy(dtype=float) - baseline
        )

    return out


def coverage_tables(
    timestamps: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    expected = pd.date_range(start, end, freq="h", tz="UTC")
    available = pd.DatetimeIndex(
        pd.to_datetime(timestamps, utc=True).drop_duplicates()
    )
    available_set = pd.Series(
        1,
        index=available,
        dtype=np.int8,
    )
    frame = pd.DataFrame(index=expected)
    frame["available"] = (
        available_set.reindex(expected).fillna(0).to_numpy(dtype=np.int8)
    )
    frame["year_utc"] = frame.index.year
    frame["month_utc"] = frame.index.month

    by_year = (
        frame.groupby("year_utc", observed=True)["available"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(
            columns={
                "sum": "available_hours",
                "count": "expected_hours",
            }
        )
    )
    by_year["missing_hours"] = (
        by_year["expected_hours"] - by_year["available_hours"]
    )
    by_year["availability_ratio"] = (
        by_year["available_hours"] / by_year["expected_hours"]
    )

    by_month = (
        frame.groupby(
            ["year_utc", "month_utc"],
            observed=True,
        )["available"]
        .agg(["sum", "count"])
        .reset_index()
        .rename(
            columns={
                "sum": "available_hours",
                "count": "expected_hours",
            }
        )
    )
    by_month["missing_hours"] = (
        by_month["expected_hours"] - by_month["available_hours"]
    )
    by_month["availability_ratio"] = (
        by_month["available_hours"] / by_month["expected_hours"]
    )

    # Exact runs of missing hourly timestamps.
    missing = frame["available"].to_numpy() == 0
    run_rows = []
    i = 0
    idx = frame.index
    while i < len(missing):
        if not missing[i]:
            i += 1
            continue
        j = i + 1
        while j < len(missing) and missing[j]:
            j += 1
        run_rows.append({
            "start_timestamp_utc": idx[i],
            "end_timestamp_utc": idx[j - 1],
            "length_hours": int(j - i),
            "start_year_utc": int(idx[i].year),
            "end_year_utc": int(idx[j - 1].year),
        })
        i = j

    gaps = pd.DataFrame(run_rows)
    if not gaps.empty:
        gaps = gaps.sort_values(
            ["length_hours", "start_timestamp_utc"],
            ascending=[False, True],
        ).reset_index(drop=True)

    return by_year, by_month, gaps


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    eps = 1e-12
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def regime_composition_tables(
    assignments: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    detail_rows = []

    for space in REGIME_SPACES:
        sub = assignments[
            assignments["regime_space"] == space
        ].copy()
        years = sub["year_utc"].to_numpy()
        regime_ids = sorted(sub["regime_id"].unique())

        for _, comp in comparisons.iterrows():
            ref_mask, ev_mask = masks_for_comparison(years, comp)
            ref = sub.loc[ref_mask, "regime_id"]
            ev = sub.loc[ev_mask, "regime_id"]

            if len(ref) == 0 or len(ev) == 0:
                continue

            p = np.array(
                [(ref == rid).mean() for rid in regime_ids],
                dtype=float,
            )
            q = np.array(
                [(ev == rid).mean() for rid in regime_ids],
                dtype=float,
            )

            deltas = q - p

            summary_rows.append({
                "comparison_id": comp["comparison_id"],
                "regime_space": space,
                "n_reference": int(len(ref)),
                "n_evaluation": int(len(ev)),
                "jensen_shannon_divergence": js_divergence(p, q),
                "total_variation_distance": float(
                    0.5 * np.abs(deltas).sum()
                ),
                "max_absolute_prevalence_delta": float(
                    np.abs(deltas).max()
                ),
            })

            for rid, pr, pe, delta in zip(
                regime_ids,
                p,
                q,
                deltas,
            ):
                detail_rows.append({
                    "comparison_id": comp["comparison_id"],
                    "regime_space": space,
                    "regime_id": int(rid),
                    "regime": f"R{int(rid)}",
                    "reference_prevalence": float(pr),
                    "evaluation_prevalence": float(pe),
                    "prevalence_delta": float(delta),
                    "prevalence_ratio_eval_over_ref": (
                        float(pe / pr)
                        if pr > 0
                        else np.nan
                    ),
                })

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def daily_count_table(
    frame: pd.DataFrame,
    event_col: str,
    mask: np.ndarray,
    regime_col: str | None = None,
    regime_id: int | None = None,
) -> pd.DataFrame:
    sub = frame.loc[mask, ["date_utc", event_col] + (
        [regime_col] if regime_col is not None else []
    )].copy()

    if regime_col is not None:
        sub = sub[sub[regime_col] == regime_id]

    if sub.empty:
        return pd.DataFrame(columns=["date_utc", "n", "events"])

    return (
        sub.groupby("date_utc", observed=True)[event_col]
        .agg(["size", "sum"])
        .reset_index()
        .rename(columns={"size": "n", "sum": "events"})
    )


def bootstrap_rate_difference(
    ref_daily: pd.DataFrame,
    ev_daily: pd.DataFrame,
    reps: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, float, float, float]:
    if ref_daily.empty or ev_daily.empty:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    ref_rate = float(
        ref_daily["events"].sum() / ref_daily["n"].sum()
    )
    ev_rate = float(
        ev_daily["events"].sum() / ev_daily["n"].sum()
    )
    observed = ev_rate - ref_rate

    if reps <= 0:
        rr = ev_rate / ref_rate if ref_rate > 0 else np.nan
        return observed, np.nan, np.nan, rr, np.nan, np.nan

    ref_n = ref_daily["n"].to_numpy(dtype=float)
    ref_e = ref_daily["events"].to_numpy(dtype=float)
    ev_n = ev_daily["n"].to_numpy(dtype=float)
    ev_e = ev_daily["events"].to_numpy(dtype=float)

    diffs = np.empty(reps, dtype=float)
    rrs = np.empty(reps, dtype=float)

    for b in range(reps):
        ir = rng.integers(0, len(ref_daily), size=len(ref_daily))
        ie = rng.integers(0, len(ev_daily), size=len(ev_daily))

        rn = ref_n[ir].sum()
        re = ref_e[ir].sum()
        en = ev_n[ie].sum()
        ee = ev_e[ie].sum()

        rr0 = re / rn if rn > 0 else np.nan
        rr1 = ee / en if en > 0 else np.nan

        diffs[b] = rr1 - rr0
        rrs[b] = (
            rr1 / rr0
            if np.isfinite(rr0) and rr0 > 0
            else np.nan
        )

    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    rr_obs = ev_rate / ref_rate if ref_rate > 0 else np.nan
    rr_lo, rr_hi = np.nanpercentile(rrs, [2.5, 97.5])

    return (
        observed,
        float(lo),
        float(hi),
        float(rr_obs),
        float(rr_lo),
        float(rr_hi),
    )


def radar_event_shift(
    radar: pd.DataFrame,
    comparisons: pd.DataFrame,
    reps: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    years = radar["year_utc"].to_numpy()
    rng = np.random.default_rng(seed)

    for _, comp in comparisons.iterrows():
        ref_mask, ev_mask = masks_for_comparison(years, comp)

        for event_id in EVENT_THRESHOLDS:
            col = f"event_{event_id}"

            ref_daily = daily_count_table(
                radar,
                col,
                ref_mask,
            )
            ev_daily = daily_count_table(
                radar,
                col,
                ev_mask,
            )

            result = bootstrap_rate_difference(
                ref_daily,
                ev_daily,
                reps,
                rng,
            )
            (
                diff,
                diff_lo,
                diff_hi,
                rr,
                rr_lo,
                rr_hi,
            ) = result

            ref_rate = float(
                radar.loc[ref_mask, col].mean()
            )
            ev_rate = float(
                radar.loc[ev_mask, col].mean()
            )

            rows.append({
                "comparison_id": comp["comparison_id"],
                "event_id": event_id,
                "n_reference": int(ref_mask.sum()),
                "n_evaluation": int(ev_mask.sum()),
                "reference_event_rate": ref_rate,
                "evaluation_event_rate": ev_rate,
                "event_rate_difference": diff,
                "event_rate_difference_ci95_low": diff_lo,
                "event_rate_difference_ci95_high": diff_hi,
                "risk_ratio_eval_over_reference": rr,
                "risk_ratio_ci95_low": rr_lo,
                "risk_ratio_ci95_high": rr_hi,
                "bootstrap_block": "UTC_day",
                "bootstrap_reps": int(reps),
            })

    return pd.DataFrame(rows)


def radar_continuous_shift(
    radar: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> pd.DataFrame:
    metrics = [
        c for c in [
            "max_dbz",
            "positive_pixel_fraction",
            "event_pixel_fraction_ge_30",
            "event_pixel_fraction_ge_40",
            "event_pixel_fraction_ge_45",
        ]
        if c in radar.columns
    ]

    rows = []
    years = radar["year_utc"].to_numpy()

    for _, comp in comparisons.iterrows():
        ref_mask, ev_mask = masks_for_comparison(years, comp)

        for metric in metrics:
            ref = radar.loc[ref_mask, metric].to_numpy(dtype=float)
            ev = radar.loc[ev_mask, metric].to_numpy(dtype=float)
            ref = ref[np.isfinite(ref)]
            ev = ev[np.isfinite(ev)]

            if len(ref) == 0 or len(ev) == 0:
                continue

            ref_mean = float(np.mean(ref))
            ev_mean = float(np.mean(ev))
            ref_std = float(np.std(ref, ddof=0))

            rows.append({
                "comparison_id": comp["comparison_id"],
                "metric": metric,
                "n_reference": int(len(ref)),
                "n_evaluation": int(len(ev)),
                "reference_mean": ref_mean,
                "evaluation_mean": ev_mean,
                "mean_difference": ev_mean - ref_mean,
                "standardized_mean_difference": (
                    (ev_mean - ref_mean) / ref_std
                    if ref_std > 1e-12
                    else np.nan
                ),
                "ks_statistic": empirical_ks(ref, ev),
                "p50_shift_in_reference_sigma": standardized_quantile_shift(
                    ref, ev, 0.50
                ),
                "p90_shift_in_reference_sigma": standardized_quantile_shift(
                    ref, ev, 0.90
                ),
                "p99_shift_in_reference_sigma": standardized_quantile_shift(
                    ref, ev, 0.99
                ),
            })

    return pd.DataFrame(rows)


def conditional_event_shift_by_regime(
    merged: pd.DataFrame,
    comparisons: pd.DataFrame,
    reps: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed + 101)

    for space in REGIME_SPACES:
        sub = merged[
            merged["regime_space"] == space
        ].copy()

        years = sub["year_utc"].to_numpy()
        regimes = sorted(sub["regime_id"].unique())

        for _, comp in comparisons.iterrows():
            ref_mask, ev_mask = masks_for_comparison(years, comp)

            for regime_id in regimes:
                for event_id in EVENT_THRESHOLDS:
                    col = f"event_{event_id}"

                    ref_daily = daily_count_table(
                        sub,
                        col,
                        ref_mask,
                        regime_col="regime_id",
                        regime_id=int(regime_id),
                    )
                    ev_daily = daily_count_table(
                        sub,
                        col,
                        ev_mask,
                        regime_col="regime_id",
                        regime_id=int(regime_id),
                    )

                    result = bootstrap_rate_difference(
                        ref_daily,
                        ev_daily,
                        reps,
                        rng,
                    )
                    (
                        diff,
                        diff_lo,
                        diff_hi,
                        rr,
                        rr_lo,
                        rr_hi,
                    ) = result

                    ref_values = sub.loc[
                        ref_mask & (
                            sub["regime_id"].to_numpy() == regime_id
                        ),
                        col,
                    ]
                    ev_values = sub.loc[
                        ev_mask & (
                            sub["regime_id"].to_numpy() == regime_id
                        ),
                        col,
                    ]

                    rows.append({
                        "comparison_id": comp["comparison_id"],
                        "regime_space": space,
                        "regime_id": int(regime_id),
                        "regime": f"R{int(regime_id)}",
                        "event_id": event_id,
                        "n_reference": int(len(ref_values)),
                        "n_evaluation": int(len(ev_values)),
                        "reference_event_rate": (
                            float(ref_values.mean())
                            if len(ref_values)
                            else np.nan
                        ),
                        "evaluation_event_rate": (
                            float(ev_values.mean())
                            if len(ev_values)
                            else np.nan
                        ),
                        "event_rate_difference": diff,
                        "event_rate_difference_ci95_low": diff_lo,
                        "event_rate_difference_ci95_high": diff_hi,
                        "risk_ratio_eval_over_reference": rr,
                        "risk_ratio_ci95_low": rr_lo,
                        "risk_ratio_ci95_high": rr_hi,
                        "bootstrap_block": "UTC_day",
                        "bootstrap_reps": int(reps),
                    })

    return pd.DataFrame(rows)


def event_rate_decomposition(
    merged: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> pd.DataFrame:
    """
    Exact Shapley-style two-component decomposition:

      total delta =
          sum_r (p1-p0) * (q0+q1)/2
        + sum_r ((p0+p1)/2) * (q1-q0)

    where:
      p = regime prevalence
      q = event rate within regime

    The two components sum exactly to the modeled total change (up to rounding).
    """
    rows = []

    for space in REGIME_SPACES:
        sub = merged[
            merged["regime_space"] == space
        ].copy()

        years = sub["year_utc"].to_numpy()
        regimes = sorted(sub["regime_id"].unique())

        for _, comp in comparisons.iterrows():
            ref_mask, ev_mask = masks_for_comparison(years, comp)

            for event_id in ["ge_30", "ge_40", "ge_45"]:
                col = f"event_{event_id}"

                p0 = []
                p1 = []
                q0 = []
                q1 = []

                for rid in regimes:
                    r0 = ref_mask & (
                        sub["regime_id"].to_numpy() == rid
                    )
                    r1 = ev_mask & (
                        sub["regime_id"].to_numpy() == rid
                    )

                    p0.append(float(r0.sum() / max(ref_mask.sum(), 1)))
                    p1.append(float(r1.sum() / max(ev_mask.sum(), 1)))
                    q0.append(
                        float(sub.loc[r0, col].mean())
                        if r0.sum() > 0
                        else np.nan
                    )
                    q1.append(
                        float(sub.loc[r1, col].mean())
                        if r1.sum() > 0
                        else np.nan
                    )

                p0 = np.asarray(p0, dtype=float)
                p1 = np.asarray(p1, dtype=float)
                q0 = np.asarray(q0, dtype=float)
                q1 = np.asarray(q1, dtype=float)

                # If a regime has zero prevalence in one period, its
                # conditional event rate is not empirically identifiable there.
                # For decomposition only, carry the observed conditional rate
                # from the other period into the zero-prevalence side. This
                # yields zero within-regime contribution for an absent regime
                # and assigns its appearance/disappearance entirely to
                # composition, preserving the exact total-rate identity.
                q0_filled = q0.copy()
                q1_filled = q1.copy()

                m = (~np.isfinite(q0_filled)) & np.isfinite(q1_filled)
                q0_filled[m] = q1_filled[m]

                m = (~np.isfinite(q1_filled)) & np.isfinite(q0_filled)
                q1_filled[m] = q0_filled[m]

                valid = (
                    np.isfinite(p0)
                    & np.isfinite(p1)
                    & np.isfinite(q0_filled)
                    & np.isfinite(q1_filled)
                )

                p0v = p0[valid]
                p1v = p1[valid]
                q0v = q0_filled[valid]
                q1v = q1_filled[valid]

                composition = float(
                    np.sum(
                        (p1v - p0v)
                        * 0.5
                        * (q0v + q1v)
                    )
                )
                within = float(
                    np.sum(
                        0.5
                        * (p0v + p1v)
                        * (q1v - q0v)
                    )
                )

                ref_rate = float(sub.loc[ref_mask, col].mean())
                ev_rate = float(sub.loc[ev_mask, col].mean())
                total = ev_rate - ref_rate
                modeled_total = composition + within

                rows.append({
                    "comparison_id": comp["comparison_id"],
                    "regime_space": space,
                    "event_id": event_id,
                    "reference_event_rate": ref_rate,
                    "evaluation_event_rate": ev_rate,
                    "total_event_rate_change": total,
                    "composition_component": composition,
                    "within_regime_component": within,
                    "decomposition_sum": modeled_total,
                    "decomposition_residual": total - modeled_total,
                    "composition_fraction_of_abs_components": (
                        abs(composition)
                        / (abs(composition) + abs(within))
                        if (abs(composition) + abs(within)) > 0
                        else np.nan
                    ),
                    "within_regime_fraction_of_abs_components": (
                        abs(within)
                        / (abs(composition) + abs(within))
                        if (abs(composition) + abs(within)) > 0
                        else np.nan
                    ),
                    "method": "symmetric_shapley_two_component",
                })

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    setup_output(args.output_dir, args.overwrite)
    logger = setup_logger(args.output_dir)

    pred_path = args.phase6_dir / "predictor_timestamp_means.parquet"
    radar_path = args.phase6_dir / "timestamp_event_metrics.parquet"
    regime_path = args.phase13_dir / "regime_assignments.parquet"

    for p in [pred_path, radar_path, regime_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    predictors = pd.read_parquet(pred_path)
    radar = pd.read_parquet(radar_path)
    assignments = pd.read_parquet(regime_path)

    predictors = ensure_time_columns(
        predictors,
        args.local_timezone,
    )
    predictors, available_predictors = canonicalize_predictors(
        predictors
    )
    radar = prepare_radar(
        radar,
        args.local_timezone,
    )
    assignments = ensure_time_columns(
        assignments,
        args.local_timezone,
    )

    if "regime_space" not in assignments.columns:
        raise RuntimeError(
            "regime_assignments.parquet missing regime_space."
        )
    if "regime_id" not in assignments.columns:
        raise RuntimeError(
            "regime_assignments.parquet missing regime_id."
        )

    comparisons = comparison_windows()
    comparisons.to_parquet(
        args.output_dir / "comparison_windows.parquet",
        index=False,
    )

    logger.info("=" * 92)
    logger.info("CORRDIFF - PHASE 14 - DISTRIBUTION SHIFT")
    logger.info("Predictor timestamps : %d", len(predictors))
    logger.info("Radar timestamps     : %d", len(radar))
    logger.info("Regime assignments   : %d", len(assignments))
    logger.info("Bootstrap reps       : %d", args.bootstrap_reps)
    logger.info("=" * 92)

    # ------------------------------------------------------------------
    # Coverage
    # ------------------------------------------------------------------
    start = pd.Timestamp(args.dataset_start_utc)
    end = pd.Timestamp(args.dataset_end_utc)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    else:
        end = end.tz_convert("UTC")

    coverage_year, coverage_month, gaps = coverage_tables(
        radar["timestamp_utc"],
        start,
        end,
    )
    coverage_year.to_parquet(
        args.output_dir / "coverage_by_year.parquet",
        index=False,
    )
    coverage_month.to_parquet(
        args.output_dir / "coverage_by_month.parquet",
        index=False,
    )
    gaps.to_parquet(
        args.output_dir / "missing_gap_runs.parquet",
        index=False,
    )

    # ------------------------------------------------------------------
    # Predictor shifts - absolute
    # ------------------------------------------------------------------
    pred_cols = [
        p for p in ALL_PREDICTORS
        if p in available_predictors
    ]

    absolute_shift = predictor_shift_rows(
        predictors,
        pred_cols,
        comparisons,
        "absolute",
        args.psi_bins,
    )
    absolute_shift.to_parquet(
        args.output_dir / "predictor_shift_metrics.parquet",
        index=False,
    )

    # Predictor shifts - anomaly against comparison-specific reference climatology.
    anomaly_parts = []
    for _, comp in comparisons.iterrows():
        anomaly_frame = build_reference_climatology_anomalies(
            predictors,
            pred_cols,
            comp,
        )
        one = pd.DataFrame([comp])
        part = predictor_shift_rows(
            anomaly_frame,
            pred_cols,
            one,
            "reference_month_hour_anomaly",
            args.psi_bins,
        )
        anomaly_parts.append(part)

    anomaly_shift = pd.concat(
        anomaly_parts,
        ignore_index=True,
    )
    anomaly_shift.to_parquet(
        args.output_dir / "predictor_shift_anomaly_metrics.parquet",
        index=False,
    )

    # ------------------------------------------------------------------
    # Regime composition shift
    # ------------------------------------------------------------------
    comp_summary, comp_detail = regime_composition_tables(
        assignments,
        comparisons,
    )
    comp_summary.to_parquet(
        args.output_dir / "regime_composition_shift.parquet",
        index=False,
    )
    comp_detail.to_parquet(
        args.output_dir / "regime_prevalence_delta.parquet",
        index=False,
    )

    # ------------------------------------------------------------------
    # Radar marginal shift
    # ------------------------------------------------------------------
    event_shift = radar_event_shift(
        radar,
        comparisons,
        args.bootstrap_reps,
        args.seed,
    )
    event_shift.to_parquet(
        args.output_dir / "radar_event_shift.parquet",
        index=False,
    )

    continuous_shift = radar_continuous_shift(
        radar,
        comparisons,
    )
    continuous_shift.to_parquet(
        args.output_dir / "radar_continuous_shift.parquet",
        index=False,
    )

    # ------------------------------------------------------------------
    # Conditional radar shift within regimes
    # ------------------------------------------------------------------
    radar_cols = [
        "timestamp_utc",
        "date_utc",
        "year_utc",
    ] + [
        f"event_{e}" for e in EVENT_THRESHOLDS
    ]

    merged = assignments[
        [
            "timestamp_utc",
            "regime_space",
            "regime_id",
        ]
    ].merge(
        radar[radar_cols],
        on="timestamp_utc",
        how="inner",
        validate="many_to_one",
    )

    conditional_shift = conditional_event_shift_by_regime(
        merged,
        comparisons,
        args.bootstrap_reps,
        args.seed,
    )
    conditional_shift.to_parquet(
        args.output_dir / "conditional_event_shift_by_regime.parquet",
        index=False,
    )

    decomposition = event_rate_decomposition(
        merged,
        comparisons,
    )
    decomposition.to_parquet(
        args.output_dir / "event_rate_decomposition.parquet",
        index=False,
    )

    warnings = [
        (
            "All target/radar shift results are conditioned on timestamps with "
            "radar availability. Coverage shift is therefore analyzed explicitly "
            "and must be considered before interpreting physical change."
        ),
        (
            "Predictor shift uses timestamp-level ERA5 summaries from Phase 6, "
            "not the full native ERA5 fields."
        ),
        (
            "Absolute predictor shift can include seasonal-composition effects. "
            "A reference month x local-hour anomaly representation is reported "
            "separately."
        ),
        (
            "Reference climatologies for anomaly-space comparisons are fit only "
            "on the reference period of each comparison."
        ),
        (
            "Regime shift uses Phase 13 KMeans assignments. Regimes are statistical "
            "states, not unique physical weather classes."
        ),
        (
            "Daily block bootstrap preserves within-day temporal dependence but "
            "does not eliminate dependence across adjacent multi-day weather systems."
        ),
        (
            "Event-rate decomposition is exact for the chosen regime partition "
            "and observed radar-available timestamps, but it is not a causal "
            "decomposition."
        ),
        (
            "The 2024 record should be interpreted with particular care if its "
            "radar availability differs substantially from earlier years."
        ),
        (
            "Phase 14 diagnoses dataset/relationship shift. It does not itself "
            "define the final train/validation/test split; that is Phase 15."
        ),
    ]

    summary = {
        "phase_version": PHASE_VERSION,
        "phase6_dir": str(args.phase6_dir),
        "phase13_dir": str(args.phase13_dir),
        "n_predictor_timestamps": int(len(predictors)),
        "n_radar_timestamps": int(len(radar)),
        "n_regime_assignment_rows": int(len(assignments)),
        "available_predictors": pred_cols,
        "dataset_start_utc": str(start),
        "dataset_end_utc": str(end),
        "bootstrap_reps": int(args.bootstrap_reps),
        "bootstrap_block": "UTC_day",
        "psi_bins": int(args.psi_bins),
        "comparisons": comparisons.to_dict(orient="records"),
        "scope": (
            "coverage, marginal ERA5, regime-composition and within-regime "
            "radar shift diagnostics"
        ),
        "warnings": warnings,
        "methodological_notes": [
            (
                "The F5-oriented comparison uses 2011-2022 as reference and "
                "2023-2024 as evaluation because Phase 12 showed its strongest "
                "performance degradation in the final forward fold."
            ),
            (
                "Year-specific comparisons use 2011-2020 as a fixed historical "
                "reference to make 2021, 2022, 2023 and 2024 directly comparable."
            ),
            (
                "Predictor shift reports standardized mean difference, KS, PSI "
                "and standardized quantile shifts. No single statistic should "
                "be treated as a standalone definition of distribution shift."
            ),
            (
                "Regime-composition shift reports Jensen-Shannon divergence, "
                "total-variation distance and per-regime prevalence deltas."
            ),
            (
                "Conditional radar shift compares P(event | regime) between "
                "reference and evaluation windows with UTC-day block-bootstrap "
                "confidence intervals."
            ),
            (
                "The symmetric Shapley decomposition separates the observed event "
                "rate change into regime-composition and within-regime components "
                "that sum exactly up to numerical rounding."
            ),
        ],
    }

    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    logger.info("=" * 92)
    logger.info("PHASE 14 COMPLETE")
    logger.info("Output: %s", args.output_dir)
    logger.info("=" * 92)


if __name__ == "__main__":
    main()
