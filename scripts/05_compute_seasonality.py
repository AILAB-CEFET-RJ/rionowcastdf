#!/usr/bin/env python3
"""
CorrDiff - Fase 5
Sazonalidade condicionada à disponibilidade observacional
============================================================

Objetivos
---------
1. Quantificar a disponibilidade temporal por mês, ano-mês e estação.
2. Calcular frequências de eco condicionadas à existência de radar:
       P(evento | radar disponível, mês/estação)
3. Caracterizar intensidade do radar ao longo do ciclo anual.
4. Caracterizar sazonalidade dos 12 canais ERA5 e das 8 variáveis derivadas
   por meio da média espacial de cada timestamp.
5. Preservar a semântica dos limiares fixos no domínio armazenado log1p
   float32, evitando artefatos numéricos de expm1 nas fronteiras 25/30 etc.

Princípios metodológicos
------------------------
- Um timestamp disponível é a unidade temporal principal.
- Todos os patches do mesmo timestamp permanecem juntos.
- Taxas pixel-a-pixel usam a distribuição armazenada em patches, portanto
  continuam ponderadas pelo overlap espacial. Isso é explicitamente reportado.
- A frequência temporal de "algum evento no campo armazenado" é calculada por
  timestamp e não sofre multiplicação artificial pelo número de patches.
- Cobertura observacional é sempre reportada junto das frequências de evento.
- Estações meteorológicas do Hemisfério Sul:
    DJF = verão, MAM = outono, JJA = inverno, SON = primavera.
- Para climatologia sazonal, apenas season_years completos dentro do período
  configurado são usados por padrão, evitando DJF parcial nas bordas.

Saídas
------
analysis_outputs/05_seasonality/
├── analysis_summary.json
├── timestamp_event_metrics.parquet
├── monthly_coverage.parquet
├── year_month_coverage.parquet
├── season_year_coverage.parquet
├── monthly_event_rates.parquet
├── seasonal_event_rates.parquet
├── year_month_event_rates.parquet
├── predictor_timestamp_means.parquet
├── monthly_predictor_statistics.parquet
├── seasonal_predictor_statistics.parquet
└── phase5.log
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


PHASE_VERSION = "phase5-seasonality-v1-availability-conditioned"

RAW_CHANNELS = [
    "tcwv", "t2m", "u10", "v10",
    "t_850", "r_850", "u_850", "v_850",
    "t_500", "r_500", "u_500", "v_500",
]

DERIVED_NAMES = [
    "wind_speed_10",
    "wind_speed_850",
    "wind_speed_500",
    "delta_t_500_850",
    "delta_t_850_surface",
    "delta_r_500_850",
    "bulk_wind_diff_10_850",
    "bulk_wind_diff_850_500",
]

ALL_PREDICTORS = RAW_CHANNELS + DERIVED_NAMES

UNITS = {
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
    "wind_speed_10": "m s^-1",
    "wind_speed_850": "m s^-1",
    "wind_speed_500": "m s^-1",
    "delta_t_500_850": "K",
    "delta_t_850_surface": "K",
    "delta_r_500_850": "percentage points",
    "bulk_wind_diff_10_850": "m s^-1",
    "bulk_wind_diff_850_500": "m s^-1",
}

FIXED_THRESHOLDS = [20, 25, 30, 35, 40, 45]

SEASON_LABEL = {
    "DJF": "DJF_verao",
    "MAM": "MAM_outono",
    "JJA": "JJA_inverno",
    "SON": "SON_primavera",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "CorrDiff Fase 5: sazonalidade condicionada à disponibilidade "
            "observacional."
        )
    )
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs/05_seasonality"),
    )
    p.add_argument(
        "--block-size",
        type=int,
        default=0,
        help="Patches por bloco; 0 usa o primeiro chunk do target/input.",
    )
    p.add_argument(
        "--skip-predictors",
        action="store_true",
        help=(
            "Executa apenas disponibilidade + radar, sem ler todo o input ERA5."
        ),
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
    logger = logging.getLogger("corrdiff.phase5")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(path / "phase5.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def open_group(path: Path) -> Any:
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


def to_datetime_utc_naive(values: np.ndarray) -> pd.DatetimeIndex:
    unit = infer_datetime_unit(values)
    idx = pd.to_datetime(values, unit=unit, utc=True)
    return idx.tz_convert(None)


def season_code(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def add_calendar_columns(df: pd.DataFrame, time_col: str = "timestamp") -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out[time_col])
    out["year"] = dt.dt.year.astype(np.int16)
    out["month"] = dt.dt.month.astype(np.int8)
    out["year_month"] = dt.dt.to_period("M").astype(str)
    out["season_code"] = out["month"].map(season_code)
    out["season"] = out["season_code"].map(SEASON_LABEL)
    out["season_year"] = out["year"] + (out["month"] == 12).astype(np.int16)
    return out


def stored_threshold(value: float) -> np.float32:
    return np.float32(np.log1p(np.float32(value)))


def derive_block(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    u10, v10 = raw["u10"], raw["v10"]
    u850, v850 = raw["u_850"], raw["v_850"]
    u500, v500 = raw["u_500"], raw["v_500"]
    return {
        "wind_speed_10": np.hypot(u10, v10),
        "wind_speed_850": np.hypot(u850, v850),
        "wind_speed_500": np.hypot(u500, v500),
        "delta_t_500_850": raw["t_500"] - raw["t_850"],
        "delta_t_850_surface": raw["t_850"] - raw["t2m"],
        "delta_r_500_850": raw["r_500"] - raw["r_850"],
        "bulk_wind_diff_10_850": np.hypot(
            u850 - u10, v850 - v10
        ),
        "bulk_wind_diff_850_500": np.hypot(
            u500 - u850, v500 - v850
        ),
    }


def load_metadata(dataset_dir: Path) -> dict[str, Any]:
    path = dataset_dir / "metadata.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def get_period(metadata: dict[str, Any], observed: pd.DatetimeIndex) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    start_raw = metadata.get("start_date") or metadata.get("begin")
    end_raw = metadata.get("end_date") or metadata.get("end")
    freq = metadata.get("time_frequency") or "1h"

    start = pd.Timestamp(start_raw) if start_raw else pd.Timestamp(observed.min())
    end = pd.Timestamp(end_raw) if end_raw else pd.Timestamp(observed.max())

    if start.tzinfo is not None:
        start = start.tz_convert(None)
    if end.tzinfo is not None:
        end = end.tz_convert(None)

    return start, end, str(freq)


def build_expected_calendar(
    start: pd.Timestamp,
    end: pd.Timestamp,
    freq: str,
) -> pd.DataFrame:
    expected = pd.DataFrame({
        "timestamp": pd.date_range(start=start, end=end, freq=freq)
    })
    return add_calendar_columns(expected)


def coverage_tables(
    expected: pd.DataFrame,
    available: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Year-month
    exp_ym = (
        expected.groupby(["year", "month", "year_month"], as_index=False)
        .size()
        .rename(columns={"size": "expected_timestamps"})
    )
    av_ym = (
        available.groupby(["year", "month", "year_month"], as_index=False)
        .size()
        .rename(columns={"size": "available_timestamps"})
    )
    ym = exp_ym.merge(av_ym, how="left")
    ym["available_timestamps"] = ym["available_timestamps"].fillna(0).astype(int)
    ym["missing_timestamps"] = (
        ym["expected_timestamps"] - ym["available_timestamps"]
    )
    ym["coverage_ratio"] = (
        ym["available_timestamps"] / ym["expected_timestamps"]
    )

    # Monthly climatological coverage.
    monthly = (
        ym.groupby("month", as_index=False)[
            ["expected_timestamps", "available_timestamps", "missing_timestamps"]
        ]
        .sum()
    )
    monthly["coverage_ratio"] = (
        monthly["available_timestamps"] / monthly["expected_timestamps"]
    )

    # Season-year coverage.
    exp_sy = (
        expected.groupby(
            ["season_code", "season", "season_year"],
            as_index=False,
        )
        .agg(
            expected_timestamps=("timestamp", "size"),
            expected_months=("month", "nunique"),
        )
    )
    av_sy = (
        available.groupby(
            ["season_code", "season", "season_year"],
            as_index=False,
        )
        .agg(
            available_timestamps=("timestamp", "size"),
            available_months=("month", "nunique"),
        )
    )
    sy = exp_sy.merge(av_sy, how="left")
    sy["available_timestamps"] = sy["available_timestamps"].fillna(0).astype(int)
    sy["available_months"] = sy["available_months"].fillna(0).astype(int)
    sy["missing_timestamps"] = (
        sy["expected_timestamps"] - sy["available_timestamps"]
    )
    sy["coverage_ratio"] = (
        sy["available_timestamps"] / sy["expected_timestamps"]
    )
    sy["complete_in_configured_period"] = sy["expected_months"] == 3

    return monthly, ym, sy


def aggregate_event_group(
    frame: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    event_count_cols = [
        c for c in frame.columns if c.startswith("event_pixels_")
    ]
    patch_event_cols = [
        c for c in frame.columns if c.startswith("event_patches_")
    ]

    rows: list[dict[str, Any]] = []

    for keys, group in frame.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))

        valid_pixels = float(group["valid_pixels"].sum())
        total_patches = float(group["patch_count"].sum())
        positive_pixels = float(group["positive_pixels"].sum())
        positive_sum = float(group["positive_radar_sum"].sum())

        common = {
            **base,
            "available_timestamps": int(len(group)),
            "valid_pixels": int(valid_pixels),
            "positive_pixels": int(positive_pixels),
            "positive_pixel_rate": (
                positive_pixels / valid_pixels if valid_pixels else np.nan
            ),
            "timestamp_any_positive_rate": float(
                (group["positive_pixels"] > 0).mean()
            ),
            "mean_positive_radar": (
                positive_sum / positive_pixels if positive_pixels else np.nan
            ),
            "mean_timestamp_max_radar": float(group["max_radar"].mean()),
            "median_timestamp_max_radar": float(group["max_radar"].median()),
        }

        rows.append({
            **common,
            "event_id": "gt_0",
            "threshold": 0.0,
            "pixel_event_rate": common["positive_pixel_rate"],
            "timestamp_any_event_rate": common["timestamp_any_positive_rate"],
            "patch_any_event_rate": float(
                group["event_patches_gt_0"].sum() / total_patches
            ) if total_patches else np.nan,
        })

        for threshold in FIXED_THRESHOLDS:
            suffix = f"ge_{threshold}"
            event_pixels = float(group[f"event_pixels_{suffix}"].sum())
            rows.append({
                **common,
                "event_id": suffix,
                "threshold": float(threshold),
                "pixel_event_rate": (
                    event_pixels / valid_pixels if valid_pixels else np.nan
                ),
                "timestamp_any_event_rate": float(
                    (group[f"event_pixels_{suffix}"] > 0).mean()
                ),
                "patch_any_event_rate": float(
                    group[f"event_patches_{suffix}"].sum() / total_patches
                ) if total_patches else np.nan,
            })

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    setup_output(args.output_dir, args.overwrite)
    logger = setup_logger(args.output_dir)

    root = open_group(args.dataset_dir / "train.zarr")
    xds = root["input"]
    yds = root["target"]
    mds = root["mask"]
    tds = root["timestamps"]

    n = int(yds.shape[0])
    h, w = int(yds.shape[-2]), int(yds.shape[-1])

    timestamp_raw = np.asarray(tds[:])
    timestamp_dt = to_datetime_utc_naive(timestamp_raw)
    unique_raw, inverse = np.unique(timestamp_raw, return_inverse=True)
    unique_dt = to_datetime_utc_naive(unique_raw)

    n_timestamps = int(len(unique_raw))
    patch_count_by_ts = np.bincount(
        inverse,
        minlength=n_timestamps,
    ).astype(np.int32)

    metadata = load_metadata(args.dataset_dir)
    start, end, freq = get_period(metadata, unique_dt)
    expected = build_expected_calendar(start, end, freq)

    available = add_calendar_columns(
        pd.DataFrame({"timestamp": unique_dt})
    )
    monthly_cov, ym_cov, sy_cov = coverage_tables(expected, available)

    monthly_cov.to_parquet(
        args.output_dir / "monthly_coverage.parquet",
        index=False,
    )
    ym_cov.to_parquet(
        args.output_dir / "year_month_coverage.parquet",
        index=False,
    )
    sy_cov.to_parquet(
        args.output_dir / "season_year_coverage.parquet",
        index=False,
    )

    # ------------------------------------------------------------------
    # Radar aggregation per unique timestamp
    # ------------------------------------------------------------------
    target_chunk = int(getattr(yds, "chunks", (256,))[0])
    block_size = args.block_size or target_chunk
    blocks = math.ceil(n / block_size)

    valid_pixels_ts = np.zeros(n_timestamps, dtype=np.int64)
    positive_pixels_ts = np.zeros(n_timestamps, dtype=np.int64)
    positive_sum_ts = np.zeros(n_timestamps, dtype=np.float64)
    max_radar_ts = np.zeros(n_timestamps, dtype=np.float32)

    event_pixels_ts = {
        threshold: np.zeros(n_timestamps, dtype=np.int64)
        for threshold in FIXED_THRESHOLDS
    }
    event_patches_ts = {
        "gt_0": np.zeros(n_timestamps, dtype=np.int32),
        **{
            f"ge_{threshold}": np.zeros(n_timestamps, dtype=np.int32)
            for threshold in FIXED_THRESHOLDS
        },
    }

    logger.info("=" * 72)
    logger.info("PHASE 5 - RADAR SEASONALITY")
    logger.info("Patches            : %d", n)
    logger.info("Unique timestamps  : %d", n_timestamps)
    logger.info("Blocks             : %d", blocks)
    logger.info("=" * 72)

    for block_no, start_i in enumerate(range(0, n, block_size), start=1):
        end_i = min(n, start_i + block_size)
        gids = inverse[start_i:end_i]

        stored = np.asarray(
            yds[start_i:end_i, 0],
            dtype=np.float32,
        )
        mask = (
            np.asarray(
                mds[start_i:end_i, 0],
                dtype=np.float32,
            ) > 0.5
        )
        valid = mask & np.isfinite(stored)

        radar = np.expm1(stored).astype(np.float32, copy=False)
        positive = valid & (stored > np.float32(0.0))

        valid_patch = valid.sum(axis=(1, 2), dtype=np.int64)
        positive_patch = positive.sum(axis=(1, 2), dtype=np.int64)
        positive_sum_patch = np.where(
            positive,
            radar,
            np.float32(0.0),
        ).sum(axis=(1, 2), dtype=np.float64)

        radar_valid_for_max = np.where(valid, radar, -np.inf)
        max_patch = radar_valid_for_max.max(axis=(1, 2))
        max_patch[~np.isfinite(max_patch)] = 0.0

        np.add.at(valid_pixels_ts, gids, valid_patch)
        np.add.at(positive_pixels_ts, gids, positive_patch)
        np.add.at(positive_sum_ts, gids, positive_sum_patch)
        np.maximum.at(max_radar_ts, gids, max_patch)

        np.add.at(
            event_patches_ts["gt_0"],
            gids,
            positive.reshape(len(gids), -1).any(axis=1).astype(np.int32),
        )

        for threshold in FIXED_THRESHOLDS:
            event = valid & (
                stored >= stored_threshold(float(threshold))
            )
            counts = event.sum(axis=(1, 2), dtype=np.int64)
            np.add.at(event_pixels_ts[threshold], gids, counts)
            np.add.at(
                event_patches_ts[f"ge_{threshold}"],
                gids,
                event.reshape(len(gids), -1).any(axis=1).astype(np.int32),
            )

        if block_no % 250 == 0 or block_no == blocks:
            logger.info(
                "Radar: %d/%d blocks processed",
                block_no,
                blocks,
            )

    ts_df = add_calendar_columns(
        pd.DataFrame({
            "timestamp": unique_dt,
            "patch_count": patch_count_by_ts,
            "valid_pixels": valid_pixels_ts,
            "positive_pixels": positive_pixels_ts,
            "positive_radar_sum": positive_sum_ts,
            "max_radar": max_radar_ts,
        })
    )

    ts_df["positive_pixel_fraction"] = np.divide(
        ts_df["positive_pixels"],
        ts_df["valid_pixels"],
        out=np.full(len(ts_df), np.nan, dtype=float),
        where=ts_df["valid_pixels"].to_numpy() > 0,
    )
    ts_df["mean_positive_radar"] = np.divide(
        ts_df["positive_radar_sum"],
        ts_df["positive_pixels"],
        out=np.full(len(ts_df), np.nan, dtype=float),
        where=ts_df["positive_pixels"].to_numpy() > 0,
    )
    ts_df["event_patches_gt_0"] = event_patches_ts["gt_0"]

    for threshold in FIXED_THRESHOLDS:
        suffix = f"ge_{threshold}"
        ts_df[f"event_pixels_{suffix}"] = event_pixels_ts[threshold]
        ts_df[f"event_patches_{suffix}"] = event_patches_ts[suffix]
        ts_df[f"event_pixel_fraction_{suffix}"] = np.divide(
            event_pixels_ts[threshold],
            valid_pixels_ts,
            out=np.full(n_timestamps, np.nan, dtype=float),
            where=valid_pixels_ts > 0,
        )

    ts_df.to_parquet(
        args.output_dir / "timestamp_event_metrics.parquet",
        index=False,
    )

    monthly_events = aggregate_event_group(ts_df, ["month"])
    ym_events = aggregate_event_group(
        ts_df,
        ["year", "month", "year_month"],
    )

    # Seasonal climatology: exclude incomplete edge season-years.
    complete_sy = sy_cov.loc[
        sy_cov["complete_in_configured_period"],
        ["season_code", "season_year"],
    ].drop_duplicates()

    seasonal_source = ts_df.merge(
        complete_sy,
        on=["season_code", "season_year"],
        how="inner",
    )
    seasonal_events = aggregate_event_group(
        seasonal_source,
        ["season_code", "season"],
    )

    monthly_events = monthly_events.merge(
        monthly_cov[[
            "month",
            "expected_timestamps",
            "available_timestamps",
            "coverage_ratio",
        ]],
        on=["month", "available_timestamps"],
        how="left",
    )

    seasonal_cov_complete = (
        sy_cov[sy_cov["complete_in_configured_period"]]
        .groupby(["season_code", "season"], as_index=False)[
            ["expected_timestamps", "available_timestamps"]
        ]
        .sum()
    )
    seasonal_cov_complete["coverage_ratio"] = (
        seasonal_cov_complete["available_timestamps"]
        / seasonal_cov_complete["expected_timestamps"]
    )

    seasonal_events = seasonal_events.merge(
        seasonal_cov_complete,
        on=["season_code", "season", "available_timestamps"],
        how="left",
    )

    ym_events = ym_events.merge(
        ym_cov[[
            "year", "month", "year_month",
            "expected_timestamps", "available_timestamps", "coverage_ratio",
        ]],
        on=["year", "month", "year_month", "available_timestamps"],
        how="left",
    )

    monthly_events.to_parquet(
        args.output_dir / "monthly_event_rates.parquet",
        index=False,
    )
    seasonal_events.to_parquet(
        args.output_dir / "seasonal_event_rates.parquet",
        index=False,
    )
    ym_events.to_parquet(
        args.output_dir / "year_month_event_rates.parquet",
        index=False,
    )

    # ------------------------------------------------------------------
    # ERA5 / derived predictor seasonality
    # ------------------------------------------------------------------
    predictor_summary_available = False
    predictor_ts_path = args.output_dir / "predictor_timestamp_means.parquet"

    if not args.skip_predictors:
        channels = list(root.attrs.get("channels", []))
        if not channels:
            channels = metadata.get("channels") or metadata.get("input_channels") or []
        missing = [name for name in RAW_CHANNELS if name not in channels]
        if missing:
            raise RuntimeError(
                f"Required input channels are missing: {missing}"
            )

        channel_index = {
            name: channels.index(name)
            for name in RAW_CHANNELS
        }

        predictor_sum = np.zeros(
            (n_timestamps, len(ALL_PREDICTORS)),
            dtype=np.float64,
        )
        predictor_count = np.zeros(
            (n_timestamps, len(ALL_PREDICTORS)),
            dtype=np.int64,
        )

        input_chunk = int(getattr(xds, "chunks", (256,))[0])
        input_block_size = args.block_size or input_chunk
        input_blocks = math.ceil(n / input_block_size)

        logger.info("=" * 72)
        logger.info("PHASE 5 - PREDICTOR SEASONALITY")
        logger.info("Input blocks: %d", input_blocks)
        logger.info("=" * 72)

        for block_no, start_i in enumerate(
            range(0, n, input_block_size),
            start=1,
        ):
            end_i = min(n, start_i + input_block_size)
            gids = inverse[start_i:end_i]

            block = np.asarray(
                xds[start_i:end_i],
                dtype=np.float32,
            )
            raw = {
                name: block[:, channel_index[name]]
                for name in RAW_CHANNELS
            }
            derived = derive_block(raw)

            for j, name in enumerate(ALL_PREDICTORS):
                arr = raw[name] if name in raw else derived[name]
                finite = np.isfinite(arr)
                patch_sum = np.where(
                    finite,
                    arr,
                    np.float32(0.0),
                ).sum(axis=(1, 2), dtype=np.float64)
                patch_count = finite.sum(
                    axis=(1, 2),
                    dtype=np.int64,
                )
                np.add.at(predictor_sum[:, j], gids, patch_sum)
                np.add.at(predictor_count[:, j], gids, patch_count)

            if block_no % 250 == 0 or block_no == input_blocks:
                logger.info(
                    "Input: %d/%d blocks processed",
                    block_no,
                    input_blocks,
                )

        predictor_mean = np.divide(
            predictor_sum,
            predictor_count,
            out=np.full_like(predictor_sum, np.nan),
            where=predictor_count > 0,
        )

        pred_df = add_calendar_columns(
            pd.DataFrame({"timestamp": unique_dt})
        )
        for j, name in enumerate(ALL_PREDICTORS):
            pred_df[name] = predictor_mean[:, j]

        pred_df.to_parquet(predictor_ts_path, index=False)

        monthly_rows: list[dict[str, Any]] = []
        seasonal_rows: list[dict[str, Any]] = []

        for month, group in pred_df.groupby("month", sort=True):
            for name in ALL_PREDICTORS:
                values = group[name].dropna().to_numpy(dtype=float)
                if values.size == 0:
                    continue
                monthly_rows.append({
                    "month": int(month),
                    "predictor": name,
                    "source": "raw" if name in RAW_CHANNELS else "derived",
                    "unit": UNITS[name],
                    "available_timestamps": int(values.size),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if values.size > 1 else np.nan,
                    "p10": float(np.quantile(values, 0.10)),
                    "median": float(np.quantile(values, 0.50)),
                    "p90": float(np.quantile(values, 0.90)),
                })

        pred_seasonal_source = pred_df.merge(
            complete_sy,
            on=["season_code", "season_year"],
            how="inner",
        )

        for (scode, season), group in pred_seasonal_source.groupby(
            ["season_code", "season"],
            sort=True,
        ):
            for name in ALL_PREDICTORS:
                values = group[name].dropna().to_numpy(dtype=float)
                if values.size == 0:
                    continue
                seasonal_rows.append({
                    "season_code": scode,
                    "season": season,
                    "predictor": name,
                    "source": "raw" if name in RAW_CHANNELS else "derived",
                    "unit": UNITS[name],
                    "available_timestamps": int(values.size),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if values.size > 1 else np.nan,
                    "p10": float(np.quantile(values, 0.10)),
                    "median": float(np.quantile(values, 0.50)),
                    "p90": float(np.quantile(values, 0.90)),
                })

        pd.DataFrame(monthly_rows).to_parquet(
            args.output_dir / "monthly_predictor_statistics.parquet",
            index=False,
        )
        pd.DataFrame(seasonal_rows).to_parquet(
            args.output_dir / "seasonal_predictor_statistics.parquet",
            index=False,
        )
        predictor_summary_available = True

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    gt0_month = monthly_events[
        monthly_events["event_id"] == "gt_0"
    ].copy()
    ge40_month = monthly_events[
        monthly_events["event_id"] == "ge_40"
    ].copy()
    ge45_month = monthly_events[
        monthly_events["event_id"] == "ge_45"
    ].copy()

    warnings: list[str] = []
    if float(monthly_cov["coverage_ratio"].min()) < 0.5:
        warnings.append(
            "At least one climatological month has <50% temporal coverage."
        )
    if float(ym_cov["coverage_ratio"].min()) < 0.25:
        warnings.append(
            "At least one year-month has <25% temporal coverage; inspect "
            "year_month_coverage.parquet before interpreting anomalies."
        )
    if int(patch_count_by_ts.min()) != int(patch_count_by_ts.max()):
        warnings.append(
            "Patch count varies by timestamp; pixel rates are therefore not "
            "equally weighted in time."
        )

    summary = {
        "phase_version": PHASE_VERSION,
        "dataset_patches": n,
        "available_unique_timestamps": n_timestamps,
        "expected_timestamps": int(len(expected)),
        "temporal_coverage_ratio": float(n_timestamps / len(expected)),
        "period_start": str(start),
        "period_end": str(end),
        "time_frequency": freq,
        "patches_per_timestamp_min": int(patch_count_by_ts.min()),
        "patches_per_timestamp_median": float(np.median(patch_count_by_ts)),
        "patches_per_timestamp_max": int(patch_count_by_ts.max()),
        "season_definition": SEASON_LABEL,
        "seasonal_climatology_complete_season_years_only": True,
        "fixed_threshold_comparison_domain": "stored_log1p_float32",
        "radar_continuous_domain": (
            "expm1(stored_target): post-clip numerical radar-legend values; "
            "physical unit not assumed"
        ),
        "predictor_seasonality_computed": predictor_summary_available,
        "monthly_lowest_coverage": (
            monthly_cov.sort_values("coverage_ratio")
            .head(3)
            .to_dict(orient="records")
        ),
        "monthly_highest_coverage": (
            monthly_cov.sort_values("coverage_ratio", ascending=False)
            .head(3)
            .to_dict(orient="records")
        ),
        "months_highest_positive_pixel_rate": (
            gt0_month.sort_values("pixel_event_rate", ascending=False)
            .head(3)[["month", "pixel_event_rate", "coverage_ratio"]]
            .to_dict(orient="records")
        ),
        "months_highest_ge40_pixel_rate": (
            ge40_month.sort_values("pixel_event_rate", ascending=False)
            .head(3)[["month", "pixel_event_rate", "coverage_ratio"]]
            .to_dict(orient="records")
        ),
        "months_highest_ge45_pixel_rate": (
            ge45_month.sort_values("pixel_event_rate", ascending=False)
            .head(3)[["month", "pixel_event_rate", "coverage_ratio"]]
            .to_dict(orient="records")
        ),
        "warnings": warnings,
        "methodological_notes": [
            (
                "Event frequencies are conditioned on available radar "
                "timestamps; raw monthly event counts should not be used as "
                "climatological frequencies."
            ),
            (
                "Pixel event rates remain diagnostics of the stored overlapping "
                "patch distribution, not de-duplicated full-field climatology."
            ),
            (
                "Timestamp_any_event_rate is the preferred temporal occurrence "
                "metric when the question is whether an event was present at "
                "an available timestamp."
            ),
            (
                "Predictor statistics are computed from one spatial mean per "
                "available timestamp, preventing months with more patches from "
                "receiving extra temporal weight."
            ),
        ],
    }

    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("=" * 72)
    logger.info("PHASE 5 COMPLETE")
    logger.info(
        "Temporal coverage: %.2f%%",
        100.0 * n_timestamps / len(expected),
    )
    logger.info("Output: %s", args.output_dir)
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
