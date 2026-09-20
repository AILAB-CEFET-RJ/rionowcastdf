#!/usr/bin/env python3
"""
CorrDiff - Fase 8
Baselines temporais de persistência e skill preditivo
=====================================================

Objetivo
--------
Transformar a dependência temporal identificada na Fase 7 em baselines
preditivos explícitos, avaliados com divisão cronológica e sem vazamento de
informação futura.

A Fase 8 NÃO pretende definir o split final da dissertação. O split usado aqui
é provisório e serve apenas para medir skill de baselines temporais. A definição
formal dos splits permanece reservada para a Fase 15.

Baselines binários
------------------
- train_prevalence
- climatology_month_hour
- persistence_1h
- persistence_mean_lags
- persistence_exp_decay
- logistic_temporal

Baselines contínuos
-------------------
- train_mean
- climatology_month_hour
- persistence_1h
- persistence_mean_lags
- persistence_exp_decay
- ridge_temporal

Eventos
-------
>0, >=20, >=30, >=40 e >=45 dBZ.

Métricas contínuas
------------------
- max_dbz
- positive_pixel_fraction
- event_pixel_fraction_ge_30
- event_pixel_fraction_ge_40
- event_pixel_fraction_ge_45

Princípios metodológicos
------------------------
1. Todos os pares de lag são exatos em UTC.
2. O cohort comum exige a presença de todos os lags configurados.
3. O split é definido pelo timestamp alvo t.
4. Climatologia e modelos são ajustados somente no treino.
5. Radar passado em t-k é permitido como feature porque representa informação
   operacionalmente disponível antes de t.
6. Skill é reportado em relação à climatologia mês x hora local treinada apenas
   no período de treino.
7. As métricas de fração de pixels continuam descrevendo a distribuição de
   patches sobrepostos, não um campo de radar espacialmente de-duplicado.

Dependência
-----------
Requer:
analysis_outputs/06_diurnal/timestamp_event_metrics.parquet

Opcional:
analysis_outputs/06_diurnal/analysis_summary.json

Saídas
------
analysis_outputs/08_persistence_baselines/
├── analysis_summary.json
├── evaluation_cohort_summary.parquet
├── binary_baseline_metrics.parquet
├── binary_baseline_metrics_by_season.parquet
├── binary_calibration.parquet
├── continuous_baseline_metrics.parquet
├── continuous_baseline_metrics_by_condition.parquet
├── model_status.parquet
└── phase8.log
"""

from __future__ import annotations

import argparse
import hashlib
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
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )
except ImportError as exc:
    raise SystemExit(
        "Fase 8 requer scikit-learn. Instale com: pip install scikit-learn"
    ) from exc


PHASE_VERSION = "phase8-persistence-baselines-v2-continuity-audited"

EVENT_SPECS = {
    "gt_0": ("positive_pixels", 0.0),
    "ge_20": ("event_pixels_ge_20", 20.0),
    "ge_30": ("event_pixels_ge_30", 30.0),
    "ge_40": ("event_pixels_ge_40", 40.0),
    "ge_45": ("event_pixels_ge_45", 45.0),
}

CONTINUOUS_METRICS = [
    "max_dbz",
    "positive_pixel_fraction",
    "event_pixel_fraction_ge_30",
    "event_pixel_fraction_ge_40",
    "event_pixel_fraction_ge_45",
]

SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CorrDiff Fase 8: baselines temporais de persistência."
    )
    p.add_argument(
        "--phase6-dir",
        type=Path,
        default=Path("analysis_outputs/06_diurnal"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs/08_persistence_baselines"),
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("datasets/corrdiff_2011_2024"),
        help=(
            "Diretório do dataset contendo train.zarr. Usado pela auditoria "
            "de continuidade target(t) vs target(t-1h)."
        ),
    )
    p.add_argument(
        "--lags",
        default="1,2,3,6",
        help="Lags passados usados no cohort e baselines. Deve conter 1.",
    )
    p.add_argument(
        "--train-end",
        default="2019-12-31T23:00:00Z",
        help="Último timestamp UTC do treino.",
    )
    p.add_argument(
        "--val-end",
        default="2021-12-31T23:00:00Z",
        help="Último timestamp UTC da validação.",
    )
    p.add_argument(
        "--decay-tau-hours",
        type=float,
        default=3.0,
        help="Tau da média exponencial dos lags.",
    )
    p.add_argument(
        "--ridge-alpha",
        type=float,
        default=1.0,
        help="Regularização L2 do baseline Ridge temporal.",
    )
    p.add_argument(
        "--audit-batch-timestamps",
        type=int,
        default=512,
        help="Número de timestamps por lote na auditoria de hashes do target.",
    )
    p.add_argument(
        "--skip-target-audit",
        action="store_true",
        help="Executa apenas os baselines, sem auditoria direta do train.zarr.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
    )
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
    logger = logging.getLogger("corrdiff.phase8")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(path / "phase8.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def parse_lags(text: str) -> list[int]:
    vals = sorted(
        set(int(x.strip()) for x in text.split(",") if x.strip())
    )
    if not vals or any(v <= 0 for v in vals):
        raise ValueError("Lags must be positive integers.")
    if 1 not in vals:
        raise ValueError("Phase 8 requires lag 1h for persistence_1h.")
    return vals


def utc_timestamp(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def load_phase6(
    phase6_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    event_path = phase6_dir / "timestamp_event_metrics.parquet"
    summary_path = phase6_dir / "analysis_summary.json"

    if not event_path.exists():
        raise FileNotFoundError(
            f"Phase 6 event metrics not found: {event_path}"
        )

    df = pd.read_parquet(event_path)
    summary: dict[str, Any] = {}
    if summary_path.exists():
        summary = json.loads(
            summary_path.read_text(encoding="utf-8")
        )

    # Compatibility with Phase 6 v1 output naming.
    if (
        "positive_pixel_fraction" not in df.columns
        and "event_pixel_fraction_gt_0" in df.columns
    ):
        df["positive_pixel_fraction"] = (
            df["event_pixel_fraction_gt_0"]
        )

    required = {
        "timestamp_utc",
        "month_local",
        "hour_local",
        "season_code",
        "positive_pixels",
        "event_pixels_ge_20",
        "event_pixels_ge_30",
        "event_pixels_ge_40",
        "event_pixels_ge_45",
        *CONTINUOUS_METRICS,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(
            f"Phase 6 event metrics missing required columns: {missing}"
        )

    df = df.copy()
    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"],
        utc=True,
    )

    for event_id, (count_col, _) in EVENT_SPECS.items():
        df[f"event_{event_id}"] = (
            df[count_col].to_numpy() > 0
        ).astype(np.int8)

    return (
        df.sort_values("timestamp_utc")
        .drop_duplicates("timestamp_utc")
        .reset_index(drop=True),
        summary,
    )


def build_common_lag_cohort(
    base: pd.DataFrame,
    lags: list[int],
) -> pd.DataFrame:
    keep = [
        "timestamp_utc",
        "month_local",
        "hour_local",
        "season_code",
        *CONTINUOUS_METRICS,
        *[f"event_{e}" for e in EVENT_SPECS],
    ]
    out = base[keep].copy()

    lag_source_cols = (
        CONTINUOUS_METRICS
        + [f"event_{e}" for e in EVENT_SPECS]
    )

    for lag in lags:
        lagged = base[
            ["timestamp_utc", *lag_source_cols]
        ].copy()
        lagged["timestamp_utc"] = (
            lagged["timestamp_utc"]
            + pd.to_timedelta(lag, unit="h")
        )
        lagged = lagged.rename(
            columns={
                c: f"{c}__lag{lag}"
                for c in lag_source_cols
            }
        )
        out = out.merge(
            lagged,
            on="timestamp_utc",
            how="inner",
            validate="one_to_one",
        )

    return out.sort_values("timestamp_utc").reset_index(drop=True)


def assign_split(
    df: pd.DataFrame,
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
) -> pd.DataFrame:
    if val_end <= train_end:
        raise ValueError("--val-end must be after --train-end.")

    out = df.copy()
    ts = out["timestamp_utc"]

    out["split"] = np.select(
        [
            ts <= train_end,
            (ts > train_end) & (ts <= val_end),
            ts > val_end,
        ],
        ["train", "val", "test"],
        default="unassigned",
    )
    out = out[out["split"] != "unassigned"].copy()

    for split in ("train", "val", "test"):
        if not (out["split"] == split).any():
            raise RuntimeError(
                f"Split {split!r} has no rows with current cutoffs."
            )
    return out


def fit_climatology(
    train: pd.DataFrame,
    target: str,
) -> tuple[pd.DataFrame, float]:
    global_mean = float(train[target].mean())
    table = (
        train.groupby(
            ["month_local", "hour_local"],
            as_index=False,
            observed=True,
        )[target]
        .mean()
        .rename(columns={target: "climatology_prediction"})
    )
    return table, global_mean


def apply_climatology(
    frame: pd.DataFrame,
    table: pd.DataFrame,
    fallback: float,
) -> np.ndarray:
    tmp = frame[["month_local", "hour_local"]].merge(
        table,
        on=["month_local", "hour_local"],
        how="left",
        sort=False,
    )
    return (
        tmp["climatology_prediction"]
        .fillna(fallback)
        .to_numpy(dtype=float)
    )


def exp_decay_weights(
    lags: list[int],
    tau: float,
) -> np.ndarray:
    if tau <= 0:
        raise ValueError("--decay-tau-hours must be > 0.")
    w = np.exp(-np.asarray(lags, dtype=float) / tau)
    return w / w.sum()


def safe_corr(
    y: np.ndarray,
    pred: np.ndarray,
    method: str,
) -> float:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(pred)
    if mask.sum() < 3:
        return np.nan

    a = pd.Series(y[mask])
    b = pd.Series(pred[mask])
    if a.nunique() < 2 or b.nunique() < 2:
        return np.nan
    return float(a.corr(b, method=method))


def expected_calibration_error(
    y: np.ndarray,
    prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, list[dict[str, Any]]]:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(prob, dtype=float), 0.0, 1.0)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # p=1 must fall into final bin.
    idx = np.minimum(
        np.digitize(p, edges[1:-1], right=False),
        n_bins - 1,
    )

    rows: list[dict[str, Any]] = []
    ece = 0.0
    n = len(y)

    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            rows.append({
                "bin": b,
                "bin_left": float(edges[b]),
                "bin_right": float(edges[b + 1]),
                "count": 0,
                "mean_prediction": np.nan,
                "observed_rate": np.nan,
            })
            continue

        mp = float(p[mask].mean())
        obs = float(y[mask].mean())
        ece += (count / n) * abs(mp - obs)
        rows.append({
            "bin": b,
            "bin_left": float(edges[b]),
            "bin_right": float(edges[b + 1]),
            "count": count,
            "mean_prediction": mp,
            "observed_rate": obs,
        })

    return float(ece), rows


def binary_metric_row(
    y: np.ndarray,
    prob: np.ndarray,
) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(prob, dtype=float), 0.0, 1.0)

    row = {
        "n": int(len(y)),
        "event_rate": float(y.mean()),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(
            log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])
        ),
    }

    if np.unique(y).size >= 2:
        row["roc_auc"] = float(roc_auc_score(y, p))
        row["average_precision"] = float(
            average_precision_score(y, p)
        )
    else:
        row["roc_auc"] = np.nan
        row["average_precision"] = np.nan

    ece, _ = expected_calibration_error(y, p, 10)
    row["ece_10bins"] = ece
    return row


def continuous_metric_row(
    y: np.ndarray,
    pred: np.ndarray,
) -> dict[str, Any]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(pred, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)

    y = y[mask]
    p = p[mask]
    if len(y) == 0:
        return {
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
            "pearson_r": np.nan,
            "spearman_rho": np.nan,
        }

    err = p - y
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(np.mean(err)),
        "pearson_r": safe_corr(y, p, "pearson"),
        "spearman_rho": safe_corr(y, p, "spearman"),
    }


def fit_binary_models(
    train: pd.DataFrame,
    frame: pd.DataFrame,
    event_id: str,
    lags: list[int],
    decay_weights: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    target = f"event_{event_id}"
    y_train = train[target].to_numpy(dtype=int)

    clim_table, prevalence = fit_climatology(train, target)
    clim_frame = apply_climatology(frame, clim_table, prevalence)
    clim_train = apply_climatology(train, clim_table, prevalence)

    lag_cols = [f"{target}__lag{lag}" for lag in lags]
    lag_matrix = frame[lag_cols].to_numpy(dtype=float)
    lag_matrix_train = train[lag_cols].to_numpy(dtype=float)

    preds: dict[str, np.ndarray] = {
        "train_prevalence": np.full(len(frame), prevalence),
        "climatology_month_hour": clim_frame,
        "persistence_1h": frame[
            f"{target}__lag1"
        ].to_numpy(dtype=float),
        "persistence_mean_lags": lag_matrix.mean(axis=1),
        "persistence_exp_decay": lag_matrix @ decay_weights,
    }

    status = {
        "target": event_id,
        "model": "logistic_temporal",
        "status": "ok",
        "detail": "",
    }

    x_train = np.column_stack(
        [lag_matrix_train, clim_train]
    )
    x_frame = np.column_stack(
        [lag_matrix, clim_frame]
    )

    if np.unique(y_train).size < 2:
        preds["logistic_temporal"] = clim_frame.copy()
        status["status"] = "fallback"
        status["detail"] = "training target has one class"
    else:
        model = LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            random_state=42,
        )
        model.fit(x_train, y_train)
        preds["logistic_temporal"] = model.predict_proba(
            x_frame
        )[:, 1]
        status["detail"] = (
            f"features={lag_cols}+climatology; "
            f"coef={model.coef_[0].tolist()}; "
            f"intercept={float(model.intercept_[0])}"
        )

    return preds, status


def fit_continuous_models(
    train: pd.DataFrame,
    frame: pd.DataFrame,
    metric: str,
    lags: list[int],
    decay_weights: np.ndarray,
    ridge_alpha: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    clim_table, train_mean = fit_climatology(train, metric)
    clim_frame = apply_climatology(frame, clim_table, train_mean)
    clim_train = apply_climatology(train, clim_table, train_mean)

    lag_cols = [f"{metric}__lag{lag}" for lag in lags]
    lag_matrix = frame[lag_cols].to_numpy(dtype=float)
    lag_matrix_train = train[lag_cols].to_numpy(dtype=float)

    preds: dict[str, np.ndarray] = {
        "train_mean": np.full(len(frame), train_mean),
        "climatology_month_hour": clim_frame,
        "persistence_1h": frame[
            f"{metric}__lag1"
        ].to_numpy(dtype=float),
        "persistence_mean_lags": np.nanmean(
            lag_matrix,
            axis=1,
        ),
        "persistence_exp_decay": np.nansum(
            lag_matrix * decay_weights[None, :],
            axis=1,
        ),
    }

    status = {
        "target": metric,
        "model": "ridge_temporal",
        "status": "ok",
        "detail": "",
    }

    y_train = train[metric].to_numpy(dtype=float)
    finite_train = (
        np.isfinite(y_train)
        & np.isfinite(lag_matrix_train).all(axis=1)
        & np.isfinite(clim_train)
    )

    if finite_train.sum() < 10:
        preds["ridge_temporal"] = clim_frame.copy()
        status["status"] = "fallback"
        status["detail"] = "fewer than 10 finite training rows"
    else:
        x_train = np.column_stack(
            [lag_matrix_train[finite_train], clim_train[finite_train]]
        )
        y_fit = y_train[finite_train]

        model = Ridge(alpha=ridge_alpha)
        model.fit(x_train, y_fit)

        finite_frame = (
            np.isfinite(lag_matrix).all(axis=1)
            & np.isfinite(clim_frame)
        )
        ridge_pred = clim_frame.copy()
        if finite_frame.any():
            ridge_pred[finite_frame] = model.predict(
                np.column_stack(
                    [lag_matrix[finite_frame], clim_frame[finite_frame]]
                )
            )
        preds["ridge_temporal"] = ridge_pred
        status["detail"] = (
            f"features={lag_cols}+climatology; "
            f"alpha={ridge_alpha}; "
            f"coef={model.coef_.tolist()}; "
            f"intercept={float(model.intercept_)}"
        )

    return preds, status



# ---------------------------------------------------------------------------
# Auditoria de continuidade do radar
# ---------------------------------------------------------------------------

def _open_zarr_group(path: Path):
    try:
        return zarr.open_group(str(path), mode="r")
    except Exception:
        return zarr.open(str(path), mode="r")


def _infer_datetime_unit(values: np.ndarray) -> str:
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


def _to_datetime_utc(values: np.ndarray) -> pd.DatetimeIndex:
    unit = _infer_datetime_unit(values)
    return pd.DatetimeIndex(pd.to_datetime(values, unit=unit, utc=True))


def _hash_target_groups(
    dataset_dir: Path,
    batch_timestamps: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    Gera um hash BLAKE2b-128 para o conjunto ordenado de patches armazenados
    em cada timestamp. O objetivo é detectar reutilização exata do target.

    O hash é deliberadamente calculado sobre bytes float32 armazenados, sem
    expm1, arredondamento ou thresholding.
    """
    zarr_path = dataset_dir / "train.zarr"
    if not zarr_path.exists():
        raise FileNotFoundError(
            f"train.zarr not found for target audit: {zarr_path}"
        )

    root = _open_zarr_group(zarr_path)
    target = root["target"]
    timestamps = np.asarray(root["timestamps"][:])

    if len(timestamps) != int(target.shape[0]):
        raise RuntimeError(
            "target and timestamps have inconsistent first dimension."
        )

    n_patches = len(timestamps)
    if n_patches == 0:
        raise RuntimeError("Empty target dataset.")

    change = np.flatnonzero(timestamps[1:] != timestamps[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, n_patches]
    unique_raw = timestamps[starts]
    unique_utc = _to_datetime_utc(unique_raw)

    if batch_timestamps <= 0:
        raise ValueError("--audit-batch-timestamps must be > 0.")

    hashes: list[str] = []
    patch_counts: list[int] = []

    n_groups = len(starts)
    logger.info("=" * 80)
    logger.info("PHASE 8 - RADAR TARGET CONTINUITY AUDIT")
    logger.info("Zarr target patches  : %d", n_patches)
    logger.info("Unique timestamps    : %d", n_groups)
    logger.info("Hash                 : BLAKE2b-128 over stored target bytes")
    logger.info("=" * 80)

    for batch_start in range(0, n_groups, batch_timestamps):
        batch_end = min(n_groups, batch_start + batch_timestamps)
        p0 = int(starts[batch_start])
        p1 = int(ends[batch_end - 1])

        block = np.asarray(
            target[p0:p1],
            dtype=np.float32,
        )

        for gi in range(batch_start, batch_end):
            local_start = int(starts[gi] - p0)
            local_end = int(ends[gi] - p0)
            arr = np.ascontiguousarray(block[local_start:local_end])

            digest = hashlib.blake2b(
                arr.tobytes(order="C"),
                digest_size=16,
            ).hexdigest()
            hashes.append(digest)
            patch_counts.append(local_end - local_start)

        if (
            batch_end == n_groups
            or batch_end % max(batch_timestamps * 20, 1) == 0
        ):
            logger.info(
                "Target hash audit: %d/%d timestamps",
                batch_end,
                n_groups,
            )

    return pd.DataFrame({
        "timestamp_utc": unique_utc,
        "target_hash_blake2b128": hashes,
        "patch_count": np.asarray(patch_counts, dtype=np.int32),
    })


def _exact_equal(a: pd.Series, b: pd.Series) -> np.ndarray:
    av = a.to_numpy()
    bv = b.to_numpy()
    return (av == bv) | (pd.isna(av) & pd.isna(bv))


def _build_continuity_pairs(
    hashes: pd.DataFrame,
    base: pd.DataFrame,
) -> pd.DataFrame:
    metric_cols = [
        "positive_pixels",
        "max_dbz",
        "positive_pixel_fraction",
        "event_pixel_fraction_ge_30",
        "event_pixel_fraction_ge_40",
        "event_pixel_fraction_ge_45",
        "event_ge_30",
        "event_ge_40",
        "event_ge_45",
    ]

    current = hashes.merge(
        base[["timestamp_utc", *metric_cols]],
        on="timestamp_utc",
        how="inner",
        validate="one_to_one",
    )

    previous = current[
        ["timestamp_utc", "target_hash_blake2b128", *metric_cols]
    ].copy()
    previous["timestamp_utc"] = (
        previous["timestamp_utc"] + pd.Timedelta(hours=1)
    )
    previous = previous.rename(
        columns={
            "target_hash_blake2b128": "target_hash_blake2b128_prev",
            **{c: f"{c}_prev" for c in metric_cols},
        }
    )

    pairs = current.merge(
        previous,
        on="timestamp_utc",
        how="inner",
        validate="one_to_one",
    ).sort_values("timestamp_utc").reset_index(drop=True)

    pairs["previous_timestamp_utc"] = (
        pairs["timestamp_utc"] - pd.Timedelta(hours=1)
    )
    pairs["year_utc"] = pairs["timestamp_utc"].dt.year.astype(np.int16)

    pairs["exact_target_equal"] = (
        pairs["target_hash_blake2b128"]
        == pairs["target_hash_blake2b128_prev"]
    )

    equal_cols = [
        "max_dbz",
        "positive_pixel_fraction",
        "event_pixel_fraction_ge_30",
        "event_pixel_fraction_ge_40",
        "event_pixel_fraction_ge_45",
    ]
    for col in equal_cols:
        pairs[f"exact_{col}_equal"] = _exact_equal(
            pairs[col],
            pairs[f"{col}_prev"],
        )

    pairs["both_dry"] = (
        (pairs["positive_pixels"] == 0)
        & (pairs["positive_pixels_prev"] == 0)
    )
    pairs["any_wet"] = ~pairs["both_dry"]

    for event_id in ("ge_30", "ge_40", "ge_45"):
        pairs[f"either_{event_id}"] = (
            (pairs[f"event_{event_id}"] > 0)
            | (pairs[f"event_{event_id}_prev"] > 0)
        )

    return pairs


def _audit_condition_rows(
    pairs: pd.DataFrame,
    year: int | None = None,
) -> list[dict[str, Any]]:
    if year is not None:
        frame = pairs[pairs["year_utc"] == year]
    else:
        frame = pairs

    conditions = {
        "all": np.ones(len(frame), dtype=bool),
        "both_dry": frame["both_dry"].to_numpy(dtype=bool),
        "any_wet": frame["any_wet"].to_numpy(dtype=bool),
        "either_ge_30": frame["either_ge_30"].to_numpy(dtype=bool),
        "either_ge_40": frame["either_ge_40"].to_numpy(dtype=bool),
        "either_ge_45": frame["either_ge_45"].to_numpy(dtype=bool),
    }

    equality_cols = [
        "exact_target_equal",
        "exact_max_dbz_equal",
        "exact_positive_pixel_fraction_equal",
        "exact_event_pixel_fraction_ge_30_equal",
        "exact_event_pixel_fraction_ge_40_equal",
        "exact_event_pixel_fraction_ge_45_equal",
    ]

    rows: list[dict[str, Any]] = []
    for condition, mask in conditions.items():
        g = frame.loc[mask]
        row: dict[str, Any] = {
            "year_utc": "ALL" if year is None else str(year),
            "condition": condition,
            "n_pairs": int(len(g)),
        }
        for col in equality_cols:
            row[f"{col}_ratio"] = (
                float(g[col].mean()) if len(g) else np.nan
            )
        rows.append(row)
    return rows


def _identical_hash_runs(
    hashes: pd.DataFrame,
    base: pd.DataFrame,
) -> pd.DataFrame:
    frame = hashes.merge(
        base[["timestamp_utc", "positive_pixels"]],
        on="timestamp_utc",
        how="inner",
        validate="one_to_one",
    ).sort_values("timestamp_utc").reset_index(drop=True)

    if frame.empty:
        return pd.DataFrame()

    runs: list[dict[str, Any]] = []
    start_idx = 0

    def close_run(end_idx: int) -> None:
        nonlocal start_idx
        length = end_idx - start_idx + 1
        if length < 2:
            return
        start_row = frame.iloc[start_idx]
        end_row = frame.iloc[end_idx]
        runs.append({
            "start_timestamp_utc": start_row["timestamp_utc"],
            "end_timestamp_utc": end_row["timestamp_utc"],
            "length_timestamps": int(length),
            "duration_hours": int(length - 1),
            "target_hash_blake2b128": start_row[
                "target_hash_blake2b128"
            ],
            "dry_target": bool(start_row["positive_pixels"] == 0),
            "start_year_utc": int(start_row["timestamp_utc"].year),
        })

    for i in range(1, len(frame)):
        prev = frame.iloc[i - 1]
        cur = frame.iloc[i]
        contiguous = (
            cur["timestamp_utc"] - prev["timestamp_utc"]
            == pd.Timedelta(hours=1)
        )
        same_hash = (
            cur["target_hash_blake2b128"]
            == prev["target_hash_blake2b128"]
        )

        if not (contiguous and same_hash):
            close_run(i - 1)
            start_idx = i

    close_run(len(frame) - 1)
    return pd.DataFrame(runs)


def run_target_continuity_audit(
    dataset_dir: Path,
    output_dir: Path,
    base: pd.DataFrame,
    batch_timestamps: int,
    logger: logging.Logger,
) -> dict[str, Any]:
    hashes = _hash_target_groups(
        dataset_dir,
        batch_timestamps,
        logger,
    )

    pairs = _build_continuity_pairs(hashes, base)
    runs = _identical_hash_runs(hashes, base)

    overall_rows = _audit_condition_rows(pairs, year=None)
    years = sorted(int(y) for y in pairs["year_utc"].unique())
    yearly_rows: list[dict[str, Any]] = []
    for year in years:
        yearly_rows.extend(
            _audit_condition_rows(pairs, year=year)
        )

    overall = pd.DataFrame(overall_rows)
    yearly = pd.DataFrame(yearly_rows)

    hashes.to_parquet(
        output_dir / "radar_timestamp_hashes.parquet",
        index=False,
    )
    pairs.to_parquet(
        output_dir / "radar_continuity_pair_audit.parquet",
        index=False,
    )
    overall.to_parquet(
        output_dir / "radar_continuity_summary.parquet",
        index=False,
    )
    yearly.to_parquet(
        output_dir / "radar_continuity_by_year.parquet",
        index=False,
    )
    runs.to_parquet(
        output_dir / "radar_identical_target_runs.parquet",
        index=False,
    )

    def ratio(condition: str, col: str) -> float | None:
        t = overall[overall["condition"] == condition]
        if t.empty:
            return None
        value = t.iloc[0][col]
        return None if pd.isna(value) else float(value)

    wet_runs = (
        runs[~runs["dry_target"]]
        if not runs.empty
        else runs
    )
    dry_runs = (
        runs[runs["dry_target"]]
        if not runs.empty
        else runs
    )

    summary = {
        "audit_version": "target-continuity-v1",
        "dataset_dir": str(dataset_dir),
        "hash_algorithm": "BLAKE2b-128",
        "hash_domain": (
            "exact stored float32 target bytes, ordered patches per timestamp"
        ),
        "timestamps_hashed": int(len(hashes)),
        "exact_1h_pairs": int(len(pairs)),
        "patch_count_min": int(hashes["patch_count"].min()),
        "patch_count_median": float(hashes["patch_count"].median()),
        "patch_count_max": int(hashes["patch_count"].max()),
        "exact_target_equal_ratio_all": ratio(
            "all", "exact_target_equal_ratio"
        ),
        "exact_target_equal_ratio_any_wet": ratio(
            "any_wet", "exact_target_equal_ratio"
        ),
        "exact_target_equal_ratio_either_ge_30": ratio(
            "either_ge_30", "exact_target_equal_ratio"
        ),
        "exact_target_equal_ratio_either_ge_40": ratio(
            "either_ge_40", "exact_target_equal_ratio"
        ),
        "exact_target_equal_ratio_either_ge_45": ratio(
            "either_ge_45", "exact_target_equal_ratio"
        ),
        "max_identical_run_timestamps_all": (
            int(runs["length_timestamps"].max())
            if not runs.empty else 1
        ),
        "max_identical_run_timestamps_wet": (
            int(wet_runs["length_timestamps"].max())
            if not wet_runs.empty else 1
        ),
        "max_identical_run_timestamps_dry": (
            int(dry_runs["length_timestamps"].max())
            if not dry_runs.empty else 1
        ),
        "notes": [
            (
                "Exact equality among both-dry timestamps is expected and "
                "must not be interpreted as evidence of duplicated radar."
            ),
            (
                "The main audit statistic is exact target equality for "
                "any_wet and high-dBZ pair conditions."
            ),
            (
                "Equal aggregate fractions alone do not prove identical "
                "spatial fields; target hashes test exact stored arrays."
            ),
            (
                "The ordered-patch hash assumes the builder preserves a "
                "stable patch ordering within timestamps. Aggregate equality "
                "metrics are order-independent."
            ),
        ],
    }
    return summary


def main() -> None:
    args = parse_args()
    lags = parse_lags(args.lags)

    train_end = utc_timestamp(args.train_end)
    val_end = utc_timestamp(args.val_end)

    setup_output(args.output_dir, args.overwrite)
    logger = setup_logger(args.output_dir)

    base, phase6_summary = load_phase6(args.phase6_dir)
    cohort = build_common_lag_cohort(base, lags)
    cohort = assign_split(cohort, train_end, val_end)

    train = cohort[cohort["split"] == "train"].copy()

    decay_weights = exp_decay_weights(
        lags,
        args.decay_tau_hours,
    )

    logger.info("=" * 80)
    logger.info("CORRDIFF - PHASE 8 - PERSISTENCE BASELINES")
    logger.info("Source timestamps : %d", len(base))
    logger.info("Common lag cohort : %d", len(cohort))
    logger.info("Lags              : %s", lags)
    logger.info("Train end         : %s", train_end)
    logger.info("Validation end    : %s", val_end)
    logger.info("Decay weights     : %s", decay_weights.tolist())
    logger.info("=" * 80)

    # ------------------------------------------------------------------
    # Cohort summary
    # ------------------------------------------------------------------
    cohort_rows = []
    for split in ("train", "val", "test"):
        g = cohort[cohort["split"] == split]
        cohort_rows.append({
            "split": split,
            "n_timestamps": int(len(g)),
            "first_timestamp_utc": (
                str(g["timestamp_utc"].min()) if len(g) else None
            ),
            "last_timestamp_utc": (
                str(g["timestamp_utc"].max()) if len(g) else None
            ),
            "fraction_of_source_available_timestamps": (
                len(g) / len(base) if len(base) else np.nan
            ),
        })
    cohort_df = pd.DataFrame(cohort_rows)
    cohort_df.to_parquet(
        args.output_dir / "evaluation_cohort_summary.parquet",
        index=False,
    )

    # ------------------------------------------------------------------
    # Binary baselines
    # ------------------------------------------------------------------
    binary_rows: list[dict[str, Any]] = []
    binary_season_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    model_status: list[dict[str, Any]] = []

    for event_id, (_, threshold) in EVENT_SPECS.items():
        preds, status = fit_binary_models(
            train,
            cohort,
            event_id,
            lags,
            decay_weights,
        )
        model_status.append(status)

        target = f"event_{event_id}"

        for split in ("train", "val", "test"):
            mask = cohort["split"].eq(split).to_numpy()
            y = cohort.loc[mask, target].to_numpy(dtype=int)

            for baseline, all_prob in preds.items():
                prob = np.asarray(all_prob)[mask]
                row = binary_metric_row(y, prob)
                row.update({
                    "split": split,
                    "event_id": event_id,
                    "threshold_dbz": threshold,
                    "baseline": baseline,
                })
                binary_rows.append(row)

                ece, bins = expected_calibration_error(
                    y,
                    prob,
                    n_bins=10,
                )
                for b in bins:
                    b.update({
                        "split": split,
                        "event_id": event_id,
                        "threshold_dbz": threshold,
                        "baseline": baseline,
                        "ece_10bins": ece,
                    })
                    calibration_rows.append(b)

                for season in SEASON_ORDER:
                    smask = (
                        mask
                        & cohort["season_code"].eq(season).to_numpy()
                    )
                    if smask.sum() == 0:
                        continue
                    ys = cohort.loc[smask, target].to_numpy(
                        dtype=int
                    )
                    ps = np.asarray(all_prob)[smask]
                    srow = binary_metric_row(ys, ps)
                    srow.update({
                        "split": split,
                        "season_code": season,
                        "event_id": event_id,
                        "threshold_dbz": threshold,
                        "baseline": baseline,
                    })
                    binary_season_rows.append(srow)

    binary_df = pd.DataFrame(binary_rows)
    binary_season_df = pd.DataFrame(binary_season_rows)
    calibration_df = pd.DataFrame(calibration_rows)

    # Brier Skill Score against same-split climatology baseline.
    ref = binary_df[
        binary_df["baseline"] == "climatology_month_hour"
    ][["split", "event_id", "brier"]].rename(
        columns={"brier": "brier_climatology"}
    )
    binary_df = binary_df.merge(
        ref,
        on=["split", "event_id"],
        how="left",
    )
    binary_df["brier_skill_vs_climatology"] = (
        1.0 - binary_df["brier"] / binary_df["brier_climatology"]
    )

    # ------------------------------------------------------------------
    # Continuous baselines
    # ------------------------------------------------------------------
    continuous_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []

    conditions = {
        "all": None,
        "gt_0": "event_gt_0",
        "ge_30": "event_ge_30",
        "ge_40": "event_ge_40",
        "ge_45": "event_ge_45",
    }

    for metric in CONTINUOUS_METRICS:
        preds, status = fit_continuous_models(
            train,
            cohort,
            metric,
            lags,
            decay_weights,
            args.ridge_alpha,
        )
        model_status.append(status)

        for split in ("train", "val", "test"):
            split_mask = cohort["split"].eq(split).to_numpy()
            y_all = cohort[metric].to_numpy(dtype=float)

            for baseline, all_pred in preds.items():
                p_all = np.asarray(all_pred, dtype=float)

                row = continuous_metric_row(
                    y_all[split_mask],
                    p_all[split_mask],
                )
                row.update({
                    "split": split,
                    "metric": metric,
                    "baseline": baseline,
                })
                continuous_rows.append(row)

                for condition_id, condition_col in conditions.items():
                    cmask = split_mask.copy()
                    if condition_col is not None:
                        cmask &= (
                            cohort[condition_col]
                            .to_numpy(dtype=int) > 0
                        )

                    crow = continuous_metric_row(
                        y_all[cmask],
                        p_all[cmask],
                    )
                    crow.update({
                        "split": split,
                        "metric": metric,
                        "baseline": baseline,
                        "condition": condition_id,
                    })
                    condition_rows.append(crow)

    continuous_df = pd.DataFrame(continuous_rows)
    condition_df = pd.DataFrame(condition_rows)

    cref = continuous_df[
        continuous_df["baseline"] == "climatology_month_hour"
    ][["split", "metric", "mae", "rmse"]].rename(
        columns={
            "mae": "mae_climatology",
            "rmse": "rmse_climatology",
        }
    )
    continuous_df = continuous_df.merge(
        cref,
        on=["split", "metric"],
        how="left",
    )
    continuous_df["mae_skill_vs_climatology"] = (
        1.0 - continuous_df["mae"] / continuous_df["mae_climatology"]
    )
    continuous_df["rmse_skill_vs_climatology"] = (
        1.0 - continuous_df["rmse"] / continuous_df["rmse_climatology"]
    )

    # Conditional skill against conditional climatology metric.
    cond_ref = condition_df[
        condition_df["baseline"] == "climatology_month_hour"
    ][["split", "metric", "condition", "mae", "rmse"]].rename(
        columns={
            "mae": "mae_climatology",
            "rmse": "rmse_climatology",
        }
    )
    condition_df = condition_df.merge(
        cond_ref,
        on=["split", "metric", "condition"],
        how="left",
    )
    condition_df["mae_skill_vs_climatology"] = (
        1.0 - condition_df["mae"] / condition_df["mae_climatology"]
    )
    condition_df["rmse_skill_vs_climatology"] = (
        1.0 - condition_df["rmse"] / condition_df["rmse_climatology"]
    )

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    binary_df.to_parquet(
        args.output_dir / "binary_baseline_metrics.parquet",
        index=False,
    )
    binary_season_df.to_parquet(
        args.output_dir / "binary_baseline_metrics_by_season.parquet",
        index=False,
    )
    calibration_df.to_parquet(
        args.output_dir / "binary_calibration.parquet",
        index=False,
    )
    continuous_df.to_parquet(
        args.output_dir / "continuous_baseline_metrics.parquet",
        index=False,
    )
    condition_df.to_parquet(
        args.output_dir / "continuous_baseline_metrics_by_condition.parquet",
        index=False,
    )
    pd.DataFrame(model_status).to_parquet(
        args.output_dir / "model_status.parquet",
        index=False,
    )

    # ------------------------------------------------------------------
    # Direct target-continuity audit
    # ------------------------------------------------------------------
    target_audit_summary: dict[str, Any] | None = None

    if args.skip_target_audit:
        logger.info("Target continuity audit skipped by --skip-target-audit.")
    else:
        target_audit_summary = run_target_continuity_audit(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            base=base,
            batch_timestamps=args.audit_batch_timestamps,
            logger=logger,
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    warnings: list[str] = []
    source_warnings = phase6_summary.get("warnings", [])
    if source_warnings:
        warnings.append(
            "Phase 6 reported coverage warnings; Phase 8 inherits "
            "non-random availability risk."
        )

    if len(cohort) / len(base) < 0.7:
        warnings.append(
            "Common-lag cohort retains <70% of Phase 6 available timestamps. "
            "Results describe periods with all configured lags available."
        )

    if target_audit_summary is not None:
        wet_equal = target_audit_summary.get(
            "exact_target_equal_ratio_any_wet"
        )
        if wet_equal is not None and wet_equal > 0.05:
            warnings.append(
                "More than 5% of exact 1h pairs with any radar echo have "
                "byte-identical stored targets. Inspect "
                "radar_continuity_by_year.parquet and "
                "radar_identical_target_runs.parquet before interpreting "
                "near-perfect persistence as physical."
            )

    test_binary = binary_df[binary_df["split"] == "test"].copy()
    best_binary = []
    for event_id, g in test_binary.groupby("event_id"):
        g = g[np.isfinite(g["brier"])]
        if g.empty:
            continue
        row = g.loc[g["brier"].idxmin()]
        best_binary.append({
            "event_id": event_id,
            "baseline_lowest_test_brier": row["baseline"],
            "test_brier": float(row["brier"]),
            "brier_skill_vs_climatology": float(
                row["brier_skill_vs_climatology"]
            ),
            "average_precision": (
                None
                if pd.isna(row["average_precision"])
                else float(row["average_precision"])
            ),
        })

    test_cont = continuous_df[
        continuous_df["split"] == "test"
    ].copy()
    best_continuous = []
    for metric, g in test_cont.groupby("metric"):
        g = g[np.isfinite(g["rmse"])]
        if g.empty:
            continue
        row = g.loc[g["rmse"].idxmin()]
        best_continuous.append({
            "metric": metric,
            "baseline_lowest_test_rmse": row["baseline"],
            "test_rmse": float(row["rmse"]),
            "rmse_skill_vs_climatology": float(
                row["rmse_skill_vs_climatology"]
            ),
            "test_mae": float(row["mae"]),
        })

    summary = {
        "phase_version": PHASE_VERSION,
        "source_phase6_version": phase6_summary.get("phase_version"),
        "source_phase6_dir": str(args.phase6_dir),
        "radar_unit": "dBZ",
        "lags_hours": lags,
        "decay_tau_hours": args.decay_tau_hours,
        "decay_weights": {
            str(lag): float(w)
            for lag, w in zip(lags, decay_weights)
        },
        "provisional_split": {
            "train_end_utc": str(train_end),
            "validation_end_utc": str(val_end),
            "test_start_utc": str(val_end + pd.Timedelta(hours=1)),
            "note": (
                "Phase 8 split is provisional for baseline evaluation; "
                "formal split design is deferred to Phase 15."
            ),
        },
        "source_available_timestamps": int(len(base)),
        "common_lag_cohort_timestamps": int(len(cohort)),
        "common_lag_cohort_retention": float(len(cohort) / len(base)),
        "cohort_counts": {
            row["split"]: int(row["n_timestamps"])
            for row in cohort_rows
        },
        "binary_reference_baseline": "climatology_month_hour",
        "continuous_reference_baseline": "climatology_month_hour",
        "target_continuity_audit": target_audit_summary,
        "best_test_binary_by_brier": best_binary,
        "best_test_continuous_by_rmse": best_continuous,
        "warnings": warnings,
        "methodological_notes": [
            (
                "All configured lags must exist exactly in UTC for a target "
                "timestamp to enter the common evaluation cohort."
            ),
            (
                "Climatology, logistic regression and ridge regression are "
                "fit using training rows only."
            ),
            (
                "Past radar observations are allowed as predictors because "
                "they are assumed operationally known before target time t."
            ),
            (
                "Brier Skill Score and continuous skill scores use the "
                "training-only month x local-hour climatology as reference."
            ),
            (
                "Rare-event evaluation should prioritize Brier score, "
                "average precision, calibration and conditional metrics; "
                "global RMSE/MAE can be dominated by dry timestamps."
            ),
            (
                "Pixel fractions still represent the overlapping-patch "
                "training distribution rather than a de-duplicated radar field."
            ),
            (
                "When target continuity audit is enabled, exact BLAKE2b hashes "
                "of stored target arrays are compared between t and t-1h; "
                "all-dry identical targets are expected and are reported "
                "separately from wet/high-dBZ pairs."
            ),
        ],
    }

    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("=" * 80)
    logger.info("PHASE 8 COMPLETE")
    logger.info("Output: %s", args.output_dir)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
