#!/usr/bin/env python3
"""
CorrDiff - Fase 13
Regimes Meteorológicos
======================

Objetivo
--------
Identificar estados atmosféricos multivariados recorrentes e verificar como
ocorrência, intensidade e extensão radar mudam entre esses regimes.

Princípios
----------
1. Regimes são definidos SOMENTE a partir de predictors ERA5.
2. Radar é usado apenas depois do clustering, para caracterização.
3. A unidade é timestamp, usando os outputs agregados da Fase 6.
4. Dois espaços são analisados:
   - absolute: estado atmosférico absoluto;
   - anomaly_month_hour: anomalias após remoção da climatologia mês x hora local.
5. Clustering usa os 12 canais raw; os 20 predictors são usados para perfil.
6. PCA reduz redundância antes do KMeans.
7. k=5 é apenas a configuração de referência; diagnósticos k=3..8 são salvos.
8. A Fase 14 permanece responsável por distribution shift formal.

Inputs esperados
----------------
analysis_outputs/06_diurnal/
├── predictor_timestamp_means.parquet
└── timestamp_event_metrics.parquet

Outputs
-------
analysis_outputs/13_regimes/
├── analysis_summary.json
├── input_schema.parquet
├── clustering_diagnostics.parquet
├── pca_explained_variance.parquet
├── pca_loadings.parquet
├── regime_assignments.parquet
├── regime_predictor_profiles.parquet
├── regime_raw_centroids.parquet
├── regime_radar_metrics.parquet
├── regime_event_enrichment.parquet
├── regime_seasonality.parquet
├── regime_monthly_prevalence.parquet
├── regime_hourly_prevalence.parquet
├── regime_yearly_prevalence.parquet
├── regime_event_metrics_by_year.parquet
├── regime_transition_1h.parquet
├── regime_crosswalk.parquet
└── phase13.log
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import (
        adjusted_rand_score,
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise SystemExit(
        "Fase 13 requer scikit-learn: pip install scikit-learn"
    ) from exc


PHASE_VERSION = "phase13-meteorological-regimes-v1-pca-kmeans-dualspace"

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

REGIME_SPACES = ["absolute", "anomaly_month_hour"]

EVENTS = {
    "gt_0": 0.0,
    "ge_20": 20.0,
    "ge_30": 30.0,
    "ge_40": 40.0,
    "ge_45": 45.0,
}

SEASONS = ["DJF", "MAM", "JJA", "SON"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CorrDiff Fase 13: regimes meteorológicos."
    )
    p.add_argument(
        "--phase6-dir",
        type=Path,
        default=Path("analysis_outputs/06_diurnal"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs/13_regimes"),
    )
    p.add_argument("--n-regimes", type=int, default=5)
    p.add_argument("--candidate-k", default="3,4,5,6,7,8")
    p.add_argument("--pca-variance", type=float, default=0.90)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--local-timezone", default="America/Sao_Paulo")
    p.add_argument("--n-init", type=int, default=20)
    p.add_argument(
        "--silhouette-sample-size",
        type=int,
        default=10000,
    )
    p.add_argument(
        "--stability-seeds",
        default="42,43,44,45,46",
    )
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def parse_int_list(value: str) -> list[int]:
    vals = sorted({int(x.strip()) for x in value.split(",") if x.strip()})
    if not vals:
        raise ValueError("Expected a non-empty integer list.")
    return vals


def setup_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {path}; use --overwrite"
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def setup_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("corrdiff.phase13")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(path / "phase13.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def resolve_timestamp_column(df: pd.DataFrame) -> str:
    for name in ["timestamp_utc", "timestamp", "time"]:
        if name in df.columns:
            return name
    raise RuntimeError(
        "Could not find timestamp column. Expected timestamp_utc/timestamp/time."
    )


def canonicalize_predictor_columns(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    rows = []

    for canonical in ALL_PREDICTORS:
        found = None
        for alias in ALIASES[canonical]:
            if alias in out.columns:
                found = alias
                break
        if found is not None:
            if found != canonical:
                out[canonical] = out[found]
            rows.append({
                "canonical_name": canonical,
                "source_column": found,
                "status": "found",
            })
        else:
            rows.append({
                "canonical_name": canonical,
                "source_column": None,
                "status": "missing",
            })

    schema = pd.DataFrame(rows)
    missing_raw = schema[
        (schema["canonical_name"].isin(RAW_CANONICAL))
        & (schema["status"] == "missing")
    ]["canonical_name"].tolist()

    if missing_raw:
        raise RuntimeError(
            f"Missing required raw predictor columns: {missing_raw}"
        )

    return out, schema


def ensure_time_columns(
    df: pd.DataFrame,
    timestamp_col: str,
    local_timezone: str,
) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out[timestamp_col], utc=True)

    out["timestamp_utc"] = ts

    if "year_utc" not in out:
        out["year_utc"] = ts.dt.year.astype(np.int16)
    local = ts.dt.tz_convert(local_timezone)
    if "month_local" not in out:
        out["month_local"] = local.dt.month.astype(np.int8)
    if "hour_local" not in out:
        out["hour_local"] = local.dt.hour.astype(np.int8)
    if "season_code" not in out:
        out["season_code"] = [
            season_from_month(int(x))
            for x in out["month_local"].to_numpy()
        ]

    out["season_code"] = pd.Categorical(
        out["season_code"],
        categories=SEASONS,
        ordered=True,
    )
    return out


def prepare_radar_metrics(
    radar: pd.DataFrame,
    local_timezone: str,
) -> pd.DataFrame:
    out = radar.copy()
    ts_col = resolve_timestamp_column(out)
    out = ensure_time_columns(out, ts_col, local_timezone)

    if "max_dbz" not in out.columns:
        raise RuntimeError(
            "Phase 13 requires max_dbz in Phase 6 timestamp_event_metrics."
        )

    max_dbz = out["max_dbz"].to_numpy(dtype=float)
    for event_id, threshold in EVENTS.items():
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


def build_anomaly_space(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    groups = df.groupby(
        ["month_local", "hour_local"],
        observed=True,
    )

    for col in columns:
        clim = groups[col].transform("mean")
        out[col] = df[col] - clim

    return out


def fit_pca_space(
    x: np.ndarray,
    variance_threshold: float,
    seed: int,
) -> tuple[StandardScaler, PCA, np.ndarray]:
    scaler = StandardScaler()
    z = scaler.fit_transform(x)

    pca_full = PCA(random_state=seed)
    pca_full.fit(z)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(
        np.searchsorted(cumulative, variance_threshold) + 1
    )
    n_components = max(2, min(n_components, z.shape[1]))

    pca = PCA(
        n_components=n_components,
        random_state=seed,
    )
    scores = pca.fit_transform(z)
    return scaler, pca, scores


def kmeans_diagnostics(
    scores: np.ndarray,
    candidate_k: list[int],
    reference_k: int,
    seed: int,
    n_init: int,
    silhouette_sample_size: int,
    stability_seeds: list[int],
    regime_space: str,
) -> tuple[pd.DataFrame, KMeans, np.ndarray]:
    rows = []
    reference_model = None
    reference_labels = None

    for k in candidate_k:
        if k < 2 or k >= len(scores):
            continue

        model = KMeans(
            n_clusters=k,
            n_init=n_init,
            random_state=seed,
        )
        labels = model.fit_predict(scores)

        sample_size = min(
            silhouette_sample_size,
            len(scores),
        )
        sil = float(
            silhouette_score(
                scores,
                labels,
                sample_size=sample_size,
                random_state=seed,
            )
        )
        ch = float(calinski_harabasz_score(scores, labels))
        db = float(davies_bouldin_score(scores, labels))

        stability_mean = np.nan
        stability_min = np.nan
        if k == reference_k:
            label_runs = []
            model_runs = []
            for s in stability_seeds:
                km = KMeans(
                    n_clusters=k,
                    n_init=n_init,
                    random_state=s,
                )
                lab = km.fit_predict(scores)
                label_runs.append(lab)
                model_runs.append(km)

            aris = [
                adjusted_rand_score(a, b)
                for a, b in combinations(label_runs, 2)
            ]
            if aris:
                stability_mean = float(np.mean(aris))
                stability_min = float(np.min(aris))

            best_idx = int(
                np.argmin([m.inertia_ for m in model_runs])
            )
            reference_model = model_runs[best_idx]
            reference_labels = label_runs[best_idx]

        rows.append({
            "regime_space": regime_space,
            "k": int(k),
            "inertia": float(model.inertia_),
            "silhouette": sil,
            "calinski_harabasz": ch,
            "davies_bouldin": db,
            "reference_k": bool(k == reference_k),
            "reference_k_stability_ari_mean": stability_mean,
            "reference_k_stability_ari_min": stability_min,
        })

    if reference_model is None or reference_labels is None:
        raise RuntimeError(
            f"Reference k={reference_k} not evaluated."
        )

    return pd.DataFrame(rows), reference_model, reference_labels


def order_regime_labels(
    labels: np.ndarray,
    frame: pd.DataFrame,
) -> tuple[np.ndarray, dict[int, int]]:
    """
    Deterministic label ordering:
    sort regimes by tcwv mean, then t2m mean.
    This gives stable display ids R1..Rk without claiming physical rank.
    """
    rows = []
    for old in np.unique(labels):
        m = labels == old
        rows.append({
            "old": int(old),
            "tcwv": float(frame.loc[m, "tcwv"].mean()),
            "t2m": float(frame.loc[m, "t2m"].mean()),
        })

    order = pd.DataFrame(rows).sort_values(
        ["tcwv", "t2m", "old"]
    )["old"].tolist()

    mapping = {
        int(old): int(new + 1)
        for new, old in enumerate(order)
    }
    new_labels = np.array(
        [mapping[int(x)] for x in labels],
        dtype=np.int16,
    )
    return new_labels, mapping


def pca_outputs(
    pca: PCA,
    raw_features: list[str],
    regime_space: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ev_rows = []
    cumulative = 0.0
    for i, ratio in enumerate(
        pca.explained_variance_ratio_,
        start=1,
    ):
        cumulative += float(ratio)
        ev_rows.append({
            "regime_space": regime_space,
            "component": f"PC{i}",
            "component_index": i,
            "explained_variance_ratio": float(ratio),
            "cumulative_explained_variance_ratio": cumulative,
        })

    load_rows = []
    for i in range(pca.components_.shape[0]):
        for j, feature in enumerate(raw_features):
            load_rows.append({
                "regime_space": regime_space,
                "component": f"PC{i+1}",
                "component_index": i + 1,
                "feature": feature,
                "loading": float(pca.components_[i, j]),
            })

    return pd.DataFrame(ev_rows), pd.DataFrame(load_rows)


def regime_predictor_profiles(
    df: pd.DataFrame,
    labels: np.ndarray,
    predictors: list[str],
    regime_space: str,
) -> pd.DataFrame:
    rows = []

    for predictor in predictors:
        if predictor not in df.columns:
            continue

        vals = df[predictor].to_numpy(dtype=float)
        global_mean = float(np.nanmean(vals))
        global_std = float(np.nanstd(vals, ddof=0))
        if not np.isfinite(global_std) or global_std < 1e-12:
            global_std = np.nan

        for regime_id in sorted(np.unique(labels)):
            m = labels == regime_id
            rvals = vals[m]

            mean = float(np.nanmean(rvals))
            median = float(np.nanmedian(rvals))
            std = float(np.nanstd(rvals, ddof=0))
            anomaly_sigma = (
                (mean - global_mean) / global_std
                if np.isfinite(global_std)
                else np.nan
            )

            rows.append({
                "regime_space": regime_space,
                "regime_id": int(regime_id),
                "regime": f"R{int(regime_id)}",
                "predictor": predictor,
                "n_timestamps": int(m.sum()),
                "regime_mean": mean,
                "regime_median": median,
                "regime_std": std,
                "global_mean": global_mean,
                "global_std": global_std,
                "regime_mean_anomaly_sigma": float(anomaly_sigma),
            })

    return pd.DataFrame(rows)


def raw_centroids(
    df: pd.DataFrame,
    labels: np.ndarray,
    raw_features: list[str],
    regime_space: str,
) -> pd.DataFrame:
    rows = []
    for regime_id in sorted(np.unique(labels)):
        m = labels == regime_id
        for feature in raw_features:
            rows.append({
                "regime_space": regime_space,
                "regime_id": int(regime_id),
                "regime": f"R{int(regime_id)}",
                "predictor": feature,
                "mean_value": float(
                    df.loc[m, feature].mean()
                ),
                "std_value": float(
                    df.loc[m, feature].std(ddof=0)
                ),
            })
    return pd.DataFrame(rows)


def regime_radar_metrics(
    df: pd.DataFrame,
    labels: np.ndarray,
    regime_space: str,
) -> pd.DataFrame:
    rows = []
    total = len(df)

    radar_continuous = [
        c for c in [
            "max_dbz",
            "positive_pixel_fraction",
            "event_pixel_fraction_ge_20",
            "event_pixel_fraction_ge_30",
            "event_pixel_fraction_ge_40",
            "event_pixel_fraction_ge_45",
            "mean_positive_dbz",
        ]
        if c in df.columns
    ]

    for regime_id in sorted(np.unique(labels)):
        m = labels == regime_id
        row: dict[str, Any] = {
            "regime_space": regime_space,
            "regime_id": int(regime_id),
            "regime": f"R{int(regime_id)}",
            "n_timestamps": int(m.sum()),
            "prevalence": float(m.sum() / total),
        }

        for event_id in EVENTS:
            col = f"event_{event_id}"
            row[f"timestamp_event_rate_{event_id}"] = float(
                df.loc[m, col].mean()
            )

        for col in radar_continuous:
            row[f"mean_{col}"] = float(
                df.loc[m, col].mean()
            )

        rows.append(row)

    return pd.DataFrame(rows)


def event_enrichment(
    df: pd.DataFrame,
    labels: np.ndarray,
    regime_space: str,
) -> pd.DataFrame:
    rows = []
    for event_id in EVENTS:
        col = f"event_{event_id}"
        global_rate = float(df[col].mean())

        for regime_id in sorted(np.unique(labels)):
            m = labels == regime_id
            rate = float(df.loc[m, col].mean())
            rows.append({
                "regime_space": regime_space,
                "regime_id": int(regime_id),
                "regime": f"R{int(regime_id)}",
                "event_id": event_id,
                "n_timestamps": int(m.sum()),
                "global_event_rate": global_rate,
                "regime_event_rate": rate,
                "risk_ratio_vs_global": (
                    rate / global_rate
                    if global_rate > 0
                    else np.nan
                ),
                "absolute_rate_difference": rate - global_rate,
            })
    return pd.DataFrame(rows)


def prevalence_by_group(
    df: pd.DataFrame,
    labels: np.ndarray,
    regime_space: str,
    group_col: str,
) -> pd.DataFrame:
    tmp = pd.DataFrame({
        group_col: df[group_col].to_numpy(),
        "regime_id": labels,
    })
    rows = []

    grouped = tmp.groupby(group_col, observed=True)
    for group_value, g in grouped:
        n_group = len(g)
        for regime_id in sorted(np.unique(labels)):
            n = int((g["regime_id"] == regime_id).sum())
            rows.append({
                "regime_space": regime_space,
                group_col: group_value,
                "regime_id": int(regime_id),
                "regime": f"R{int(regime_id)}",
                "n_timestamps": n,
                "group_total_timestamps": int(n_group),
                "prevalence_within_group": float(n / n_group),
            })

    return pd.DataFrame(rows)


def seasonality_table(
    df: pd.DataFrame,
    labels: np.ndarray,
    regime_space: str,
) -> pd.DataFrame:
    tmp = pd.DataFrame({
        "season_code": df["season_code"].astype(str).to_numpy(),
        "regime_id": labels,
    })
    rows = []
    total_by_season = tmp.groupby(
        "season_code",
        observed=True,
    ).size()
    total_by_regime = tmp.groupby(
        "regime_id",
        observed=True,
    ).size()

    for season in SEASONS:
        for regime_id in sorted(np.unique(labels)):
            m = (
                (tmp["season_code"] == season)
                & (tmp["regime_id"] == regime_id)
            )
            n = int(m.sum())
            rows.append({
                "regime_space": regime_space,
                "season_code": season,
                "regime_id": int(regime_id),
                "regime": f"R{int(regime_id)}",
                "n_timestamps": n,
                "prevalence_regime_within_season": (
                    n / int(total_by_season.get(season, 0))
                    if int(total_by_season.get(season, 0)) > 0
                    else np.nan
                ),
                "season_fraction_within_regime": (
                    n / int(total_by_regime.get(regime_id, 0))
                    if int(total_by_regime.get(regime_id, 0)) > 0
                    else np.nan
                ),
            })
    return pd.DataFrame(rows)


def event_metrics_by_year(
    df: pd.DataFrame,
    labels: np.ndarray,
    regime_space: str,
) -> pd.DataFrame:
    tmp = df[["year_utc"]].copy()
    tmp["regime_id"] = labels
    for event_id in EVENTS:
        tmp[f"event_{event_id}"] = df[f"event_{event_id}"].to_numpy()

    rows = []
    for (year, regime_id), g in tmp.groupby(
        ["year_utc", "regime_id"],
        observed=True,
    ):
        row = {
            "regime_space": regime_space,
            "year_utc": int(year),
            "regime_id": int(regime_id),
            "regime": f"R{int(regime_id)}",
            "n_timestamps": int(len(g)),
        }
        for event_id in EVENTS:
            row[f"timestamp_event_rate_{event_id}"] = float(
                g[f"event_{event_id}"].mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def transition_matrix_1h(
    df: pd.DataFrame,
    labels: np.ndarray,
    regime_space: str,
) -> pd.DataFrame:
    tmp = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(
            df["timestamp_utc"],
            utc=True,
        ),
        "regime_from": labels,
    }).sort_values("timestamp_utc")

    nxt = tmp.rename(
        columns={
            "timestamp_utc": "timestamp_next",
            "regime_from": "regime_to",
        }
    ).copy()
    nxt["timestamp_utc"] = (
        nxt["timestamp_next"] - pd.Timedelta(hours=1)
    )

    pairs = tmp.merge(
        nxt[["timestamp_utc", "regime_to"]],
        on="timestamp_utc",
        how="inner",
    )

    counts = (
        pairs.groupby(
            ["regime_from", "regime_to"],
            observed=True,
        )
        .size()
        .rename("n_pairs")
        .reset_index()
    )
    totals = (
        counts.groupby("regime_from", observed=True)["n_pairs"]
        .sum()
        .rename("n_from_pairs")
        .reset_index()
    )
    counts = counts.merge(totals, on="regime_from", how="left")
    counts["transition_probability"] = (
        counts["n_pairs"] / counts["n_from_pairs"]
    )
    counts["regime_space"] = regime_space
    counts["from_regime"] = counts["regime_from"].map(
        lambda x: f"R{int(x)}"
    )
    counts["to_regime"] = counts["regime_to"].map(
        lambda x: f"R{int(x)}"
    )
    return counts[
        [
            "regime_space",
            "regime_from",
            "from_regime",
            "regime_to",
            "to_regime",
            "n_pairs",
            "n_from_pairs",
            "transition_probability",
        ]
    ]


def crosswalk(
    absolute_labels: np.ndarray,
    anomaly_labels: np.ndarray,
) -> pd.DataFrame:
    tmp = pd.DataFrame({
        "absolute_regime_id": absolute_labels,
        "anomaly_regime_id": anomaly_labels,
    })
    counts = (
        tmp.groupby(
            ["absolute_regime_id", "anomaly_regime_id"],
            observed=True,
        )
        .size()
        .rename("n_timestamps")
        .reset_index()
    )
    abs_tot = (
        counts.groupby(
            "absolute_regime_id",
            observed=True,
        )["n_timestamps"]
        .sum()
        .rename("absolute_regime_total")
        .reset_index()
    )
    ano_tot = (
        counts.groupby(
            "anomaly_regime_id",
            observed=True,
        )["n_timestamps"]
        .sum()
        .rename("anomaly_regime_total")
        .reset_index()
    )
    counts = counts.merge(abs_tot, on="absolute_regime_id")
    counts = counts.merge(ano_tot, on="anomaly_regime_id")
    counts["fraction_within_absolute_regime"] = (
        counts["n_timestamps"] / counts["absolute_regime_total"]
    )
    counts["fraction_within_anomaly_regime"] = (
        counts["n_timestamps"] / counts["anomaly_regime_total"]
    )
    return counts


def main() -> None:
    args = parse_args()
    candidate_k = parse_int_list(args.candidate_k)
    stability_seeds = parse_int_list(args.stability_seeds)

    if args.n_regimes not in candidate_k:
        candidate_k = sorted(candidate_k + [args.n_regimes])

    setup_output(args.output_dir, args.overwrite)
    logger = setup_logger(args.output_dir)

    pred_path = args.phase6_dir / "predictor_timestamp_means.parquet"
    radar_path = args.phase6_dir / "timestamp_event_metrics.parquet"

    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    if not radar_path.exists():
        raise FileNotFoundError(radar_path)

    predictors = pd.read_parquet(pred_path)
    radar = pd.read_parquet(radar_path)

    pred_ts = resolve_timestamp_column(predictors)
    predictors = ensure_time_columns(
        predictors,
        pred_ts,
        args.local_timezone,
    )
    predictors, schema = canonicalize_predictor_columns(predictors)

    radar = prepare_radar_metrics(
        radar,
        args.local_timezone,
    )

    schema.to_parquet(
        args.output_dir / "input_schema.parquet",
        index=False,
    )

    keep_pred = [
        "timestamp_utc",
        "year_utc",
        "month_local",
        "hour_local",
        "season_code",
    ] + [
        p for p in ALL_PREDICTORS
        if p in predictors.columns
    ]

    radar_cols = [
        "timestamp_utc",
        "max_dbz",
    ] + [
        f"event_{e}" for e in EVENTS
    ]
    radar_cols += [
        c for c in [
            "positive_pixel_fraction",
            "event_pixel_fraction_ge_20",
            "event_pixel_fraction_ge_30",
            "event_pixel_fraction_ge_40",
            "event_pixel_fraction_ge_45",
            "mean_positive_dbz",
        ]
        if c in radar.columns
    ]

    df = predictors[keep_pred].merge(
        radar[radar_cols],
        on="timestamp_utc",
        how="inner",
        validate="one_to_one",
    )

    if df.empty:
        raise RuntimeError("No aligned Phase 6 timestamps after merge.")

    raw_matrix = df[RAW_CANONICAL].to_numpy(dtype=float)
    finite = np.isfinite(raw_matrix).all(axis=1)
    if not finite.all():
        logger.warning(
            "Dropping %d timestamps with non-finite raw predictors.",
            int((~finite).sum()),
        )
        df = df.loc[finite].reset_index(drop=True)

    logger.info("=" * 88)
    logger.info("CORRDIFF - PHASE 13 - METEOROLOGICAL REGIMES")
    logger.info("Phase 6 directory : %s", args.phase6_dir)
    logger.info("Aligned timestamps: %d", len(df))
    logger.info("Reference k       : %d", args.n_regimes)
    logger.info("Candidate k       : %s", candidate_k)
    logger.info("PCA variance      : %.3f", args.pca_variance)
    logger.info("=" * 88)

    anomaly_raw = build_anomaly_space(df, RAW_CANONICAL)

    diagnostics_parts = []
    pca_ev_parts = []
    pca_load_parts = []
    assignment_parts = []
    profile_parts = []
    centroid_parts = []
    radar_parts = []
    enrichment_parts = []
    season_parts = []
    month_parts = []
    hour_parts = []
    year_parts = []
    event_year_parts = []
    transition_parts = []

    labels_by_space: dict[str, np.ndarray] = {}
    pca_components_by_space: dict[str, int] = {}

    for regime_space in REGIME_SPACES:
        logger.info("-" * 88)
        logger.info("Regime space: %s", regime_space)

        if regime_space == "absolute":
            x_frame = df[RAW_CANONICAL]
        else:
            x_frame = anomaly_raw[RAW_CANONICAL]

        x = x_frame.to_numpy(dtype=float)
        scaler, pca, scores = fit_pca_space(
            x,
            args.pca_variance,
            args.seed,
        )
        pca_components_by_space[regime_space] = int(
            pca.n_components_
        )

        diag, model, labels0 = kmeans_diagnostics(
            scores=scores,
            candidate_k=candidate_k,
            reference_k=args.n_regimes,
            seed=args.seed,
            n_init=args.n_init,
            silhouette_sample_size=args.silhouette_sample_size,
            stability_seeds=stability_seeds,
            regime_space=regime_space,
        )

        labels, mapping = order_regime_labels(
            labels0,
            df,
        )
        labels_by_space[regime_space] = labels

        diagnostics_parts.append(diag)

        ev, load = pca_outputs(
            pca,
            RAW_CANONICAL,
            regime_space,
        )
        pca_ev_parts.append(ev)
        pca_load_parts.append(load)

        assignment_parts.append(pd.DataFrame({
            "timestamp_utc": df["timestamp_utc"].to_numpy(),
            "year_utc": df["year_utc"].to_numpy(),
            "month_local": df["month_local"].to_numpy(),
            "hour_local": df["hour_local"].to_numpy(),
            "season_code": df["season_code"].astype(str).to_numpy(),
            "regime_space": regime_space,
            "regime_id": labels,
            "regime": [f"R{x}" for x in labels],
        }))

        available_profile_predictors = [
            p for p in ALL_PREDICTORS
            if p in df.columns
        ]
        profile_parts.append(
            regime_predictor_profiles(
                df,
                labels,
                available_profile_predictors,
                regime_space,
            )
        )
        centroid_parts.append(
            raw_centroids(
                df,
                labels,
                RAW_CANONICAL,
                regime_space,
            )
        )
        radar_parts.append(
            regime_radar_metrics(
                df,
                labels,
                regime_space,
            )
        )
        enrichment_parts.append(
            event_enrichment(
                df,
                labels,
                regime_space,
            )
        )
        season_parts.append(
            seasonality_table(
                df,
                labels,
                regime_space,
            )
        )
        month_parts.append(
            prevalence_by_group(
                df,
                labels,
                regime_space,
                "month_local",
            )
        )
        hour_parts.append(
            prevalence_by_group(
                df,
                labels,
                regime_space,
                "hour_local",
            )
        )
        year_parts.append(
            prevalence_by_group(
                df,
                labels,
                regime_space,
                "year_utc",
            )
        )
        event_year_parts.append(
            event_metrics_by_year(
                df,
                labels,
                regime_space,
            )
        )
        transition_parts.append(
            transition_matrix_1h(
                df,
                labels,
                regime_space,
            )
        )

        logger.info(
            "%s | PCA components=%d | regime counts=%s",
            regime_space,
            pca.n_components_,
            dict(
                pd.Series(labels)
                .value_counts()
                .sort_index()
            ),
        )
        logger.info(
            "%s | label mapping old->display: %s",
            regime_space,
            mapping,
        )

    pd.concat(
        diagnostics_parts,
        ignore_index=True,
    ).to_parquet(
        args.output_dir / "clustering_diagnostics.parquet",
        index=False,
    )
    pd.concat(
        pca_ev_parts,
        ignore_index=True,
    ).to_parquet(
        args.output_dir / "pca_explained_variance.parquet",
        index=False,
    )
    pd.concat(
        pca_load_parts,
        ignore_index=True,
    ).to_parquet(
        args.output_dir / "pca_loadings.parquet",
        index=False,
    )
    pd.concat(
        assignment_parts,
        ignore_index=True,
    ).to_parquet(
        args.output_dir / "regime_assignments.parquet",
        index=False,
    )
    pd.concat(
        profile_parts,
        ignore_index=True,
    ).to_parquet(
        args.output_dir / "regime_predictor_profiles.parquet",
        index=False,
    )
    pd.concat(
        centroid_parts,
        ignore_index=True,
    ).to_parquet(
        args.output_dir / "regime_raw_centroids.parquet",
        index=False,
    )
    pd.concat(
        radar_parts,
        ignore_index=True,
    ).to_parquet(
        args.output_dir / "regime_radar_metrics.parquet",
        index=False,
    )
    pd.concat(
        enrichment_parts,
        ignore_index=True,
    ).to_parquet(
        args.output_dir / "regime_event_enrichment.parquet",
        index=False,
    )
    pd.concat(
        season_parts,
        ignore_index=True,
    ).to_parquet(
        args.output_dir / "regime_seasonality.parquet",
        index=False,
    )
    pd.concat(
        month_parts,
        ignore_index=True,
    ).to_parquet(
        args.output_dir / "regime_monthly_prevalence.parquet",
        index=False,
    )
    pd.concat(
        hour_parts,
        ignore_index=True,
    ).to_parquet(
        args.output_dir / "regime_hourly_prevalence.parquet",
        index=False,
    )
    pd.concat(
        year_parts,
        ignore_index=True,
    ).to_parquet(
        args.output_dir / "regime_yearly_prevalence.parquet",
        index=False,
    )
    pd.concat(
        event_year_parts,
        ignore_index=True,
    ).to_parquet(
        args.output_dir / "regime_event_metrics_by_year.parquet",
        index=False,
    )
    pd.concat(
        transition_parts,
        ignore_index=True,
    ).to_parquet(
        args.output_dir / "regime_transition_1h.parquet",
        index=False,
    )

    cw = crosswalk(
        labels_by_space["absolute"],
        labels_by_space["anomaly_month_hour"],
    )
    cw.to_parquet(
        args.output_dir / "regime_crosswalk.parquet",
        index=False,
    )

    warnings = [
        (
            "Regime occupancy is conditioned on Phase 6 timestamps with radar "
            "availability. Missingness may not be random with respect to weather."
        ),
        (
            "The reference number of regimes is fixed by configuration; candidate "
            "k diagnostics are sensitivity information, not an automatic proof of "
            "a uniquely correct number of meteorological regimes."
        ),
        (
            "Absolute-space regimes can reflect seasonality and the diurnal cycle. "
            "The anomaly_month_hour space is provided to isolate departures from "
            "month x local-hour climatology."
        ),
        (
            "Regimes are defined from ERA5 predictors only. Radar metrics are "
            "joined after clustering to avoid direct target leakage."
        ),
        (
            "The 12 raw channels define clustering; derived variables characterize "
            "regimes but do not influence cluster geometry, reducing redundancy."
        ),
        (
            "KMeans imposes approximately convex clusters in PCA space. Meteorological "
            "states need not be truly discrete or spherical."
        ),
        (
            "One-hour transitions use only exact consecutive UTC timestamp pairs; "
            "gaps are not bridged."
        ),
        (
            "Regime labels R1..Rk are display identifiers ordered by mean tcwv then "
            "t2m. They are not ordinal physical rankings."
        ),
        (
            "Phase 13 is descriptive. Phase 14 remains responsible for formal "
            "distribution-shift analysis, especially the 2023-2024 degradation."
        ),
    ]

    summary = {
        "phase_version": PHASE_VERSION,
        "phase6_dir": str(args.phase6_dir),
        "n_aligned_timestamps": int(len(df)),
        "reference_n_regimes": int(args.n_regimes),
        "candidate_k": candidate_k,
        "pca_variance_threshold": float(args.pca_variance),
        "pca_components_by_space": pca_components_by_space,
        "regime_spaces": REGIME_SPACES,
        "clustering_features": RAW_CANONICAL,
        "profile_predictors": [
            p for p in ALL_PREDICTORS
            if p in df.columns
        ],
        "events": EVENTS,
        "scope": (
            "timestamp-level unsupervised ERA5 regime discovery with radar "
            "characterization after clustering"
        ),
        "warnings": warnings,
        "methodological_notes": [
            (
                "absolute space answers which broad atmospheric states recur "
                "over the historical record."
            ),
            (
                "anomaly_month_hour space answers which synoptic/anomalous states "
                "recur after removing the mean seasonal-diurnal background."
            ),
            (
                "PCA is fit separately in each regime space on standardized raw "
                "predictors and retains the configured cumulative variance."
            ),
            (
                "Cluster stability is summarized with Adjusted Rand Index across "
                "multiple KMeans seeds for the reference k."
            ),
            (
                "Event enrichment is expressed relative to the global event rate "
                "among the same radar-available timestamps."
            ),
            (
                "Yearly regime prevalence and regime-specific event rates are "
                "screening outputs for Phase 14, not a formal shift test."
            ),
            (
                "All associations between regime and radar are descriptive and "
                "do not establish causality."
            ),
        ],
    }

    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("=" * 88)
    logger.info("PHASE 13 COMPLETE")
    logger.info("Output: %s", args.output_dir)
    logger.info("=" * 88)


if __name__ == "__main__":
    main()
