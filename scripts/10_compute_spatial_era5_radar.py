#!/usr/bin/env python3
"""
CorrDiff - Fase 10
Relações espaciais ERA5 × Radar
================================

Objetivo
--------
Quantificar até que ponto os campos ERA5 apresentados ao CorrDiff se relacionam
espacialmente com o target radar, separando:

1. associação pixel a pixel bruta;
2. associação espacial após centralização dentro de cada patch;
3. associação após suavização local do preditor em janelas crescentes;
4. contraste do preditor dentro vs fora das regiões radar;
5. associação entre preditor em x e radar deslocado em x + delta;
6. associação em nível de patch.

Escopo
------
A análise opera na distribuição de patches 32x32 sobrepostos do train.zarr.
Não reconstrói o campo completo 70x84 e não interpreta a grade ERA5 interpolada
como informação meteorológica independente em escala sub-grid.

A Fase 10 é descritiva. Correlação espacial não implica causalidade.

Saídas
------
analysis_outputs/10_spatial_era5_radar/
├── analysis_summary.json
├── patch_strata_counts.parquet
├── sampled_patch_manifest.parquet
├── input_channel_metadata.parquet
├── pixel_spatial_associations.parquet
├── pixel_spatial_associations_by_season.parquet
├── spatial_cross_offset_associations.parquet
├── within_patch_event_contrasts.parquet
├── within_patch_event_contrasts_by_season.parquet
├── patch_level_associations.parquet
└── phase10.log
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
    raise SystemExit(
        "Fase 10 requer zarr. Instale no ambiente de análise com: pip install zarr"
    ) from exc

try:
    from scipy import ndimage
except ImportError as exc:
    raise SystemExit(
        "Fase 10 requer scipy. Instale com: pip install scipy"
    ) from exc


PHASE_VERSION = "phase10-spatial-era5-radar-v1-aligned-centered-offset"

CANONICAL_RAW_CHANNELS = [
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

STRATUM_NAMES = {
    0: "dry",
    1: "gt0_lt20",
    2: "ge20_lt30",
    3: "ge30_lt40",
    4: "ge40_lt45",
    5: "ge45",
}

SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]

RADAR_FIELDS = [
    "target_log1p",
    "gt_0",
    "ge_20",
    "ge_30",
    "ge_40",
    "ge_45",
]

PATCH_RADAR_METRICS = [
    "max_dbz",
    "positive_pixel_fraction",
    "event_pixel_fraction_ge_30",
    "event_pixel_fraction_ge_40",
    "event_pixel_fraction_ge_45",
]

# dy, dx are radar displacements relative to predictor position.
OFFSET_DIRECTIONS = {
    "CENTER": (0, 0),
    "E": (0, 1),
    "W": (0, -1),
    "S": (1, 0),
    "N": (-1, 0),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CorrDiff Fase 10: relações espaciais ERA5 × Radar."
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("datasets/corrdiff_2011_2024"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs/10_spatial_era5_radar"),
    )
    p.add_argument(
        "--sample-per-stratum",
        type=int,
        default=3000,
        help=(
            "Máximo de patches por estrato de intensidade. Estratos menores "
            "são incluídos integralmente."
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--smoothing-windows",
        default="1,3,5,9,17",
        help=(
            "Janelas quadradas ímpares em pixels para médias locais dos "
            "preditores. 1 = sem suavização."
        ),
    )
    p.add_argument(
        "--offset-steps",
        default="2,4,8",
        help=(
            "Deslocamentos em pixels para associação predictor(x) × "
            "radar(x+delta)."
        ),
    )
    p.add_argument(
        "--radar-resolution-km",
        type=float,
        default=2.0,
    )
    p.add_argument(
        "--local-timezone",
        default="America/Sao_Paulo",
    )
    p.add_argument(
        "--channel-names",
        default=None,
        help=(
            "Lista separada por vírgulas para os canais raw de input. "
            "Se omitida e input tiver 12 canais, usa a ordem canônica "
            "tcwv,t2m,u10,v10,t_850,r_850,u_850,v_850,t_500,r_500,u_500,v_500."
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
    logger = logging.getLogger("corrdiff.phase10")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(path / "phase10.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def parse_positive_ints(text: str, *, odd: bool = False) -> list[int]:
    vals = sorted(
        set(int(x.strip()) for x in text.split(",") if x.strip())
    )
    if not vals or any(v <= 0 for v in vals):
        raise ValueError("Expected positive integers.")
    if odd and any(v % 2 == 0 for v in vals):
        raise ValueError("Smoothing windows must be odd integers.")
    return vals


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
        pd.to_datetime(
            arr,
            unit=infer_datetime_unit(arr),
            utc=True,
        )
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

    # Try common Zarr attrs before falling back.
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
            "Phase 10 derived predictors require the canonical raw fields. "
            f"Missing: {missing}"
        )


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

    chunk0 = (
        int(target.chunks[0])
        if getattr(target, "chunks", None)
        else 256
    )

    t20 = STORED_THRESHOLDS["ge_20"]
    t30 = STORED_THRESHOLDS["ge_30"]
    t40 = STORED_THRESHOLDS["ge_40"]
    t45 = STORED_THRESHOLDS["ge_45"]

    logger.info("=" * 80)
    logger.info("PHASE 10 - PASS A - TARGET STRATA")
    logger.info("Target patches : %d", n)
    logger.info("Target shape   : %s", tuple(target.shape))
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
    if sample_per_stratum <= 0:
        raise ValueError("--sample-per-stratum must be > 0.")

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

    patch_indices = manifest["patch_index"].to_numpy(dtype=np.int64)
    tm = time_meta.iloc[patch_indices].reset_index(drop=True)
    return pd.concat([manifest.reset_index(drop=True), tm], axis=1)


def derive_predictors(
    raw: np.ndarray,
    raw_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """
    raw shape = [batch, raw_channel, y, x]
    """
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

    all_fields = np.concatenate(
        [
            raw.astype(np.float32, copy=False),
            np.stack(derived, axis=1).astype(np.float32, copy=False),
        ],
        axis=1,
    )
    return all_fields, raw_names + DERIVED_CHANNELS


def radar_fields_from_target(
    target_block: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """
    target_block shape [batch, 1, y, x] or [batch,y,x]
    output [batch, radar_field, y, x]
    """
    t = np.asarray(target_block, dtype=np.float32)
    if t.ndim == 4 and t.shape[1] == 1:
        t = t[:, 0]
    if t.ndim != 3:
        raise RuntimeError(
            f"Unexpected target block shape: {target_block.shape}"
        )

    fields = [t]
    for event_id in EVENT_THRESHOLDS:
        fields.append(event_mask(t, event_id).astype(np.float32))

    return np.stack(fields, axis=1), list(RADAR_FIELDS)


def patch_radar_metrics(
    target_block: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    t = np.asarray(target_block, dtype=np.float32)
    if t.ndim == 4 and t.shape[1] == 1:
        t = t[:, 0]

    flat = t.reshape(t.shape[0], -1)
    max_dbz = np.expm1(
        flat.max(axis=1).astype(np.float64)
    )

    pos = (flat > 0).mean(axis=1)
    ge30 = (flat >= STORED_THRESHOLDS["ge_30"]).mean(axis=1)
    ge40 = (flat >= STORED_THRESHOLDS["ge_40"]).mean(axis=1)
    ge45 = (flat >= STORED_THRESHOLDS["ge_45"]).mean(axis=1)

    y = np.column_stack([max_dbz, pos, ge30, ge40, ge45])
    return y.astype(np.float64), list(PATCH_RADAR_METRICS)


class CorrMatrixAccumulator:
    """
    Weighted streaming correlation matrix between columns of X and Y.

    update_matrix:
        X shape [n_obs, p]
        Y shape [n_obs, q]
        weights shape [n_obs]

    update_fields:
        X shape [batch, p, y, x]
        Y shape [batch, q, y, x]
        patch_weights shape [batch]
    """

    def __init__(self, x_names: list[str], y_names: list[str]) -> None:
        self.x_names = list(x_names)
        self.y_names = list(y_names)

        p = len(x_names)
        q = len(y_names)

        self.sw = np.zeros((p, q), dtype=np.float64)
        self.sx = np.zeros((p, q), dtype=np.float64)
        self.sy = np.zeros((p, q), dtype=np.float64)
        self.sx2 = np.zeros((p, q), dtype=np.float64)
        self.sy2 = np.zeros((p, q), dtype=np.float64)
        self.sxy = np.zeros((p, q), dtype=np.float64)
        self.sample_count = np.zeros((p, q), dtype=np.int64)

    def update_matrix(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)

        if x.ndim != 2 or y.ndim != 2:
            raise ValueError("update_matrix expects 2D X and Y.")
        if len(x) != len(y) or len(x) != len(w):
            raise ValueError("X/Y/weights row counts differ.")

        for i in range(x.shape[1]):
            xi = x[:, i]
            for j in range(y.shape[1]):
                yj = y[:, j]
                mask = np.isfinite(xi) & np.isfinite(yj) & np.isfinite(w)
                if not np.any(mask):
                    continue

                xv = xi[mask]
                yv = yj[mask]
                ww = w[mask]

                sw = ww.sum()
                self.sw[i, j] += sw
                self.sx[i, j] += np.dot(ww, xv)
                self.sy[i, j] += np.dot(ww, yv)
                self.sx2[i, j] += np.dot(ww, xv * xv)
                self.sy2[i, j] += np.dot(ww, yv * yv)
                self.sxy[i, j] += np.dot(ww, xv * yv)
                self.sample_count[i, j] += int(mask.sum())

    def update_fields(
        self,
        x: np.ndarray,
        y: np.ndarray,
        patch_weights: np.ndarray,
    ) -> None:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        w = np.asarray(patch_weights, dtype=np.float64)

        if x.ndim != 4 or y.ndim != 4:
            raise ValueError("update_fields expects [batch,channel,y,x].")
        if x.shape[0] != y.shape[0] or x.shape[0] != len(w):
            raise ValueError("Batch dimensions differ.")
        if x.shape[2:] != y.shape[2:]:
            raise ValueError("Spatial shapes differ.")

        # Dataset construction requires fully valid ERA5 inputs and Phase 0
        # verified valid targets. Fail loudly if this assumption is violated.
        if not np.isfinite(x).all():
            raise RuntimeError("Non-finite predictor value in sampled block.")
        if not np.isfinite(y).all():
            raise RuntimeError("Non-finite radar value in sampled block.")

        b, p, h, ww = x.shape
        q = y.shape[1]
        npx = h * ww

        xf = x.reshape(b, p, npx)
        yf = y.reshape(b, q, npx)

        weighted_mass = float(np.sum(w) * npx)

        sx_vec = np.einsum("bpn,b->p", xf, w, optimize=True)
        sy_vec = np.einsum("bqn,b->q", yf, w, optimize=True)
        sx2_vec = np.einsum("bpn,bpn,b->p", xf, xf, w, optimize=True)
        sy2_vec = np.einsum("bqn,bqn,b->q", yf, yf, w, optimize=True)
        sxy = np.einsum("bpn,bqn,b->pq", xf, yf, w, optimize=True)

        self.sw += weighted_mass
        self.sx += sx_vec[:, None]
        self.sy += sy_vec[None, :]
        self.sx2 += sx2_vec[:, None]
        self.sy2 += sy2_vec[None, :]
        self.sxy += sxy
        self.sample_count += b * npx

    def rows(self, extra: dict[str, Any]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []

        for i, x_name in enumerate(self.x_names):
            for j, y_name in enumerate(self.y_names):
                sw = self.sw[i, j]

                if sw <= 0:
                    corr = np.nan
                    mx = np.nan
                    my = np.nan
                    sx = np.nan
                    sy = np.nan
                else:
                    mx = self.sx[i, j] / sw
                    my = self.sy[i, j] / sw
                    vx = self.sx2[i, j] / sw - mx * mx
                    vy = self.sy2[i, j] / sw - my * my
                    cov = self.sxy[i, j] / sw - mx * my
                    sx = math.sqrt(max(vx, 0.0))
                    sy = math.sqrt(max(vy, 0.0))
                    corr = (
                        cov / math.sqrt(vx * vy)
                        if vx > 0 and vy > 0
                        else np.nan
                    )

                out.append({
                    **extra,
                    "predictor": x_name,
                    "radar_field": y_name,
                    "weighted_observation_mass": float(sw),
                    "sample_observations": int(self.sample_count[i, j]),
                    "predictor_mean": float(mx)
                    if np.isfinite(mx) else np.nan,
                    "predictor_std": float(sx)
                    if np.isfinite(sx) else np.nan,
                    "radar_mean": float(my)
                    if np.isfinite(my) else np.nan,
                    "radar_std": float(sy)
                    if np.isfinite(sy) else np.nan,
                    "pearson_r": float(corr)
                    if np.isfinite(corr) else np.nan,
                })
        return out


class ContrastAccumulator:
    def __init__(self, predictor_names: list[str]) -> None:
        p = len(predictor_names)
        self.predictor_names = list(predictor_names)
        self.sw = np.zeros(p, dtype=np.float64)
        self.sdelta = np.zeros(p, dtype=np.float64)
        self.sdelta2 = np.zeros(p, dtype=np.float64)
        self.snorm = np.zeros(p, dtype=np.float64)
        self.snorm2 = np.zeros(p, dtype=np.float64)
        self.spositive = np.zeros(p, dtype=np.float64)
        self.sample_patch_count = np.zeros(p, dtype=np.int64)

    def update(
        self,
        delta: np.ndarray,
        normalized_delta: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        delta = np.asarray(delta, dtype=np.float64)
        norm = np.asarray(normalized_delta, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)

        for i in range(delta.shape[1]):
            d = delta[:, i]
            n = norm[:, i]
            mask = np.isfinite(d) & np.isfinite(n) & np.isfinite(w)
            if not np.any(mask):
                continue

            dd = d[mask]
            nn = n[mask]
            ww = w[mask]

            self.sw[i] += ww.sum()
            self.sdelta[i] += np.dot(ww, dd)
            self.sdelta2[i] += np.dot(ww, dd * dd)
            self.snorm[i] += np.dot(ww, nn)
            self.snorm2[i] += np.dot(ww, nn * nn)
            self.spositive[i] += np.dot(ww, (dd > 0).astype(float))
            self.sample_patch_count[i] += int(mask.sum())

    def rows(self, extra: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        for i, name in enumerate(self.predictor_names):
            sw = self.sw[i]
            if sw <= 0:
                mean_d = std_d = mean_n = std_n = positive = np.nan
            else:
                mean_d = self.sdelta[i] / sw
                var_d = self.sdelta2[i] / sw - mean_d * mean_d
                std_d = math.sqrt(max(var_d, 0.0))

                mean_n = self.snorm[i] / sw
                var_n = self.snorm2[i] / sw - mean_n * mean_n
                std_n = math.sqrt(max(var_n, 0.0))

                positive = self.spositive[i] / sw

            rows.append({
                **extra,
                "predictor": name,
                "weighted_event_patch_mass": float(sw),
                "sample_event_patches": int(self.sample_patch_count[i]),
                "mean_event_minus_background": float(mean_d)
                if np.isfinite(mean_d) else np.nan,
                "std_event_minus_background": float(std_d)
                if np.isfinite(std_d) else np.nan,
                "mean_delta_in_patch_sigma": float(mean_n)
                if np.isfinite(mean_n) else np.nan,
                "std_delta_in_patch_sigma": float(std_n)
                if np.isfinite(std_n) else np.nan,
                "weighted_fraction_positive_delta": float(positive)
                if np.isfinite(positive) else np.nan,
            })
        return rows


def smooth_valid_interior(
    fields: np.ndarray,
    window: int,
) -> tuple[np.ndarray, int]:
    """
    Returns locally averaged predictors and the number of border pixels to
    remove so only windows fully contained inside the patch are evaluated.
    """
    if window == 1:
        return fields, 0

    radius = window // 2
    smoothed = ndimage.uniform_filter(
        fields,
        size=(1, 1, window, window),
        mode="nearest",
    )
    return smoothed, radius


def crop_interior(
    arr: np.ndarray,
    radius: int,
) -> np.ndarray:
    if radius == 0:
        return arr
    return arr[:, :, radius:-radius, radius:-radius]


def centered_within_patch(arr: np.ndarray) -> np.ndarray:
    return arr - arr.mean(axis=(2, 3), keepdims=True)


def offset_views(
    predictors: np.ndarray,
    radar: np.ndarray,
    dy: int,
    dx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pair predictor at (y,x) with radar at (y+dy,x+dx).
    """
    h, w = predictors.shape[-2:]

    if abs(dy) >= h or abs(dx) >= w:
        raise ValueError("Offset exceeds patch dimensions.")

    if dy >= 0:
        pya, pyb = 0, h - dy
        rya, ryb = dy, h
    else:
        pya, pyb = -dy, h
        rya, ryb = 0, h + dy

    if dx >= 0:
        pxa, pxb = 0, w - dx
        rxa, rxb = dx, w
    else:
        pxa, pxb = -dx, w
        rxa, rxb = 0, w + dx

    return (
        predictors[:, :, pya:pyb, pxa:pxb],
        radar[:, :, rya:ryb, rxa:rxb],
    )


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
        })

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    windows = parse_positive_ints(
        args.smoothing_windows,
        odd=True,
    )
    offset_steps = parse_positive_ints(args.offset_steps)

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
            "Phase 10 requires spatially aligned input and target patches."
        )

    patch_h, patch_w = map(int, target_arr.shape[-2:])
    if max(windows) > min(patch_h, patch_w):
        raise ValueError(
            f"Smoothing window too large for patch: {windows}"
        )
    if max(offset_steps) >= min(patch_h, patch_w):
        raise ValueError(
            f"Offset too large for patch: {offset_steps}"
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
    logger.info("CORRDIFF - PHASE 10 - SPATIAL ERA5 x RADAR")
    logger.info("Dataset              : %s", args.dataset_dir)
    logger.info("Input shape          : %s", tuple(input_arr.shape))
    logger.info("Target shape         : %s", tuple(target_arr.shape))
    logger.info("Raw channels         : %s", raw_names)
    logger.info("Derived channels     : %s", DERIVED_CHANNELS)
    logger.info("Smoothing windows    : %s", windows)
    logger.info("Offset steps         : %s", offset_steps)
    logger.info("Resolution           : %.3f km/pixel", args.radar_resolution_km)
    logger.info("=" * 80)

    # ------------------------------------------------------------------
    # PASS A - exact radar stratification
    # ------------------------------------------------------------------
    strata, strata_counts = classify_patch_strata(
        target_arr,
        logger,
    )
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

    # ------------------------------------------------------------------
    # Accumulators
    # ------------------------------------------------------------------
    global_pixel: dict[tuple[int, str], CorrMatrixAccumulator] = {}
    season_pixel: dict[
        tuple[str, int, str],
        CorrMatrixAccumulator,
    ] = {}

    for window in windows:
        for mode in ("raw", "within_patch_centered"):
            global_pixel[(window, mode)] = CorrMatrixAccumulator(
                predictor_names,
                RADAR_FIELDS,
            )
            for season in SEASON_ORDER:
                season_pixel[(season, window, mode)] = CorrMatrixAccumulator(
                    predictor_names,
                    RADAR_FIELDS,
                )

    offset_acc: dict[
        tuple[str, int, str],
        CorrMatrixAccumulator,
    ] = {}
    offset_specs: list[tuple[str, int, int, int]] = [
        ("CENTER", 0, 0, 0)
    ]
    for step in offset_steps:
        for label, (dy0, dx0) in OFFSET_DIRECTIONS.items():
            if label == "CENTER":
                continue
            offset_specs.append(
                (label, step, dy0 * step, dx0 * step)
            )
            for mode in ("raw", "within_patch_centered"):
                offset_acc[(label, step, mode)] = CorrMatrixAccumulator(
                    predictor_names,
                    RADAR_FIELDS,
                )
    for mode in ("raw", "within_patch_centered"):
        offset_acc[("CENTER", 0, mode)] = CorrMatrixAccumulator(
            predictor_names,
            RADAR_FIELDS,
        )

    contrast_global = {
        event_id: ContrastAccumulator(predictor_names)
        for event_id in EVENT_THRESHOLDS
    }
    contrast_season = {
        (season, event_id): ContrastAccumulator(predictor_names)
        for season in SEASON_ORDER
        for event_id in EVENT_THRESHOLDS
    }

    patch_level = CorrMatrixAccumulator(
        predictor_names,
        PATCH_RADAR_METRICS,
    )

    # ------------------------------------------------------------------
    # PASS B - chunk-aware input/target reads
    # ------------------------------------------------------------------
    chunk0 = (
        int(input_arr.chunks[0])
        if getattr(input_arr, "chunks", None)
        else 256
    )

    patch_indices = manifest["patch_index"].to_numpy(dtype=np.int64)
    chunk_ids = patch_indices // chunk0
    unique_chunk_ids = np.unique(chunk_ids)

    logger.info("=" * 80)
    logger.info("PHASE 10 - PASS B - ALIGNED INPUT/TARGET")
    logger.info("Chunks containing sample: %d", len(unique_chunk_ids))
    logger.info("=" * 80)

    processed = 0

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

        local_indices = (
            patch_indices[pos] - chunk_start
        ).astype(np.int64)

        raw = input_block[local_indices]
        tgt = target_block[local_indices]

        predictors, names_check = derive_predictors(
            raw,
            raw_names,
        )
        if names_check != predictor_names:
            raise RuntimeError("Predictor name ordering changed unexpectedly.")

        radar, radar_names_check = radar_fields_from_target(tgt)
        if radar_names_check != RADAR_FIELDS:
            raise RuntimeError("Radar field ordering changed unexpectedly.")

        weights = manifest.iloc[pos]["sampling_weight"].to_numpy(
            dtype=np.float64
        )
        seasons = (
            manifest.iloc[pos]["season_code"]
            .astype(str)
            .to_numpy()
        )

        if not np.isfinite(predictors).all():
            raise RuntimeError(
                "Non-finite ERA5 predictor in selected sample."
            )
        if not np.isfinite(radar).all():
            raise RuntimeError(
                "Non-finite radar target in selected sample."
            )

        # --------------------------------------------------------------
        # 1-3. Pixelwise raw/centered + smoothing scales
        # --------------------------------------------------------------
        for window in windows:
            x_smooth, radius = smooth_valid_interior(
                predictors,
                window,
            )
            x_eval = crop_interior(x_smooth, radius)
            y_eval = crop_interior(radar, radius)

            global_pixel[(window, "raw")].update_fields(
                x_eval,
                y_eval,
                weights,
            )
            global_pixel[
                (window, "within_patch_centered")
            ].update_fields(
                centered_within_patch(x_eval),
                centered_within_patch(y_eval),
                weights,
            )

            for season in SEASON_ORDER:
                sm = seasons == season
                if not np.any(sm):
                    continue

                season_pixel[
                    (season, window, "raw")
                ].update_fields(
                    x_eval[sm],
                    y_eval[sm],
                    weights[sm],
                )
                season_pixel[
                    (season, window, "within_patch_centered")
                ].update_fields(
                    centered_within_patch(x_eval[sm]),
                    centered_within_patch(y_eval[sm]),
                    weights[sm],
                )

        # --------------------------------------------------------------
        # 4. Within-patch event-vs-background contrasts
        # --------------------------------------------------------------
        x_flat = predictors.reshape(
            predictors.shape[0],
            predictors.shape[1],
            -1,
        )
        x_flat64 = x_flat.astype(np.float64, copy=False)
        total_sum = x_flat64.sum(axis=2)
        patch_std = predictors.std(
            axis=(2, 3),
            ddof=0,
        ).astype(np.float64)

        target_2d = (
            tgt[:, 0]
            if tgt.ndim == 4 and tgt.shape[1] == 1
            else np.squeeze(tgt)
        )

        for event_id in EVENT_THRESHOLDS:
            mask = event_mask(
                np.asarray(target_2d, dtype=np.float32),
                event_id,
            ).reshape(len(pos), -1)

            count_event = mask.sum(axis=1)
            count_bg = mask.shape[1] - count_event
            valid_patch = (count_event > 0) & (count_bg > 0)

            if not np.any(valid_patch):
                continue

            mf = mask.astype(np.float64)
            event_sum = np.einsum(
                "bpn,bn->bp",
                x_flat64,
                mf,
                optimize=True,
            )
            bg_sum = total_sum - event_sum

            event_mean = event_sum / np.maximum(
                count_event[:, None],
                1,
            )
            bg_mean = bg_sum / np.maximum(
                count_bg[:, None],
                1,
            )
            delta = event_mean - bg_mean

            norm = np.divide(
                delta,
                patch_std,
                out=np.full_like(delta, np.nan),
                where=patch_std > 1e-12,
            )

            contrast_global[event_id].update(
                delta[valid_patch],
                norm[valid_patch],
                weights[valid_patch],
            )

            for season in SEASON_ORDER:
                sm = valid_patch & (seasons == season)
                if np.any(sm):
                    contrast_season[
                        (season, event_id)
                    ].update(
                        delta[sm],
                        norm[sm],
                        weights[sm],
                    )

        # --------------------------------------------------------------
        # 5. Cross-offset associations
        # --------------------------------------------------------------
        for label, step, dy, dx in offset_specs:
            xp, yr = offset_views(
                predictors,
                radar,
                dy,
                dx,
            )
            offset_acc[(label, step, "raw")].update_fields(
                xp,
                yr,
                weights,
            )
            offset_acc[
                (label, step, "within_patch_centered")
            ].update_fields(
                centered_within_patch(xp),
                centered_within_patch(yr),
                weights,
            )

        # --------------------------------------------------------------
        # 6. Patch-level associations
        # --------------------------------------------------------------
        x_patch = predictors.mean(
            axis=(2, 3),
            dtype=np.float64,
        )
        y_patch, patch_names_check = patch_radar_metrics(tgt)
        if patch_names_check != PATCH_RADAR_METRICS:
            raise RuntimeError("Patch radar metric ordering changed.")

        patch_level.update_matrix(
            x_patch,
            y_patch,
            weights,
        )

        processed += len(pos)

        if ci == len(unique_chunk_ids) or ci % 200 == 0:
            logger.info(
                "PASS B: chunks %d/%d | patches %d/%d",
                ci,
                len(unique_chunk_ids),
                processed,
                len(manifest),
            )

    # ------------------------------------------------------------------
    # Save association tables
    # ------------------------------------------------------------------
    pixel_rows = []
    for (window, mode), acc in global_pixel.items():
        pixel_rows.extend(
            acc.rows({
                "smoothing_window_pixels": window,
                "smoothing_window_span_km": (
                    (window - 1) * args.radar_resolution_km
                ),
                "smoothing_half_width_km": (
                    (window // 2) * args.radar_resolution_km
                ),
                "association_mode": mode,
            })
        )

    pixel_df = pd.DataFrame(pixel_rows)
    pixel_df.to_parquet(
        args.output_dir / "pixel_spatial_associations.parquet",
        index=False,
    )

    season_rows = []
    for (season, window, mode), acc in season_pixel.items():
        season_rows.extend(
            acc.rows({
                "season_code": season,
                "smoothing_window_pixels": window,
                "smoothing_window_span_km": (
                    (window - 1) * args.radar_resolution_km
                ),
                "smoothing_half_width_km": (
                    (window // 2) * args.radar_resolution_km
                ),
                "association_mode": mode,
            })
        )

    pd.DataFrame(season_rows).to_parquet(
        args.output_dir
        / "pixel_spatial_associations_by_season.parquet",
        index=False,
    )

    offset_rows = []
    for label, step, dy, dx in offset_specs:
        distance_km = (
            math.sqrt(dy * dy + dx * dx)
            * args.radar_resolution_km
        )
        for mode in ("raw", "within_patch_centered"):
            offset_rows.extend(
                offset_acc[(label, step, mode)].rows({
                    "association_mode": mode,
                    "offset_direction": label,
                    "offset_step_pixels": step,
                    "radar_offset_dy_pixels": dy,
                    "radar_offset_dx_pixels": dx,
                    "offset_distance_km": float(distance_km),
                    "interpretation": (
                        "predictor(x) vs radar(x+delta)"
                    ),
                })
            )

    pd.DataFrame(offset_rows).to_parquet(
        args.output_dir
        / "spatial_cross_offset_associations.parquet",
        index=False,
    )

    contrast_rows = []
    for event_id, acc in contrast_global.items():
        contrast_rows.extend(
            acc.rows({
                "event_id": event_id,
                "threshold_dbz": EVENT_THRESHOLDS[event_id],
            })
        )
    pd.DataFrame(contrast_rows).to_parquet(
        args.output_dir / "within_patch_event_contrasts.parquet",
        index=False,
    )

    contrast_season_rows = []
    for (season, event_id), acc in contrast_season.items():
        contrast_season_rows.extend(
            acc.rows({
                "season_code": season,
                "event_id": event_id,
                "threshold_dbz": EVENT_THRESHOLDS[event_id],
            })
        )
    pd.DataFrame(contrast_season_rows).to_parquet(
        args.output_dir
        / "within_patch_event_contrasts_by_season.parquet",
        index=False,
    )

    pd.DataFrame(
        patch_level.rows({
            "aggregation": "patch_mean_predictor_vs_patch_radar_metric",
        })
    ).to_parquet(
        args.output_dir / "patch_level_associations.parquet",
        index=False,
    )

    # ------------------------------------------------------------------
    # Summary / QC
    # ------------------------------------------------------------------
    sample_ratio = len(manifest) / n
    warnings: list[str] = []

    if sample_ratio < 0.05:
        warnings.append(
            "The aligned input/target analysis uses a stratified patch sample "
            "containing <5% of all patches. Inverse stratum weights are used."
        )

    if channel_name_source == "canonical_12_channel_fallback":
        warnings.append(
            "Input channel names were not read from Zarr metadata; the "
            "canonical 12-channel builder order was used. Verify this order "
            "against dataset metadata before publication."
        )

    summary = {
        "phase_version": PHASE_VERSION,
        "dataset_dir": str(args.dataset_dir),
        "input_shape": [int(x) for x in input_arr.shape],
        "target_shape": [int(x) for x in target_arr.shape],
        "patch_shape": [patch_h, patch_w],
        "total_patches": n,
        "sampled_patches": int(len(manifest)),
        "sample_ratio": float(sample_ratio),
        "sample_per_stratum": int(args.sample_per_stratum),
        "seed": int(args.seed),
        "raw_channels": raw_names,
        "derived_channels": DERIVED_CHANNELS,
        "channel_name_source": channel_name_source,
        "radar_fields": RADAR_FIELDS,
        "smoothing_windows_pixels": windows,
        "offset_steps_pixels": offset_steps,
        "radar_resolution_km": float(args.radar_resolution_km),
        "local_timezone": args.local_timezone,
        "radar_unit": "dBZ",
        "stored_target": "log1p(clip(dBZ, min=0)) float32",
        "scope": (
            "aligned overlapping 32x32 training-patch distribution; "
            "not de-duplicated full-field radar climatology"
        ),
        "association_modes": {
            "raw": (
                "pixelwise predictor-radar association across the weighted "
                "sample; may include season/time/regime differences"
            ),
            "within_patch_centered": (
                "predictor and radar values de-meaned independently within "
                "each patch before correlation; emphasizes local spatial "
                "co-location and removes patch-level mean differences"
            ),
        },
        "warnings": warnings,
        "methodological_notes": [
            (
                "ERA5 fields have been spatially mapped to the CorrDiff grid. "
                "Fine-grid values must not be interpreted as independent "
                "meteorological observations below ERA5 native effective "
                "resolution."
            ),
            (
                "Smoothing-window associations are a screening analysis of "
                "local context. Formal scale decomposition is deferred to "
                "Phase 11."
            ),
            (
                "Within-patch event contrasts compare predictor means in event "
                "pixels against non-event pixels of the same patch, reducing "
                "large-scale background and seasonal confounding but not "
                "establishing causality."
            ),
            (
                "Cross-offset associations use predictor(x) versus "
                "radar(x+delta). A stronger displaced association can reflect "
                "spatial organization, interpolation, advection, or patch "
                "geometry and must not be read as causal displacement."
            ),
            (
                "Overlapping patches duplicate physical pixels and pixel pairs. "
                "Weighted observation mass is not an independent sample size."
            ),
            (
                "bulk_wind_diff variables are vector wind differences in m/s, "
                "not height-normalized vertical shear."
            ),
        ],
    }

    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("=" * 80)
    logger.info("PHASE 10 COMPLETE")
    logger.info("Output: %s", args.output_dir)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
