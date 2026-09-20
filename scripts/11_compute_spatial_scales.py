#!/usr/bin/env python3
"""
CorrDiff - Fase 11
Análise formal de escalas espaciais
===================================

Objetivo
--------
Separar a informação espacial do ERA5 e do radar por escala nominal da grade
CorrDiff, sem interpretar a interpolação do ERA5 como resolução meteorológica
física de 2 km.

A fase combina duas visões complementares:

1. Decomposição Haar 2D ortonormal:
   - energia espacial por escala;
   - correlação predictor-radar de detalhes na mesma escala;
   - correlação entre energia de detalhes de escalas diferentes em nível de patch.

2. Coarse graining por médias em blocos:
   - associação predictor-radar em grades agregadas de 2, 4, 8, 16 e 32 km;
   - modo raw e modo within_patch_centered.

Os patches são 32x32 e sobrepostos. A análise representa a distribuição
espacial apresentada ao modelo, não uma climatologia de campo completo.

Saídas
------
analysis_outputs/11_spatial_scales/
├── analysis_summary.json
├── patch_strata_counts.parquet
├── sampled_patch_manifest.parquet
├── input_channel_metadata.parquet
├── haar_scale_energy.parquet
├── haar_scale_energy_by_season.parquet
├── haar_same_scale_associations.parquet
├── haar_same_scale_associations_by_season.parquet
├── haar_patch_energy_cross_scale.parquet
├── coarse_grain_associations.parquet
├── coarse_grain_associations_by_season.parquet
└── phase11.log
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
        "Fase 11 requer zarr. Instale no ambiente de análise com: pip install zarr"
    ) from exc


PHASE_VERSION = "phase11-spatial-scales-v1-haar-coarsegraining"

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

HAAR_ORIENTATIONS = [
    "EW_DETAIL",
    "NS_DETAIL",
    "DIAGONAL_DETAIL",
]

COARSE_MODES = [
    "raw",
    "within_patch_centered",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CorrDiff Fase 11: análise formal de escalas espaciais."
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("datasets/corrdiff_2011_2024"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs/11_spatial_scales"),
    )
    p.add_argument(
        "--sample-per-stratum",
        type=int,
        default=3000,
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--radar-resolution-km",
        type=float,
        default=2.0,
    )
    p.add_argument(
        "--coarse-blocks",
        default="1,2,4,8,16",
        help=(
            "Larguras de bloco, em pixels, para coarse graining. "
            "Devem dividir a dimensão espacial do patch."
        ),
    )
    p.add_argument(
        "--local-timezone",
        default="America/Sao_Paulo",
    )
    p.add_argument(
        "--channel-names",
        default=None,
        help=(
            "Lista separada por vírgulas dos canais raw. Se omitida e houver "
            "12 canais, usa a ordem canônica do builder."
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
    logger = logging.getLogger("corrdiff.phase11")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(path / "phase11.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def parse_positive_ints(text: str) -> list[int]:
    vals = sorted(
        set(int(x.strip()) for x in text.split(",") if x.strip())
    )
    if not vals or any(v <= 0 for v in vals):
        raise ValueError("Expected positive integers.")
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
            "Phase 11 derived predictors require the canonical raw fields. "
            f"Missing: {missing}"
        )


def predictor_metadata(
    raw_names: list[str],
    source: str,
) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(raw_names):
        rows.append({
            "field": name,
            "field_group": "predictor",
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
            "field": name,
            "field_group": "predictor",
            "source": "derived",
            "input_channel_index": np.nan,
            "channel_name_source": source,
            "definition": defs[name],
        })

    for field in RADAR_FIELDS:
        rows.append({
            "field": field,
            "field_group": "radar",
            "source": (
                "stored_target"
                if field == "target_log1p"
                else "threshold_mask"
            ),
            "input_channel_index": np.nan,
            "channel_name_source": "target",
            "definition": field,
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
    logger.info("PHASE 11 - PASS A - TARGET STRATA")
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


def radar_fields_from_target(
    target_block: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
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


def haar_step(
    arr: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    One orthonormal 2D Haar step.

    Input: [batch, channel, y, x] with even y/x.

    EW_DETAIL contrasts adjacent columns.
    NS_DETAIL contrasts adjacent rows.
    DIAGONAL_DETAIL is the diagonal contrast.

    The 1/2 normalization preserves sum-of-squares energy across the 2x2
    transform.
    """
    h, w = arr.shape[-2:]
    if h % 2 or w % 2:
        raise ValueError("Haar step requires even spatial dimensions.")

    a = arr[..., 0::2, 0::2]
    b = arr[..., 0::2, 1::2]
    c = arr[..., 1::2, 0::2]
    d = arr[..., 1::2, 1::2]

    ll = (a + b + c + d) * 0.5
    ew = (a - b + c - d) * 0.5
    ns = (a + b - c - d) * 0.5
    diag = (a - b - c + d) * 0.5

    return ll, {
        "EW_DETAIL": ew,
        "NS_DETAIL": ns,
        "DIAGONAL_DETAIL": diag,
    }


def haar_decompose(
    arr: np.ndarray,
    levels: int,
) -> tuple[list[dict[str, np.ndarray]], np.ndarray]:
    approx = np.asarray(arr, dtype=np.float32)
    details: list[dict[str, np.ndarray]] = []

    for _ in range(levels):
        approx, d = haar_step(approx)
        details.append(d)

    return details, approx


def block_mean(
    arr: np.ndarray,
    block: int,
) -> np.ndarray:
    b, c, h, w = arr.shape
    if h % block or w % block:
        raise ValueError(
            f"Block {block} does not divide spatial shape {(h, w)}."
        )
    return arr.reshape(
        b,
        c,
        h // block,
        block,
        w // block,
        block,
    ).mean(axis=(3, 5), dtype=np.float64).astype(np.float32)


def centered_within_patch(arr: np.ndarray) -> np.ndarray:
    return arr - arr.mean(axis=(2, 3), keepdims=True)


class CorrMatrixAccumulator:
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

        if not np.isfinite(x).all():
            raise RuntimeError("Non-finite predictor coefficient/value.")
        if not np.isfinite(y).all():
            raise RuntimeError("Non-finite radar coefficient/value.")

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
        out = []

        for i, x_name in enumerate(self.x_names):
            for j, y_name in enumerate(self.y_names):
                sw = self.sw[i, j]

                if sw <= 0:
                    corr = mx = my = sx = sy = np.nan
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


class EnergyAccumulator:
    """
    Weighted sum-of-squares energy for fields.

    Energy is accumulated as sum(w_patch * coefficient^2), preserving the
    orthonormal Haar decomposition meaning across levels with different
    coefficient counts.
    """

    def __init__(self, field_names: list[str]) -> None:
        p = len(field_names)
        self.field_names = list(field_names)
        self.energy = np.zeros(p, dtype=np.float64)
        self.weighted_coeff_mass = np.zeros(p, dtype=np.float64)
        self.sample_coeff_count = np.zeros(p, dtype=np.int64)

    def update(
        self,
        fields: np.ndarray,
        patch_weights: np.ndarray,
    ) -> None:
        a = np.asarray(fields, dtype=np.float64)
        w = np.asarray(patch_weights, dtype=np.float64)

        if a.ndim != 4:
            raise ValueError("EnergyAccumulator expects [batch,field,y,x].")
        if len(w) != a.shape[0]:
            raise ValueError("Weight batch dimension mismatch.")
        if not np.isfinite(a).all():
            raise RuntimeError("Non-finite coefficient in energy calculation.")

        npx = a.shape[2] * a.shape[3]
        e = np.einsum(
            "bfyx,bfyx,b->f",
            a,
            a,
            w,
            optimize=True,
        )
        self.energy += e
        self.weighted_coeff_mass += np.sum(w) * npx
        self.sample_coeff_count += a.shape[0] * npx

    def rows(
        self,
        extra: dict[str, Any],
    ) -> list[dict[str, Any]]:
        rows = []
        for i, name in enumerate(self.field_names):
            mass = self.weighted_coeff_mass[i]
            rms = (
                math.sqrt(self.energy[i] / mass)
                if mass > 0
                else np.nan
            )
            rows.append({
                **extra,
                "field": name,
                "weighted_energy": float(self.energy[i]),
                "weighted_coefficient_mass": float(mass),
                "sample_coefficients": int(self.sample_coeff_count[i]),
                "rms_coefficient": float(rms)
                if np.isfinite(rms) else np.nan,
            })
        return rows


def patch_detail_rms(
    details: dict[str, np.ndarray],
) -> np.ndarray:
    """
    RMS pooled over the three Haar orientations at a level.
    Returns [batch, channel].
    """
    sq = None
    n_coeff = 0

    for orientation in HAAR_ORIENTATIONS:
        a = np.asarray(details[orientation], dtype=np.float64)
        e = np.sum(a * a, axis=(2, 3))
        sq = e if sq is None else sq + e
        n_coeff += a.shape[2] * a.shape[3]

    return np.sqrt(sq / n_coeff)


def add_energy_fractions(
    df: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    out = df.copy()

    detail = out[out["orientation"] == "ALL_DETAIL"].copy()
    totals = (
        detail.groupby(group_cols, observed=True)["weighted_energy"]
        .sum()
        .rename("_total_detail_energy")
        .reset_index()
    )

    out = out.merge(
        totals,
        on=group_cols,
        how="left",
        validate="many_to_one",
    )
    out["detail_energy_fraction"] = np.where(
        out["orientation"] == "ALL_DETAIL",
        np.divide(
            out["weighted_energy"],
            out["_total_detail_energy"],
            out=np.full(len(out), np.nan, dtype=float),
            where=out["_total_detail_energy"].to_numpy() > 0,
        ),
        np.nan,
    )
    return out.drop(columns="_total_detail_energy")


def main() -> None:
    args = parse_args()
    coarse_blocks = parse_positive_ints(args.coarse_blocks)

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
            "Phase 11 requires aligned input and target patch shapes."
        )

    patch_h, patch_w = map(int, target_arr.shape[-2:])
    if patch_h != patch_w:
        raise RuntimeError(
            "Phase 11 Haar decomposition currently requires square patches."
        )

    levels = int(math.log2(patch_h))
    if 2 ** levels != patch_h:
        raise RuntimeError(
            "Patch size must be a power of two for the Haar decomposition."
        )

    for block in coarse_blocks:
        if patch_h % block or patch_w % block:
            raise ValueError(
                f"Coarse block {block} does not divide patch shape "
                f"{(patch_h, patch_w)}."
            )
        if block >= patch_h:
            raise ValueError(
                f"Coarse block {block} leaves only one cell and cannot support "
                "within-patch centered correlation. Use blocks < patch size."
            )

    raw_names, channel_name_source = resolve_raw_channel_names(
        input_arr,
        args.channel_names,
    )
    validate_required_raw_channels(raw_names)
    predictor_names = raw_names + DERIVED_CHANNELS

    metadata = predictor_metadata(raw_names, channel_name_source)
    metadata.to_parquet(
        args.output_dir / "input_channel_metadata.parquet",
        index=False,
    )

    logger.info("=" * 80)
    logger.info("CORRDIFF - PHASE 11 - SPATIAL SCALES")
    logger.info("Dataset              : %s", args.dataset_dir)
    logger.info("Input shape          : %s", tuple(input_arr.shape))
    logger.info("Target shape         : %s", tuple(target_arr.shape))
    logger.info("Patch size           : %dx%d", patch_h, patch_w)
    logger.info("Haar levels          : %d", levels)
    logger.info("Coarse blocks        : %s", coarse_blocks)
    logger.info("Resolution           : %.3f km/pixel", args.radar_resolution_km)
    logger.info("Raw channels         : %s", raw_names)
    logger.info("=" * 80)

    # ------------------------------------------------------------------
    # PASS A - exact target strata
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
    all_field_names = predictor_names + RADAR_FIELDS
    field_group = {
        **{name: "predictor" for name in predictor_names},
        **{name: "radar" for name in RADAR_FIELDS},
    }

    # Energy per level/orientation, global + season.
    energy_global: dict[
        tuple[int, str],
        EnergyAccumulator,
    ] = {}
    energy_season: dict[
        tuple[str, int, str],
        EnergyAccumulator,
    ] = {}

    for level in range(1, levels + 1):
        for orientation in HAAR_ORIENTATIONS + ["ALL_DETAIL"]:
            energy_global[(level, orientation)] = EnergyAccumulator(
                all_field_names
            )
            for season in SEASON_ORDER:
                energy_season[
                    (season, level, orientation)
                ] = EnergyAccumulator(all_field_names)

    # Same-scale Haar coefficient correlations.
    haar_corr: dict[
        tuple[int, str],
        CorrMatrixAccumulator,
    ] = {}
    haar_corr_season: dict[
        tuple[str, int, str],
        CorrMatrixAccumulator,
    ] = {}

    for level in range(1, levels + 1):
        for orientation in HAAR_ORIENTATIONS + ["ALL_DETAIL"]:
            haar_corr[(level, orientation)] = CorrMatrixAccumulator(
                predictor_names,
                RADAR_FIELDS,
            )
            for season in SEASON_ORDER:
                haar_corr_season[
                    (season, level, orientation)
                ] = CorrMatrixAccumulator(
                    predictor_names,
                    RADAR_FIELDS,
                )

    # Patch-level cross-scale correlations of pooled detail RMS.
    cross_scale = {
        (px, ry): CorrMatrixAccumulator(
            predictor_names,
            RADAR_FIELDS,
        )
        for px in range(1, levels + 1)
        for ry in range(1, levels + 1)
    }

    # Coarse-grained associations.
    coarse_corr: dict[
        tuple[int, str],
        CorrMatrixAccumulator,
    ] = {}
    coarse_corr_season: dict[
        tuple[str, int, str],
        CorrMatrixAccumulator,
    ] = {}

    for block in coarse_blocks:
        for mode in COARSE_MODES:
            coarse_corr[(block, mode)] = CorrMatrixAccumulator(
                predictor_names,
                RADAR_FIELDS,
            )
            for season in SEASON_ORDER:
                coarse_corr_season[
                    (season, block, mode)
                ] = CorrMatrixAccumulator(
                    predictor_names,
                    RADAR_FIELDS,
                )

    # ------------------------------------------------------------------
    # PASS B - aligned sampled patches
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
    logger.info("PHASE 11 - PASS B - SCALE DECOMPOSITION")
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
            raise RuntimeError("Predictor ordering changed unexpectedly.")

        radar, radar_names_check = radar_fields_from_target(tgt)
        if radar_names_check != RADAR_FIELDS:
            raise RuntimeError("Radar ordering changed unexpectedly.")

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
                "Non-finite radar field in selected sample."
            )

        # --------------------------------------------------------------
        # Haar decomposition
        # --------------------------------------------------------------
        x_details, _ = haar_decompose(predictors, levels)
        y_details, _ = haar_decompose(radar, levels)

        x_patch_rms_by_level: dict[int, np.ndarray] = {}
        y_patch_rms_by_level: dict[int, np.ndarray] = {}

        for level in range(1, levels + 1):
            xd = x_details[level - 1]
            yd = y_details[level - 1]

            x_patch_rms_by_level[level] = patch_detail_rms(xd)
            y_patch_rms_by_level[level] = patch_detail_rms(yd)

            # Orientation-specific energy and correlation.
            for orientation in HAAR_ORIENTATIONS:
                xa = xd[orientation]
                ya = yd[orientation]
                both = np.concatenate([xa, ya], axis=1)

                energy_global[
                    (level, orientation)
                ].update(
                    both,
                    weights,
                )
                haar_corr[
                    (level, orientation)
                ].update_fields(
                    xa,
                    ya,
                    weights,
                )

                for season in SEASON_ORDER:
                    sm = seasons == season
                    if not np.any(sm):
                        continue
                    energy_season[
                        (season, level, orientation)
                    ].update(
                        both[sm],
                        weights[sm],
                    )
                    haar_corr_season[
                        (season, level, orientation)
                    ].update_fields(
                        xa[sm],
                        ya[sm],
                        weights[sm],
                    )

            # Pooled orientations: update sequentially into same accumulator.
            for orientation in HAAR_ORIENTATIONS:
                xa = xd[orientation]
                ya = yd[orientation]
                both = np.concatenate([xa, ya], axis=1)

                energy_global[
                    (level, "ALL_DETAIL")
                ].update(
                    both,
                    weights,
                )
                haar_corr[
                    (level, "ALL_DETAIL")
                ].update_fields(
                    xa,
                    ya,
                    weights,
                )

                for season in SEASON_ORDER:
                    sm = seasons == season
                    if not np.any(sm):
                        continue
                    energy_season[
                        (season, level, "ALL_DETAIL")
                    ].update(
                        both[sm],
                        weights[sm],
                    )
                    haar_corr_season[
                        (season, level, "ALL_DETAIL")
                    ].update_fields(
                        xa[sm],
                        ya[sm],
                        weights[sm],
                    )

        # Cross-scale patch energy correlation.
        for px in range(1, levels + 1):
            for ry in range(1, levels + 1):
                cross_scale[(px, ry)].update_matrix(
                    x_patch_rms_by_level[px],
                    y_patch_rms_by_level[ry],
                    weights,
                )

        # --------------------------------------------------------------
        # Coarse graining
        # --------------------------------------------------------------
        for block in coarse_blocks:
            xb = block_mean(predictors, block)
            yb = block_mean(radar, block)

            coarse_corr[(block, "raw")].update_fields(
                xb,
                yb,
                weights,
            )
            coarse_corr[
                (block, "within_patch_centered")
            ].update_fields(
                centered_within_patch(xb),
                centered_within_patch(yb),
                weights,
            )

            for season in SEASON_ORDER:
                sm = seasons == season
                if not np.any(sm):
                    continue

                coarse_corr_season[
                    (season, block, "raw")
                ].update_fields(
                    xb[sm],
                    yb[sm],
                    weights[sm],
                )
                coarse_corr_season[
                    (season, block, "within_patch_centered")
                ].update_fields(
                    centered_within_patch(xb[sm]),
                    centered_within_patch(yb[sm]),
                    weights[sm],
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
    # Save Haar energy
    # ------------------------------------------------------------------
    energy_rows = []
    for (level, orientation), acc in energy_global.items():
        support_pixels = 2 ** level
        support_km = support_pixels * args.radar_resolution_km
        rows = acc.rows({
            "haar_level": level,
            "nominal_support_pixels": support_pixels,
            "nominal_support_km": float(support_km),
            "orientation": orientation,
        })
        for row in rows:
            row["field_group"] = field_group[row["field"]]
        energy_rows.extend(rows)

    energy_df = pd.DataFrame(energy_rows)
    energy_df = add_energy_fractions(
        energy_df,
        ["field", "field_group"],
    )
    energy_df.to_parquet(
        args.output_dir / "haar_scale_energy.parquet",
        index=False,
    )

    energy_season_rows = []
    for (season, level, orientation), acc in energy_season.items():
        support_pixels = 2 ** level
        support_km = support_pixels * args.radar_resolution_km
        rows = acc.rows({
            "season_code": season,
            "haar_level": level,
            "nominal_support_pixels": support_pixels,
            "nominal_support_km": float(support_km),
            "orientation": orientation,
        })
        for row in rows:
            row["field_group"] = field_group[row["field"]]
        energy_season_rows.extend(rows)

    energy_season_df = pd.DataFrame(energy_season_rows)
    energy_season_df = add_energy_fractions(
        energy_season_df,
        ["season_code", "field", "field_group"],
    )
    energy_season_df.to_parquet(
        args.output_dir / "haar_scale_energy_by_season.parquet",
        index=False,
    )

    # ------------------------------------------------------------------
    # Save same-scale Haar associations
    # ------------------------------------------------------------------
    haar_rows = []
    for (level, orientation), acc in haar_corr.items():
        support_pixels = 2 ** level
        support_km = support_pixels * args.radar_resolution_km
        haar_rows.extend(
            acc.rows({
                "haar_level": level,
                "nominal_support_pixels": support_pixels,
                "nominal_support_km": float(support_km),
                "orientation": orientation,
            })
        )

    pd.DataFrame(haar_rows).to_parquet(
        args.output_dir / "haar_same_scale_associations.parquet",
        index=False,
    )

    haar_season_rows = []
    for (season, level, orientation), acc in haar_corr_season.items():
        support_pixels = 2 ** level
        support_km = support_pixels * args.radar_resolution_km
        haar_season_rows.extend(
            acc.rows({
                "season_code": season,
                "haar_level": level,
                "nominal_support_pixels": support_pixels,
                "nominal_support_km": float(support_km),
                "orientation": orientation,
            })
        )

    pd.DataFrame(haar_season_rows).to_parquet(
        args.output_dir
        / "haar_same_scale_associations_by_season.parquet",
        index=False,
    )

    # ------------------------------------------------------------------
    # Save cross-scale patch energy correlations
    # ------------------------------------------------------------------
    cross_rows = []
    for (px, ry), acc in cross_scale.items():
        cross_rows.extend(
            acc.rows({
                "predictor_haar_level": px,
                "predictor_nominal_support_km": float(
                    (2 ** px) * args.radar_resolution_km
                ),
                "radar_haar_level": ry,
                "radar_nominal_support_km": float(
                    (2 ** ry) * args.radar_resolution_km
                ),
                "quantity": (
                    "patch RMS detail energy correlation across scales"
                ),
            })
        )

    pd.DataFrame(cross_rows).to_parquet(
        args.output_dir / "haar_patch_energy_cross_scale.parquet",
        index=False,
    )

    # ------------------------------------------------------------------
    # Save coarse-grain associations
    # ------------------------------------------------------------------
    coarse_rows = []
    for (block, mode), acc in coarse_corr.items():
        coarse_rows.extend(
            acc.rows({
                "block_pixels": block,
                "nominal_block_width_km": float(
                    block * args.radar_resolution_km
                ),
                "coarse_grid_size": int(patch_h // block),
                "association_mode": mode,
            })
        )

    pd.DataFrame(coarse_rows).to_parquet(
        args.output_dir / "coarse_grain_associations.parquet",
        index=False,
    )

    coarse_season_rows = []
    for (season, block, mode), acc in coarse_corr_season.items():
        coarse_season_rows.extend(
            acc.rows({
                "season_code": season,
                "block_pixels": block,
                "nominal_block_width_km": float(
                    block * args.radar_resolution_km
                ),
                "coarse_grid_size": int(patch_h // block),
                "association_mode": mode,
            })
        )

    pd.DataFrame(coarse_season_rows).to_parquet(
        args.output_dir
        / "coarse_grain_associations_by_season.parquet",
        index=False,
    )

    # ------------------------------------------------------------------
    # Summary / QC
    # ------------------------------------------------------------------
    sample_ratio = len(manifest) / n
    warnings = []

    if sample_ratio < 0.05:
        warnings.append(
            "The scale analysis uses a stratified sample containing <5% of "
            "all patches. Inverse stratum weights are used."
        )

    if channel_name_source == "canonical_12_channel_fallback":
        warnings.append(
            "Input channel names were not read from Zarr metadata; the "
            "canonical 12-channel builder order was used. Verify this order "
            "against dataset metadata before publication."
        )

    scale_supports = [
        {
            "haar_level": level,
            "nominal_support_pixels": 2 ** level,
            "nominal_support_km": float(
                (2 ** level) * args.radar_resolution_km
            ),
        }
        for level in range(1, levels + 1)
    ]

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
        "radar_resolution_km": float(args.radar_resolution_km),
        "haar_levels": levels,
        "haar_scale_supports": scale_supports,
        "coarse_blocks_pixels": coarse_blocks,
        "coarse_block_widths_km": [
            float(b * args.radar_resolution_km)
            for b in coarse_blocks
        ],
        "local_timezone": args.local_timezone,
        "radar_unit": "dBZ",
        "stored_target": "log1p(clip(dBZ, min=0)) float32",
        "scope": (
            "aligned overlapping 32x32 training-patch scale distribution; "
            "not de-duplicated full-field climatology"
        ),
        "warnings": warnings,
        "methodological_notes": [
            (
                "Haar nominal support is a dyadic spatial support on the "
                "2-km CorrDiff grid. It is not the physical/native resolution "
                "of ERA5 and must not be interpreted as such."
            ),
            (
                "The orthonormal Haar detail-energy fractions quantify how "
                "spatial variance presented to the model is distributed over "
                "dyadic supports. They are not a Fourier power spectrum."
            ),
            (
                "Fine-scale variation in interpolated ERA5 channels can be "
                "generated by interpolation of broad gradients. It does not "
                "represent independent meteorological observations at 2 km."
            ),
            (
                "Coarse-grain correlations aggregate both predictor and radar "
                "before association. within_patch_centered mode removes each "
                "coarse patch mean and emphasizes spatial co-variation."
            ),
            (
                "The 64-km Haar level spans the complete 32x32 patch and is "
                "especially sensitive to patch boundaries and patch placement."
            ),
            (
                "Overlapping patches duplicate physical structures. Weighted "
                "coefficient/observation mass is not an independent sample size."
            ),
            (
                "The continuous radar decomposition uses stored log1p target "
                "values; event masks use fixed stored-domain dBZ thresholds."
            ),
            (
                "bulk_wind_diff variables are vector wind differences in m/s, "
                "not height-normalized vertical shear."
            ),
            (
                "All associations are descriptive and do not establish causal "
                "meteorological relationships."
            ),
        ],
    }

    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("=" * 80)
    logger.info("PHASE 11 COMPLETE")
    logger.info("Output: %s", args.output_dir)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
