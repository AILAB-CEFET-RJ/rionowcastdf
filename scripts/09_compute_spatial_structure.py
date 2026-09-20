#!/usr/bin/env python3
"""
CorrDiff - Fase 9
Estrutura espacial do radar
============================

A Fase 9 caracteriza a estrutura espacial do target radar dentro dos patches
32x32 armazenados no dataset CorrDiff.

Escopo
------
1. Scan exato de todos os patches para classificar intensidade máxima.
2. Amostragem estratificada reprodutível por intensidade.
3. Correlação espacial / semivariograma em múltiplas direções e escalas.
4. Morfologia de eventos em >0, >=20, >=30, >=40 e >=45 dBZ.
5. Estrutura morfológica global e por estação.
6. Diagnósticos explícitos de truncamento por borda do patch.

IMPORTANTE
----------
Esta fase NÃO reconstrói um campo radar completo de 70x84. Ela caracteriza a
distribuição espacial apresentada ao modelo nos patches sobrepostos 32x32.

Consequentemente:
- pares/pixels físicos podem aparecer em mais de um patch;
- componentes conectados podem ser truncados pelas bordas dos patches;
- os resultados não devem ser chamados de climatologia espacial de campo
  completo.

O objetivo é descrever a geometria e as escalas espaciais da distribuição de
treinamento do CorrDiff.

Semântica do target
-------------------
O builder armazena:
    target = log1p(clip(dBZ, min=0))

Todos os thresholds são aplicados diretamente no domínio armazenado float32,
seguindo a correção metodológica da Fase 4 v3.1.

Saídas
------
analysis_outputs/09_spatial_structure/
├── analysis_summary.json
├── patch_strata_counts.parquet
├── sampled_patch_manifest.parquet
├── spatial_lag_statistics.parquet
├── patch_morphology_sample.parquet
├── morphology_summary.parquet
├── morphology_by_season.parquet
└── phase9.log
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
    from scipy import ndimage
except ImportError as exc:
    raise SystemExit(
        "Fase 9 requer scipy. Instale com: pip install scipy"
    ) from exc


PHASE_VERSION = "phase9-radar-spatial-structure-v1-patch-distribution"

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

DIRECTIONS = {
    "EW": (0, 1),
    "NS": (1, 0),
    "NW_SE": (1, 1),
    "NE_SW": (1, -1),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CorrDiff Fase 9: estrutura espacial do radar."
    )
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("datasets/corrdiff_2011_2024"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs/09_spatial_structure"),
    )
    p.add_argument(
        "--sample-per-stratum",
        type=int,
        default=5000,
        help=(
            "Máximo de patches amostrados em cada estrato de intensidade. "
            "Estratos menores são incluídos integralmente."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    p.add_argument(
        "--spatial-steps",
        default="1,2,4,8,16",
        help="Deslocamentos em pixels usados na análise espacial.",
    )
    p.add_argument(
        "--radar-resolution-km",
        type=float,
        default=2.0,
        help="Resolução horizontal do radar em km/pixel.",
    )
    p.add_argument(
        "--local-timezone",
        default="America/Sao_Paulo",
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
    logger = logging.getLogger("corrdiff.phase9")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(path / "phase9.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def parse_steps(text: str) -> list[int]:
    vals = sorted(
        set(int(x.strip()) for x in text.split(",") if x.strip())
    )
    if not vals or any(v <= 0 for v in vals):
        raise ValueError("Spatial steps must be positive integers.")
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

    unit = infer_datetime_unit(arr)
    return pd.DatetimeIndex(
        pd.to_datetime(arr, unit=unit, utc=True)
    )


def stored_threshold(dbz: float) -> np.float32:
    return np.float32(
        np.log1p(np.float32(dbz))
    )


STORED_THRESHOLDS = {
    event_id: stored_threshold(dbz)
    for event_id, dbz in EVENT_THRESHOLDS.items()
    if dbz > 0
}


def event_mask(stored_patch: np.ndarray, event_id: str) -> np.ndarray:
    if event_id == "gt_0":
        return stored_patch > np.float32(0.0)
    return stored_patch >= STORED_THRESHOLDS[event_id]


def get_target_2d(block_row: np.ndarray) -> np.ndarray:
    arr = np.asarray(block_row, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise RuntimeError(
            f"Expected 2D target patch after squeeze; got shape={arr.shape}"
        )
    return arr


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
    seasons = [season_from_month(int(m)) for m in months]

    return pd.DataFrame({
        "timestamp_utc": utc,
        "year_utc": np.asarray(utc.year, dtype=np.int16),
        "month_local": months,
        "hour_local": np.asarray(local.hour, dtype=np.int8),
        "season_code": pd.Categorical(
            seasons,
            categories=SEASON_ORDER,
            ordered=True,
        ),
    })


def classify_patch_strata(
    target,
    logger: logging.Logger,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    PASS A: classifica todos os patches por máximo de refletividade no domínio
    armazenado, sem expm1.
    """
    n = int(target.shape[0])
    strata = np.empty(n, dtype=np.uint8)

    chunk0 = (
        int(target.chunks[0])
        if getattr(target, "chunks", None)
        else 256
    )

    logger.info("=" * 80)
    logger.info("PHASE 9 - PASS A - PATCH INTENSITY STRATA")
    logger.info("Target patches      : %d", n)
    logger.info("Target shape        : %s", tuple(target.shape))
    logger.info("First-axis chunk    : %d", chunk0)
    logger.info("=" * 80)

    t20 = STORED_THRESHOLDS["ge_20"]
    t30 = STORED_THRESHOLDS["ge_30"]
    t40 = STORED_THRESHOLDS["ge_40"]
    t45 = STORED_THRESHOLDS["ge_45"]

    for start in range(0, n, chunk0):
        end = min(n, start + chunk0)
        block = np.asarray(target[start:end], dtype=np.float32)
        flat = block.reshape(end - start, -1)
        mx = np.max(flat, axis=1)

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
    rows = []
    for sid in range(6):
        rows.append({
            "stratum_id": sid,
            "stratum": STRATUM_NAMES[sid],
            "patch_count": int(counts[sid]),
            "patch_ratio": float(counts[sid] / n),
        })

    return strata, pd.DataFrame(rows)


def build_sample_manifest(
    strata: np.ndarray,
    timestamps_raw: np.ndarray,
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
                rng.choice(
                    candidates,
                    size=n_sample,
                    replace=False,
                )
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
        raise RuntimeError("Sampling produced no patches.")

    # Patch-level timestamps are already aligned to target rows.
    patch_indices = manifest["patch_index"].to_numpy(dtype=np.int64)
    tm = time_meta.iloc[patch_indices].reset_index(drop=True)

    manifest = pd.concat(
        [manifest.reset_index(drop=True), tm],
        axis=1,
    )
    return manifest


class PairAccumulator:
    """Weighted Pearson + semivariance accumulator."""

    __slots__ = (
        "sum_weight",
        "sum_x",
        "sum_y",
        "sum_x2",
        "sum_y2",
        "sum_xy",
        "sum_sqdiff",
        "raw_pair_count",
    )

    def __init__(self) -> None:
        self.sum_weight = 0.0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_x2 = 0.0
        self.sum_y2 = 0.0
        self.sum_xy = 0.0
        self.sum_sqdiff = 0.0
        self.raw_pair_count = 0

    def update(
        self,
        x: np.ndarray,
        y: np.ndarray,
        patch_weight: float,
    ) -> None:
        xf = np.asarray(x, dtype=np.float64).ravel()
        yf = np.asarray(y, dtype=np.float64).ravel()

        mask = np.isfinite(xf) & np.isfinite(yf)
        if not np.any(mask):
            return

        xf = xf[mask]
        yf = yf[mask]

        n = int(len(xf))
        w = float(patch_weight)

        self.sum_weight += w * n
        self.sum_x += w * float(xf.sum())
        self.sum_y += w * float(yf.sum())
        self.sum_x2 += w * float(np.dot(xf, xf))
        self.sum_y2 += w * float(np.dot(yf, yf))
        self.sum_xy += w * float(np.dot(xf, yf))
        d = xf - yf
        self.sum_sqdiff += w * float(np.dot(d, d))
        self.raw_pair_count += n

    def finalize(self) -> dict[str, float | int]:
        sw = self.sum_weight
        if sw <= 0:
            return {
                "weighted_pair_mass": 0.0,
                "sample_pair_count": 0,
                "pearson_r": np.nan,
                "semivariance": np.nan,
                "mean_x": np.nan,
                "mean_y": np.nan,
            }

        mx = self.sum_x / sw
        my = self.sum_y / sw

        vx = self.sum_x2 / sw - mx * mx
        vy = self.sum_y2 / sw - my * my
        cov = self.sum_xy / sw - mx * my

        if vx > 0 and vy > 0:
            corr = cov / math.sqrt(vx * vy)
        else:
            corr = np.nan

        return {
            "weighted_pair_mass": float(sw),
            "sample_pair_count": int(self.raw_pair_count),
            "pearson_r": float(corr) if np.isfinite(corr) else np.nan,
            "semivariance": float(0.5 * self.sum_sqdiff / sw),
            "mean_x": float(mx),
            "mean_y": float(my),
        }


def shifted_pair(
    arr: np.ndarray,
    dy: int,
    dx: int,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = arr.shape

    if abs(dy) >= h or abs(dx) >= w:
        return np.empty((0,), dtype=arr.dtype), np.empty((0,), dtype=arr.dtype)

    if dy >= 0:
        y0a, y1a = 0, h - dy
        y0b, y1b = dy, h
    else:
        y0a, y1a = -dy, h
        y0b, y1b = 0, h + dy

    if dx >= 0:
        x0a, x1a = 0, w - dx
        x0b, x1b = dx, w
    else:
        x0a, x1a = -dx, w
        x0b, x1b = 0, w + dx

    return (
        arr[y0a:y1a, x0a:x1a],
        arr[y0b:y1b, x0b:x1b],
    )


def internal_edge_transition_density(mask: np.ndarray) -> float:
    hdiff = np.count_nonzero(mask[:, 1:] != mask[:, :-1])
    vdiff = np.count_nonzero(mask[1:, :] != mask[:-1, :])
    denom = (
        mask.shape[0] * (mask.shape[1] - 1)
        + (mask.shape[0] - 1) * mask.shape[1]
    )
    return float((hdiff + vdiff) / denom) if denom else np.nan


def morphology_metrics(mask: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    n_pixels = int(mask.size)
    event_pixels = int(mask.sum())

    out: dict[str, Any] = {
        "event_pixels": event_pixels,
        "event_fraction": event_pixels / n_pixels,
        "component_count": 0,
        "largest_component_pixels": 0,
        "largest_component_fraction_of_event": np.nan,
        "internal_edge_transition_density": internal_edge_transition_density(mask),
        "touches_border": False,
        "largest_component_touches_border": False,
        "largest_component_bbox_fraction": np.nan,
        "largest_component_elongation": np.nan,
    }

    if event_pixels == 0:
        return out

    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    out["touches_border"] = bool(np.any(mask & border))

    structure = np.array(
        [[0, 1, 0],
         [1, 1, 1],
         [0, 1, 0]],
        dtype=np.uint8,
    )
    labels, ncomp = ndimage.label(mask, structure=structure)
    out["component_count"] = int(ncomp)

    if ncomp == 0:
        return out

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest_label = int(np.argmax(sizes))
    largest_size = int(sizes[largest_label])

    out["largest_component_pixels"] = largest_size
    out["largest_component_fraction_of_event"] = (
        largest_size / event_pixels
    )

    largest_mask = labels == largest_label
    out["largest_component_touches_border"] = bool(
        np.any(largest_mask & border)
    )

    coords = np.argwhere(largest_mask)
    if len(coords):
        rmin, cmin = coords.min(axis=0)
        rmax, cmax = coords.max(axis=0)
        bbox_area = int((rmax - rmin + 1) * (cmax - cmin + 1))
        out["largest_component_bbox_fraction"] = (
            largest_size / bbox_area
            if bbox_area > 0
            else np.nan
        )

    if len(coords) >= 3:
        centered = coords.astype(np.float64) - coords.mean(axis=0)
        cov = (centered.T @ centered) / max(len(coords) - 1, 1)
        eig = np.linalg.eigvalsh(cov)
        eig = np.sort(np.maximum(eig, 0.0))
        if eig[-1] > 0:
            if eig[0] > 1e-12:
                out["largest_component_elongation"] = float(
                    math.sqrt(eig[-1] / eig[0])
                )
            else:
                out["largest_component_elongation"] = np.inf

    return out


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = values.to_numpy(dtype=float)
    w = weights.to_numpy(dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(mask):
        return np.nan
    return float(np.average(v[mask], weights=w[mask]))


def weighted_quantile(
    values: pd.Series,
    weights: pd.Series,
    q: float,
) -> float:
    v = values.to_numpy(dtype=float)
    w = weights.to_numpy(dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(mask):
        return np.nan

    v = v[mask]
    w = w[mask]
    order = np.argsort(v)
    v = v[order]
    w = w[order]

    cdf = np.cumsum(w)
    cutoff = q * cdf[-1]
    idx = int(np.searchsorted(cdf, cutoff, side="left"))
    idx = min(idx, len(v) - 1)
    return float(v[idx])


def summarize_morphology_group(
    g: pd.DataFrame,
    extra: dict[str, Any],
) -> dict[str, Any]:
    w = g["sampling_weight"]
    event = g[g["event_pixels"] > 0]
    we = event["sampling_weight"]

    row: dict[str, Any] = {
        **extra,
        "sample_rows": int(len(g)),
        "weighted_patch_mass": float(w.sum()),
        "patch_any_event_rate": weighted_mean(
            (g["event_pixels"] > 0).astype(float),
            w,
        ),
        "mean_event_fraction_all_patches": weighted_mean(
            g["event_fraction"],
            w,
        ),
        "event_fraction_p50_all_patches": weighted_quantile(
            g["event_fraction"],
            w,
            0.50,
        ),
        "event_fraction_p90_all_patches": weighted_quantile(
            g["event_fraction"],
            w,
            0.90,
        ),
        "event_fraction_p99_all_patches": weighted_quantile(
            g["event_fraction"],
            w,
            0.99,
        ),
        "event_patch_sample_rows": int(len(event)),
        "weighted_event_patch_mass": float(we.sum()) if len(event) else 0.0,
        "mean_event_fraction_given_event": weighted_mean(
            event["event_fraction"],
            we,
        ) if len(event) else np.nan,
        "mean_component_count_given_event": weighted_mean(
            event["component_count"],
            we,
        ) if len(event) else np.nan,
        "mean_largest_component_fraction_of_event": weighted_mean(
            event["largest_component_fraction_of_event"],
            we,
        ) if len(event) else np.nan,
        "mean_internal_edge_transition_density_given_event": weighted_mean(
            event["internal_edge_transition_density"],
            we,
        ) if len(event) else np.nan,
        "border_touch_rate_given_event": weighted_mean(
            event["touches_border"].astype(float),
            we,
        ) if len(event) else np.nan,
        "largest_component_border_touch_rate_given_event": weighted_mean(
            event["largest_component_touches_border"].astype(float),
            we,
        ) if len(event) else np.nan,
        "mean_largest_component_bbox_fill": weighted_mean(
            event["largest_component_bbox_fraction"],
            we,
        ) if len(event) else np.nan,
    }
    return row


def exact_patch_event_rates(
    strata_counts: pd.DataFrame,
) -> dict[str, float]:
    counts = {
        int(r["stratum_id"]): int(r["patch_count"])
        for _, r in strata_counts.iterrows()
    }
    total = sum(counts.values())

    def rate(ids: list[int]) -> float:
        return (
            sum(counts.get(i, 0) for i in ids) / total
            if total
            else np.nan
        )

    return {
        "gt_0": rate([1, 2, 3, 4, 5]),
        "ge_20": rate([2, 3, 4, 5]),
        "ge_30": rate([3, 4, 5]),
        "ge_40": rate([4, 5]),
        "ge_45": rate([5]),
    }


def main() -> None:
    args = parse_args()
    steps = parse_steps(args.spatial_steps)

    setup_output(args.output_dir, args.overwrite)
    logger = setup_logger(args.output_dir)

    zarr_path = args.dataset_dir / "train.zarr"
    if not zarr_path.exists():
        raise FileNotFoundError(f"train.zarr not found: {zarr_path}")

    root = open_zarr_group(zarr_path)
    target = root["target"]
    timestamps_raw = np.asarray(root["timestamps"][:])

    if int(target.shape[0]) != len(timestamps_raw):
        raise RuntimeError(
            "target and timestamps have inconsistent first dimension."
        )

    patch_shape = tuple(
        int(x)
        for x in np.squeeze(np.empty(target.shape[1:])).shape
    )
    if len(patch_shape) != 2:
        # More reliable runtime check from first item.
        patch_shape = get_target_2d(
            np.asarray(target[0], dtype=np.float32)
        ).shape

    if any(step >= min(patch_shape) for step in steps):
        raise ValueError(
            f"At least one spatial step is too large for patch shape "
            f"{patch_shape}: {steps}"
        )

    logger.info("=" * 80)
    logger.info("CORRDIFF - PHASE 9 - RADAR SPATIAL STRUCTURE")
    logger.info("Dataset            : %s", args.dataset_dir)
    logger.info("Target shape       : %s", tuple(target.shape))
    logger.info("Patch shape        : %s", patch_shape)
    logger.info("Spatial steps px   : %s", steps)
    logger.info("Radar resolution   : %.3f km/pixel", args.radar_resolution_km)
    logger.info("Sample/stratum     : %d", args.sample_per_stratum)
    logger.info("=" * 80)

    # ------------------------------------------------------------------
    # PASS A - exact strata over all patches
    # ------------------------------------------------------------------
    strata, strata_counts = classify_patch_strata(
        target,
        logger,
    )
    strata_counts.to_parquet(
        args.output_dir / "patch_strata_counts.parquet",
        index=False,
    )

    # Patch-level time metadata.
    time_meta = build_patch_time_metadata(
        timestamps_raw,
        args.local_timezone,
    )

    # ------------------------------------------------------------------
    # Stratified patch sample
    # ------------------------------------------------------------------
    manifest = build_sample_manifest(
        strata=strata,
        timestamps_raw=timestamps_raw,
        time_meta=time_meta,
        sample_per_stratum=args.sample_per_stratum,
        seed=args.seed,
    )
    manifest.to_parquet(
        args.output_dir / "sampled_patch_manifest.parquet",
        index=False,
    )

    logger.info("Sampled patches     : %d", len(manifest))

    # Spatial accumulators.
    fields = ["target_log1p", *EVENT_THRESHOLDS.keys()]
    accum: dict[tuple[str, str, int], PairAccumulator] = {}

    for field in fields:
        for direction in DIRECTIONS:
            for step in steps:
                accum[(field, direction, step)] = PairAccumulator()

    morphology_rows: list[dict[str, Any]] = []

    # Chunk-aware sample read.
    chunk0 = (
        int(target.chunks[0])
        if getattr(target, "chunks", None)
        else 256
    )

    manifest = manifest.sort_values("patch_index").reset_index(drop=True)
    patch_indices = manifest["patch_index"].to_numpy(dtype=np.int64)
    chunk_ids = patch_indices // chunk0

    unique_chunk_ids = np.unique(chunk_ids)

    logger.info("=" * 80)
    logger.info("PHASE 9 - PASS B - SPATIAL + MORPHOLOGY")
    logger.info("Input chunks with sample: %d", len(unique_chunk_ids))
    logger.info("=" * 80)

    processed = 0

    for ci, chunk_id in enumerate(unique_chunk_ids, start=1):
        sel = np.flatnonzero(chunk_ids == chunk_id)
        chunk_start = int(chunk_id * chunk0)
        chunk_end = min(int(target.shape[0]), chunk_start + chunk0)

        block = np.asarray(
            target[chunk_start:chunk_end],
            dtype=np.float32,
        )

        for manifest_pos in sel:
            rec = manifest.iloc[int(manifest_pos)]
            patch_index = int(rec["patch_index"])
            local_idx = patch_index - chunk_start
            stored = get_target_2d(block[local_idx])

            patch_weight = float(rec["sampling_weight"])
            season = str(rec["season_code"])
            timestamp_utc = rec["timestamp_utc"]

            masks = {
                event_id: event_mask(stored, event_id)
                for event_id in EVENT_THRESHOLDS
            }

            # Spatial dependence on stored log1p target and binary masks.
            spatial_fields = {
                "target_log1p": stored,
                **{
                    event_id: mask.astype(np.float32)
                    for event_id, mask in masks.items()
                },
            }

            for direction, (dy0, dx0) in DIRECTIONS.items():
                for step in steps:
                    dy = dy0 * step
                    dx = dx0 * step

                    for field, arr in spatial_fields.items():
                        x, y = shifted_pair(arr, dy, dx)
                        accum[(field, direction, step)].update(
                            x,
                            y,
                            patch_weight,
                        )

            # Morphology per event threshold.
            for event_id, dbz in EVENT_THRESHOLDS.items():
                m = morphology_metrics(masks[event_id])
                morphology_rows.append({
                    "patch_index": patch_index,
                    "timestamp_utc": timestamp_utc,
                    "season_code": season,
                    "year_utc": int(rec["year_utc"]),
                    "month_local": int(rec["month_local"]),
                    "hour_local": int(rec["hour_local"]),
                    "stratum_id": int(rec["stratum_id"]),
                    "stratum": rec["stratum"],
                    "sampling_weight": patch_weight,
                    "event_id": event_id,
                    "threshold_dbz": dbz,
                    **m,
                })

            processed += 1

        if (
            ci == len(unique_chunk_ids)
            or ci % 200 == 0
        ):
            logger.info(
                "PASS B: chunks %d/%d | sampled patches %d/%d",
                ci,
                len(unique_chunk_ids),
                processed,
                len(manifest),
            )

    morphology = pd.DataFrame(morphology_rows)
    morphology.to_parquet(
        args.output_dir / "patch_morphology_sample.parquet",
        index=False,
    )

    # ------------------------------------------------------------------
    # Spatial lag table
    # ------------------------------------------------------------------
    spatial_rows = []

    for (field, direction, step), ac in accum.items():
        dy0, dx0 = DIRECTIONS[direction]
        dy = dy0 * step
        dx = dx0 * step
        stats = ac.finalize()

        event_threshold = (
            EVENT_THRESHOLDS[field]
            if field in EVENT_THRESHOLDS
            else np.nan
        )

        row = {
            "field": field,
            "threshold_dbz": event_threshold,
            "direction": direction,
            "step_pixels": step,
            "dy_pixels": dy,
            "dx_pixels": dx,
            "distance_km": float(
                math.sqrt(dy * dy + dx * dx)
                * args.radar_resolution_km
            ),
            **stats,
        }

        # For binary fields, semivariance = half mismatch probability.
        if field in EVENT_THRESHOLDS:
            row["pair_mismatch_probability"] = (
                2.0 * row["semivariance"]
                if np.isfinite(row["semivariance"])
                else np.nan
            )
        else:
            row["pair_mismatch_probability"] = np.nan

        spatial_rows.append(row)

    spatial_df = pd.DataFrame(spatial_rows).sort_values(
        ["field", "direction", "step_pixels"]
    )
    spatial_df.to_parquet(
        args.output_dir / "spatial_lag_statistics.parquet",
        index=False,
    )

    # ------------------------------------------------------------------
    # Morphology summaries
    # ------------------------------------------------------------------
    summary_rows = []
    for event_id, g in morphology.groupby(
        "event_id",
        sort=False,
    ):
        summary_rows.append(
            summarize_morphology_group(
                g,
                {
                    "event_id": event_id,
                    "threshold_dbz": EVENT_THRESHOLDS[event_id],
                },
            )
        )

    morphology_summary = pd.DataFrame(summary_rows)
    morphology_summary.to_parquet(
        args.output_dir / "morphology_summary.parquet",
        index=False,
    )

    season_rows = []
    for (season, event_id), g in morphology.groupby(
        ["season_code", "event_id"],
        observed=True,
        sort=False,
    ):
        season_rows.append(
            summarize_morphology_group(
                g,
                {
                    "season_code": str(season),
                    "event_id": event_id,
                    "threshold_dbz": EVENT_THRESHOLDS[event_id],
                },
            )
        )

    morphology_by_season = pd.DataFrame(season_rows)
    if not morphology_by_season.empty:
        morphology_by_season["season_code"] = pd.Categorical(
            morphology_by_season["season_code"],
            categories=SEASON_ORDER,
            ordered=True,
        )
        morphology_by_season = morphology_by_season.sort_values(
            ["season_code", "threshold_dbz"]
        )

    morphology_by_season.to_parquet(
        args.output_dir / "morphology_by_season.parquet",
        index=False,
    )

    # ------------------------------------------------------------------
    # Summary / QC
    # ------------------------------------------------------------------
    exact_rates = exact_patch_event_rates(strata_counts)

    warnings: list[str] = []

    high_border = morphology_summary[
        morphology_summary["event_id"].isin(["ge_30", "ge_40", "ge_45"])
        & (
            morphology_summary["border_touch_rate_given_event"] > 0.5
        )
    ]
    if not high_border.empty:
        warnings.append(
            "At least one intense-event threshold has >50% patch-border "
            "contact among event patches. Connected-component morphology is "
            "therefore strongly affected by patch truncation."
        )

    sample_ratio = len(manifest) / int(target.shape[0])
    if sample_ratio < 0.05:
        warnings.append(
            "Spatial/morphology statistics use a stratified sample containing "
            "<5% of all patches; inverse stratum weights are used for global "
            "summaries."
        )

    summary = {
        "phase_version": PHASE_VERSION,
        "dataset_dir": str(args.dataset_dir),
        "target_shape": [int(x) for x in target.shape],
        "patch_shape": [int(x) for x in patch_shape],
        "total_patches": int(target.shape[0]),
        "sampled_patches": int(len(manifest)),
        "sample_ratio": float(sample_ratio),
        "sample_per_stratum": int(args.sample_per_stratum),
        "seed": int(args.seed),
        "radar_resolution_km": float(args.radar_resolution_km),
        "spatial_steps_pixels": steps,
        "directions": DIRECTIONS,
        "local_timezone": args.local_timezone,
        "radar_unit": "dBZ",
        "stored_target": "log1p(clip(dBZ, min=0)) float32",
        "threshold_rule": (
            "thresholds are applied directly in stored float32 log1p domain"
        ),
        "stratification": {
            str(int(row["stratum_id"])): {
                "name": row["stratum"],
                "patch_count": int(row["patch_count"]),
                "patch_ratio": float(row["patch_ratio"]),
            }
            for _, row in strata_counts.iterrows()
        },
        "exact_patch_any_event_rates": exact_rates,
        "scope": (
            "overlapping 32x32 training-patch spatial distribution; "
            "not de-duplicated full-field radar climatology"
        ),
        "component_connectivity": "4-neighbor",
        "warnings": warnings,
        "methodological_notes": [
            (
                "Spatial correlations and semivariograms are computed within "
                "stored patches only; no cross-patch reconstruction is used."
            ),
            (
                "Inverse stratum-sampling weights recover patch-distribution "
                "summaries, but spatial pairs can still be duplicated because "
                "the original patches overlap."
            ),
            (
                "Connected components touching patch borders may continue "
                "outside the patch; component count and size are therefore "
                "patch-truncated diagnostics."
            ),
            (
                "The continuous spatial variogram is calculated on the stored "
                "log1p target, not directly on linear dBZ."
            ),
            (
                "Binary spatial semivariance equals half the mismatch "
                "probability between pixels at the specified displacement."
            ),
        ],
    }

    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("=" * 80)
    logger.info("PHASE 9 COMPLETE")
    logger.info("Output: %s", args.output_dir)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
