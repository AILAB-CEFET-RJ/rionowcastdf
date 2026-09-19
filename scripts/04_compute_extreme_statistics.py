#!/usr/bin/env python3
"""
CorrDiff — Fase 4 v3
Eventos intensos/extremos com amostragem estratificada GLOBAL
==============================================================

Motivação
---------
A v2 corrigiu o desbalanceamento dentro de uma subamostra de blocos, mas os
blocos selecionados ainda podiam sub-representar episódios intensos. A v3
remove essa limitação:

PASSO A
    Varre TODO o target + mask do train.zarr.
    Conta exatamente os estratos globais e mantém um reservoir GLOBAL de
    endereços de pixels para cada estrato de intensidade.

PASSO B
    Agrupa os endereços selecionados por chunk/bloco do input.
    Lê apenas os chunks do ERA5 necessários para recuperar os 20 preditores
    nos pixels selecionados.

Os pesos passam a ser globais:

    peso_estrato = N_global_estrato / n_reservoir_estrato

Assim, as estatísticas ponderadas representam a distribuição em patches do
dataset inteiro, não apenas uma subamostra temporal de blocos.

IMPORTANTE
----------
- "Global" aqui significa a distribuição armazenada em patches, com overlap.
  Não é climatologia espacial de pixels físicos deduplicados.
- O radar é tratado como "valor numérico da legenda" após expm1(target).
  A unidade física não é assumida como dBZ.
- effective_n_event corrige apenas a desigualdade dos pesos. Não corrige
  dependência espacial, overlap de patches nem autocorrelação temporal.
- >=50 continua sendo ultra-raro; interprete separadamente.

Estratos disjuntos
------------------
zero
(0, 20)
[20, 25)
[25, 30)
[30, 35)
[35, 40)
[40, 45)
[45, 50)
>= 50

Saídas
------
analysis_outputs/04_extremes/
├── analysis_summary.json
├── predictor_catalog.parquet
├── global_scan_blocks.parquet
├── stratum_sampling.parquet
├── input_blocks_read.parquet
├── threshold_definitions.parquet
├── event_prevalence.parquet
├── conditional_predictor_stats.parquet
├── effect_sizes.parquet
├── event_rate_by_predictor_decile.parquet
├── extreme_samples.npz
└── phase4.log
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


PHASE_VERSION = "phase4-extremes-v3-global-stratified-weighted"

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

STRATA = [
    "zero",
    "gt0_lt20",
    "ge20_lt25",
    "ge25_lt30",
    "ge30_lt35",
    "ge35_lt40",
    "ge40_lt45",
    "ge45_lt50",
    "ge50",
]

FIXED_THRESHOLDS = [20, 25, 30, 35, 40, 45, 50]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "CorrDiff Fase 4 v3: full target/mask scan + global stratified "
            "reservoir + weighted extreme-event analysis."
        )
    )
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument(
        "--phase0-dir",
        type=Path,
        default=Path("analysis_outputs/00_quality"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs/04_extremes"),
    )
    p.add_argument(
        "--scan-block-size",
        type=int,
        default=0,
        help=(
            "Patches por bloco no PASSO A. 0 usa o primeiro chunk do target."
        ),
    )
    p.add_argument(
        "--stratum-size",
        type=int,
        default=20000,
        help="Capacidade do reservoir global para cada estrato não-zero.",
    )
    p.add_argument(
        "--zero-stratum-size",
        type=int,
        default=20000,
        help=(
            "Capacidade do reservoir global do estrato zero. Reduza para "
            "diminuir I/O no PASSO B se necessário."
        ),
    )
    p.add_argument("--deciles", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260918)
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
    logger = logging.getLogger("corrdiff.phase4.v3")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
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



def safe_hypergeometric(
    rng: np.random.Generator,
    ngood: int,
    nbad: int,
    nsample: int,
) -> int:
    """
    Exact hypergeometric draw that avoids NumPy's <1e9 ngood/nbad limit.

    Uses NumPy's fast implementation while both population components are
    below the library limit. For larger populations, performs the at-most
    `nsample` draws sequentially without replacement. In this Phase 4
    reservoir, nsample is the reservoir capacity (typically 20,000), so the
    fallback is only triggered late in the global zero-stratum scan and is
    still inexpensive.
    """
    ngood = int(ngood)
    nbad = int(nbad)
    nsample = int(nsample)

    if nsample <= 0 or ngood <= 0:
        return 0
    if nbad <= 0:
        return min(nsample, ngood)
    if nsample > ngood + nbad:
        raise ValueError(
            f"nsample={nsample} exceeds population={ngood + nbad}"
        )

    # NumPy Generator.hypergeometric rejects either component >= 1e9.
    if ngood < 1_000_000_000 and nbad < 1_000_000_000:
        return int(
            rng.hypergeometric(
                ngood=ngood,
                nbad=nbad,
                nsample=nsample,
            )
        )

    # Exact sequential sampling without replacement.
    good = ngood
    bad = nbad
    selected_good = 0

    for draw_index in range(nsample):
        remaining_draws = nsample - draw_index

        if good <= 0:
            break
        if bad <= 0:
            selected_good += remaining_draws
            break

        if rng.random() < (good / (good + bad)):
            selected_good += 1
            good -= 1
        else:
            bad -= 1

    return selected_good


@dataclass
class AddressReservoir:
    """Uniform reservoir over a stream, updated in vectorized batches."""

    capacity: int
    rng: np.random.Generator
    total_seen: int = 0
    addresses: np.ndarray | None = None
    radar: np.ndarray | None = None

    def update(
        self,
        candidate_addresses: np.ndarray,
        candidate_radar: np.ndarray,
    ) -> None:
        m = int(candidate_addresses.size)
        if m == 0:
            return

        old_total = int(self.total_seen)
        new_total = old_total + m
        target_size = min(int(self.capacity), new_total)

        if old_total == 0:
            k_new = target_size
            old_keep = np.empty(0, dtype=np.int64)
        else:
            # Exact batch reservoir update:
            # among target_size samples from old_total + m stream elements,
            # the number coming from this new batch is hypergeometric.
            k_new = safe_hypergeometric(
                rng=self.rng,
                ngood=m,
                nbad=old_total,
                nsample=target_size,
            )
            old_needed = target_size - k_new
            old_size = (
                0 if self.addresses is None else int(self.addresses.size)
            )
            if old_needed > old_size:
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
            chosen = self.rng.choice(
                m,
                size=k_new,
                replace=False,
            )
            new_addr = candidate_addresses[chosen].astype(
                np.int64, copy=False
            )
            new_radar = candidate_radar[chosen].astype(
                np.float32, copy=False
            )
        else:
            new_addr = np.empty(0, dtype=np.int64)
            new_radar = np.empty(0, dtype=np.float32)

        if self.addresses is not None and old_keep.size:
            self.addresses = np.concatenate(
                [self.addresses[old_keep], new_addr]
            )
            self.radar = np.concatenate(
                [self.radar[old_keep], new_radar]
            )
        else:
            self.addresses = new_addr
            self.radar = new_radar

        self.total_seen = new_total


def derive_selected(
    raw: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
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

    sw = float(ww.sum())
    sw2 = float(np.square(ww).sum())
    denom = sw - sw2 / sw
    if denom <= 0:
        return float("nan")

    return float(np.sum(ww * np.square(xx - mu)) / denom)


def weighted_std(x: np.ndarray, w: np.ndarray) -> float:
    value = weighted_var(x, w)
    if not np.isfinite(value) or value < 0:
        return float("nan")
    return math.sqrt(value)


def weighted_quantile(
    values: np.ndarray,
    quantiles: np.ndarray | list[float] | float,
    weights: np.ndarray,
) -> np.ndarray:
    q = np.atleast_1d(np.asarray(quantiles, dtype=np.float64))
    mask = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
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
    targets = np.clip(q, 0.0, 1.0) * total
    idx = np.searchsorted(cumulative, targets, side="left")
    idx = np.clip(idx, 0, len(v) - 1)
    return v[idx]


def effective_sample_size(w: np.ndarray) -> float:
    w = w[np.isfinite(w) & (w > 0)].astype(np.float64)
    if w.size == 0:
        return 0.0

    s1 = float(w.sum())
    s2 = float(np.square(w).sum())
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

    if {"threshold", "event_pixel_ratio"}.issubset(df.columns):
        for _, row in df.iterrows():
            out[str(row["threshold"])] = float(
                row["event_pixel_ratio"]
            )
    return out


def exact_fixed_count(
    stratum_counts: dict[str, int],
    threshold: int,
) -> int:
    order = {
        20: [
            "ge20_lt25", "ge25_lt30", "ge30_lt35",
            "ge35_lt40", "ge40_lt45", "ge45_lt50", "ge50",
        ],
        25: [
            "ge25_lt30", "ge30_lt35", "ge35_lt40",
            "ge40_lt45", "ge45_lt50", "ge50",
        ],
        30: [
            "ge30_lt35", "ge35_lt40",
            "ge40_lt45", "ge45_lt50", "ge50",
        ],
        35: [
            "ge35_lt40", "ge40_lt45", "ge45_lt50", "ge50",
        ],
        40: ["ge40_lt45", "ge45_lt50", "ge50"],
        45: ["ge45_lt50", "ge50"],
        50: ["ge50"],
    }
    return int(sum(stratum_counts[name] for name in order[threshold]))


def main() -> None:
    args = parse_args()
    prepare_output(args.output_dir, args.overwrite)
    logger = setup_logger(args.output_dir)
    rng = np.random.default_rng(args.seed)

    root = open_group(args.dataset_dir / "train.zarr")
    xds = root["input"]
    yds = root["target"]
    mds = root["mask"]

    n = int(yds.shape[0])
    height = int(yds.shape[-2])
    width = int(yds.shape[-1])
    pixels_per_patch = height * width

    channels = list(root.attrs.get("channels", []))
    if not channels:
        metadata = args.dataset_dir / "metadata.json"
        if metadata.exists():
            channels = list(
                json.loads(
                    metadata.read_text(encoding="utf-8")
                ).get("channels", [])
            )

    missing = [name for name in RAW_CHANNELS if name not in channels]
    if missing:
        raise RuntimeError(
            f"Required input channels are missing: {missing}"
        )

    channel_index = {
        name: channels.index(name) for name in channels
    }

    target_chunk0 = int(getattr(yds, "chunks", (256,))[0])
    input_chunk0 = int(getattr(xds, "chunks", (256,))[0])
    scan_block_size = args.scan_block_size or target_chunk0

    capacities = {
        name: (
            args.zero_stratum_size
            if name == "zero"
            else args.stratum_size
        )
        for name in STRATA
    }
    reservoirs = {
        name: AddressReservoir(
            capacity=capacities[name],
            rng=rng,
        )
        for name in STRATA
    }

    # ------------------------------------------------------------------
    # PASS A — global target/mask scan
    # ------------------------------------------------------------------
    total_scan_blocks = math.ceil(n / scan_block_size)
    logger.info("=" * 72)
    logger.info("PHASE 4 v3 — PASS A: GLOBAL TARGET/MASK SCAN")
    logger.info("Patches in Zarr       : %d", n)
    logger.info("Target scan block size: %d", scan_block_size)
    logger.info("Target scan blocks    : %d", total_scan_blocks)
    logger.info("=" * 72)

    valid_pixels_global = 0
    scan_rows: list[dict[str, Any]] = []

    for block_no, start in enumerate(
        range(0, n, scan_block_size),
        start=1,
    ):
        end = min(n, start + scan_block_size)

        stored = np.asarray(
            yds[start:end, 0],
            dtype=np.float32,
        )
        y = np.expm1(stored).astype(np.float32, copy=False)

        mask = (
            np.asarray(
                mds[start:end, 0],
                dtype=np.float32,
            )
            > 0.5
        )

        valid = mask & np.isfinite(y)
        flat_y = y.reshape(-1)
        flat_valid = valid.reshape(-1)

        valid_count = int(flat_valid.sum())
        valid_pixels_global += valid_count

        row: dict[str, Any] = {
            "scan_block": block_no - 1,
            "start_patch": start,
            "end_patch": end,
            "patches": end - start,
            "valid_pixels": valid_count,
        }

        for stratum_name in STRATA:
            local_ids = np.flatnonzero(
                flat_valid & stratum_mask(flat_y, stratum_name)
            )
            count = int(local_ids.size)
            row[f"count_{stratum_name}"] = count

            if count:
                local_patch = local_ids // pixels_per_patch
                pixel_flat = local_ids % pixels_per_patch

                global_patch = start + local_patch
                addresses = (
                    global_patch.astype(np.int64)
                    * pixels_per_patch
                    + pixel_flat.astype(np.int64)
                )

                reservoirs[stratum_name].update(
                    addresses,
                    flat_y[local_ids],
                )

        scan_rows.append(row)

        if block_no % 100 == 0 or block_no == total_scan_blocks:
            logger.info(
                "PASS A: %d/%d blocks scanned",
                block_no,
                total_scan_blocks,
            )

    scan_df = pd.DataFrame(scan_rows)
    scan_df.to_parquet(
        args.output_dir / "global_scan_blocks.parquet",
        index=False,
    )

    stratum_counts = {
        name: int(reservoirs[name].total_seen)
        for name in STRATA
    }

    if sum(stratum_counts.values()) != valid_pixels_global:
        raise RuntimeError(
            "Global stratum counts do not sum to valid pixel count: "
            f"{sum(stratum_counts.values())} != {valid_pixels_global}"
        )

    # Assemble sampled addresses.
    addr_parts: list[np.ndarray] = []
    radar_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    stratum_parts: list[np.ndarray] = []
    stratum_rows: list[dict[str, Any]] = []

    for name in STRATA:
        reservoir = reservoirs[name]
        sample_count = (
            0
            if reservoir.addresses is None
            else int(reservoir.addresses.size)
        )
        population_count = int(reservoir.total_seen)

        weight = (
            population_count / sample_count
            if sample_count > 0
            else np.nan
        )
        fraction = (
            sample_count / population_count
            if population_count > 0
            else np.nan
        )

        stratum_rows.append({
            "stratum": name,
            "global_population_count": population_count,
            "global_population_ratio": (
                population_count / valid_pixels_global
                if valid_pixels_global > 0
                else np.nan
            ),
            "sample_count": sample_count,
            "sampling_fraction_global": fraction,
            "inverse_sampling_weight_global": weight,
        })

        if sample_count:
            addr_parts.append(
                reservoir.addresses.astype(np.int64, copy=False)
            )
            radar_parts.append(
                reservoir.radar.astype(np.float32, copy=False)
            )
            weight_parts.append(
                np.full(
                    sample_count,
                    weight,
                    dtype=np.float64,
                )
            )
            stratum_parts.append(
                np.full(
                    sample_count,
                    name,
                    dtype="U32",
                )
            )

    stratum_df = pd.DataFrame(stratum_rows)
    stratum_df.to_parquet(
        args.output_dir / "stratum_sampling.parquet",
        index=False,
    )

    if not addr_parts:
        raise RuntimeError("No pixels were retained by global reservoirs")

    addresses = np.concatenate(addr_parts)
    Y = np.concatenate(radar_parts)
    W = np.concatenate(weight_parts)
    S = np.concatenate(stratum_parts)

    # ------------------------------------------------------------------
    # PASS B — recover predictors at sampled global addresses
    # ------------------------------------------------------------------
    logger.info("=" * 72)
    logger.info("PHASE 4 v3 — PASS B: INPUT EXTRACTION")
    logger.info("Retained global sample: %d pixels", len(addresses))
    logger.info("=" * 72)

    patch_index = addresses // pixels_per_patch
    pixel_flat = addresses % pixels_per_patch
    row_index = pixel_flat // width
    col_index = pixel_flat % width

    input_block_id = patch_index // input_chunk0
    unique_blocks = np.unique(input_block_id)

    X = np.full(
        (len(addresses), len(ALL_PREDICTORS)),
        np.nan,
        dtype=np.float32,
    )

    input_block_rows: list[dict[str, Any]] = []

    for pos, block_id in enumerate(unique_blocks, start=1):
        selected = np.flatnonzero(input_block_id == block_id)

        start = int(block_id * input_chunk0)
        end = min(n, start + input_chunk0)
        x_block = np.asarray(
            xds[start:end],
            dtype=np.float32,
        )

        lp = (patch_index[selected] - start).astype(np.int64)
        rr = row_index[selected].astype(np.int64)
        cc = col_index[selected].astype(np.int64)

        raw: dict[str, np.ndarray] = {}
        for name in RAW_CHANNELS:
            raw[name] = x_block[
                lp,
                channel_index[name],
                rr,
                cc,
            ].astype(np.float32, copy=False)

        derived = derive_selected(raw)

        for j, name in enumerate(ALL_PREDICTORS):
            values = raw[name] if name in raw else derived[name]
            X[selected, j] = values

        input_block_rows.append({
            "input_block_id": int(block_id),
            "start_patch": start,
            "end_patch": end,
            "selected_pixels": int(selected.size),
        })

        if pos % 100 == 0 or pos == len(unique_blocks):
            logger.info(
                "PASS B: %d/%d input chunks read",
                pos,
                len(unique_blocks),
            )

    input_blocks_df = pd.DataFrame(input_block_rows)
    input_blocks_df.to_parquet(
        args.output_dir / "input_blocks_read.parquet",
        index=False,
    )

    finite_rows = np.isfinite(X).all(axis=1)
    if not np.all(finite_rows):
        bad = int((~finite_rows).sum())
        raise RuntimeError(
            f"Selected global sample contains {bad} rows with non-finite "
            "input/derived predictors. Dataset was expected to be fully finite."
        )

    catalog = pd.DataFrame({
        "predictor": ALL_PREDICTORS,
        "source": [
            "raw" if name in RAW_CHANNELS else "derived"
            for name in ALL_PREDICTORS
        ],
        "unit": [UNITS[name] for name in ALL_PREDICTORS],
    })
    catalog.to_parquet(
        args.output_dir / "predictor_catalog.parquet",
        index=False,
    )

    # ------------------------------------------------------------------
    # Threshold definitions
    # ------------------------------------------------------------------
    global_q = weighted_quantile(
        Y,
        [0.90, 0.95, 0.99],
        W,
    )

    positive = Y > 0
    positive_q = weighted_quantile(
        Y[positive],
        [0.90, 0.95, 0.99],
        W[positive],
    )

    thresholds: list[dict[str, Any]] = []

    for value in FIXED_THRESHOLDS:
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
            "label": (
                f"Radar > P{int(q * 100)} global "
                "(global stratified estimate)"
            ),
        })

    for q, value in zip([0.90, 0.95, 0.99], positive_q):
        thresholds.append({
            "event_id": f"positive_p{int(q * 100)}",
            "family": "positive_quantile",
            "quantile": q,
            "threshold": float(value),
            "operator": ">",
            "label": (
                f"Radar > P{int(q * 100)} positive-only "
                "(global stratified estimate)"
            ),
        })

    threshold_df = pd.DataFrame(thresholds)
    threshold_df.to_parquet(
        args.output_dir / "threshold_definitions.parquet",
        index=False,
    )

    phase0_reference = load_phase0_reference(args.phase0_dir)

    # ------------------------------------------------------------------
    # Weighted analyses
    # ------------------------------------------------------------------
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

        weighted_event_count = float(W[event].sum())
        weighted_non_event_count = float(W[~event].sum())
        weighted_event_rate = (
            weighted_event_count / total_weight
            if total_weight > 0
            else np.nan
        )

        exact_count_v3 = np.nan
        exact_rate_v3 = np.nan
        phase0_rate = np.nan
        phase0_abs_diff = np.nan

        if th["family"] == "fixed":
            threshold_int = int(th["threshold"])
            exact_count_v3 = exact_fixed_count(
                stratum_counts,
                threshold_int,
            )
            exact_rate_v3 = (
                exact_count_v3 / valid_pixels_global
                if valid_pixels_global > 0
                else np.nan
            )
            phase0_rate = phase0_reference.get(
                f"ge_{threshold_int}",
                np.nan,
            )
            if np.isfinite(phase0_rate):
                phase0_abs_diff = abs(exact_rate_v3 - phase0_rate)

        reported_rate = (
            exact_rate_v3
            if np.isfinite(exact_rate_v3)
            else weighted_event_rate
        )
        rate_source = (
            "phase4_v3_exact_global_scan"
            if np.isfinite(exact_rate_v3)
            else "phase4_v3_global_stratified_weighted"
        )

        prevalence_rows.append({
            **th,
            "sample_n": int(len(Y)),
            "sample_event_count_unweighted": int(event.sum()),
            "sample_non_event_count_unweighted": int((~event).sum()),
            "effective_sample_size_all": effective_sample_size(W),
            "effective_sample_size_event": effective_sample_size(W[event]),
            "weighted_event_count_global_estimate": weighted_event_count,
            "weighted_event_rate_global_estimate": weighted_event_rate,
            "exact_global_event_count_v3": exact_count_v3,
            "exact_global_event_rate_v3": exact_rate_v3,
            "reference_event_rate_phase0": phase0_rate,
            "abs_rate_diff_v3_vs_phase0": phase0_abs_diff,
            "event_rate": reported_rate,
            "event_rate_source": rate_source,
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
                    xv,
                    [0.25, 0.50, 0.75, 0.95],
                    wv,
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

            a = values[ev]
            wa = W[ev]
            b = values[ne]
            wb = W[ne]

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
                "unit": UNITS[name],
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
                "weight_scope": "full_dataset_global_strata",
            })

            # Predictor deciles are estimated from the same globally weighted
            # stratified sample.
            x_all = values[finite]
            w_all = W[finite]
            event_all = event[finite]

            edges = weighted_quantile(
                x_all,
                np.linspace(0.0, 1.0, args.deciles + 1),
                w_all,
            )
            edges = np.unique(
                edges[np.isfinite(edges)]
            )

            if edges.size < 2:
                continue

            codes = np.searchsorted(
                edges[1:-1],
                x_all,
                side="right",
            )
            n_bins = (
                int(codes.max()) + 1
                if codes.size
                else 0
            )

            for decile in range(n_bins):
                selector = codes == decile
                if not np.any(selector):
                    continue

                ww = w_all[selector]
                xx = x_all[selector]
                ee = event_all[selector]

                denominator = float(ww.sum())
                rate = (
                    float(ww[ee].sum() / denominator)
                    if denominator > 0
                    else np.nan
                )

                decile_rows.append({
                    "event_id": th["event_id"],
                    "predictor": name,
                    "decile": decile + 1,
                    "sample_n": int(selector.sum()),
                    "effective_sample_size": effective_sample_size(ww),
                    "estimated_population_weight": denominator,
                    "predictor_mean": weighted_mean(xx, ww),
                    "predictor_min": float(np.min(xx)),
                    "predictor_max": float(np.max(xx)),
                    "event_rate": rate,
                    "weighted": True,
                    "weight_scope": "full_dataset_global_strata",
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
        global_address=addresses.astype(np.int64),
        stratum=S,
        predictor_names=np.asarray(
            ALL_PREDICTORS,
            dtype="U64",
        ),
    )

    fixed = prevalence_df[
        prevalence_df["family"] == "fixed"
    ].copy()

    max_abs_v3_vs_phase0 = np.nan
    if fixed["abs_rate_diff_v3_vs_phase0"].notna().any():
        max_abs_v3_vs_phase0 = float(
            fixed["abs_rate_diff_v3_vs_phase0"].max()
        )

    exact_positive_count = (
        valid_pixels_global - stratum_counts["zero"]
    )
    exact_positive_rate = (
        exact_positive_count / valid_pixels_global
        if valid_pixels_global > 0
        else np.nan
    )

    total_input_blocks = math.ceil(n / input_chunk0)

    summary = {
        "phase_version": PHASE_VERSION,
        "sample_source": "full_zarr_global_stratified_reservoir",
        "pass_a_scope": "all target/mask patches",
        "pass_a_scan_block_size": int(scan_block_size),
        "pass_a_blocks_scanned": int(total_scan_blocks),
        "global_patches_scanned": int(n),
        "global_valid_pixels_scanned": int(valid_pixels_global),
        "exact_global_positive_pixel_count": int(exact_positive_count),
        "exact_global_positive_pixel_rate": float(exact_positive_rate),
        "retained_global_stratified_sample_rows": int(len(Y)),
        "stratum_size": int(args.stratum_size),
        "zero_stratum_size": int(args.zero_stratum_size),
        "pass_b_input_chunk_size": int(input_chunk0),
        "pass_b_unique_input_blocks_read": int(len(unique_blocks)),
        "pass_b_total_input_blocks": int(total_input_blocks),
        "pass_b_input_block_fraction_read": float(
            len(unique_blocks) / total_input_blocks
        ),
        "predictor_count": int(len(ALL_PREDICTORS)),
        "threshold_count": int(len(thresholds)),
        "phase0_reference_available": bool(phase0_reference),
        "max_abs_fixed_threshold_rate_diff_v3_vs_phase0": (
            max_abs_v3_vs_phase0
        ),
        "weighting_method": (
            "global inverse stratum sampling fraction: "
            "global_population_count/sample_count"
        ),
        "radar_domain": (
            "expm1(stored_target): post-clip numerical radar-legend "
            "values; physical unit not assumed"
        ),
        "interpretation_note": (
            "PASS A scans the full target/mask Zarr. Fixed-threshold event "
            "prevalence is exact for the stored patch distribution. "
            "Conditional predictor statistics use a globally stratified, "
            "inverse-probability-weighted pixel sample. ESS reflects unequal "
            "weights only and does not remove spatial/temporal dependence."
        ),
    }

    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    logger.info("=" * 72)
    logger.info("PHASE 4 v3 COMPLETE")
    logger.info("Global valid pixels       : %d", valid_pixels_global)
    logger.info("Global retained sample    : %d", len(Y))
    logger.info(
        "Input blocks read          : %d / %d (%.2f%%)",
        len(unique_blocks),
        total_input_blocks,
        100.0 * len(unique_blocks) / total_input_blocks,
    )
    logger.info(
        "Max |v3 exact - Phase0|    : %s",
        max_abs_v3_vs_phase0,
    )
    logger.info("Output                    : %s", args.output_dir)
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
