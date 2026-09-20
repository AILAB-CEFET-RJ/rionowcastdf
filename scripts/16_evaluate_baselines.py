#!/usr/bin/env python3
"""
CorrDiff - Fase 16
Avaliação formal dos baselines
==============================

Baselines:
- zero
- climatology_month_hour_slot
- persistence_1h_radar_reference   [usa radar t-1h; referência observacional]
- pixel_mlp
- unet_deterministic
- unet_gaussian

Por padrão SOMENTE validation é permitida.
Para abrir test_primary/test_stress_ood é obrigatório --allow-locked-splits.

Métricas:
- RMSE / MAE / bias em log1p e dBZ;
- métricas categóricas >=20/30/40/45 dBZ;
- FSS em suportes nominais 2/4/8/16 km;
- métricas de máximo por patch;
- Gaussian NLL, CRPS, Brier probabilístico e ECE para unet_gaussian;
- diagnóstico por estação no nível de máximo/evento do patch.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:
    raise SystemExit("Phase 16 requires PyTorch.") from exc

from phase16_common import (
    CHANNELS,
    build_model,
    gaussian_crps,
    gaussian_nll,
    normal_exceedance_probability,
    open_group,
    zarr_take_first_axis,
)


PHASE_VERSION = "phase16-baselines-v1.1-zarr-oindex-frozen-split"

THRESHOLDS = [20.0, 30.0, 40.0, 45.0]
FSS_SUPPORT_PIXELS = [1, 2, 4, 8]
GRID_KM = 2.0
SEASONS = ["DJF", "MAM", "JJA", "SON"]

LEARNED_MODELS = [
    "pixel_mlp",
    "unet_deterministic",
    "unet_gaussian",
]

NONLEARNED = [
    "zero",
    "train_global_mean_field",
    "train_slot_mean",
    "climatology_month_hour_slot",
    "persistence_1h_radar_reference",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("datasets/corrdiff_2011_2024"),
    )
    p.add_argument(
        "--phase15-dir",
        type=Path,
        default=Path("analysis_outputs/15_formal_splits"),
    )
    p.add_argument(
        "--phase16-dir",
        type=Path,
        default=Path("analysis_outputs/16_baselines"),
    )
    p.add_argument(
        "--splits",
        default="validation",
        help="Comma-separated: validation,test_primary,test_stress_ood",
    )
    p.add_argument(
        "--baselines",
        default=(
            "zero,train_global_mean_field,train_slot_mean,"
            "climatology_month_hour_slot,"
            "persistence_1h_radar_reference,"
            "pixel_mlp,unet_deterministic,unet_gaussian"
        ),
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    p.add_argument("--amp", action="store_true")
    p.add_argument(
        "--allow-locked-splits",
        action="store_true",
        help=(
            "Required to evaluate test_primary or test_stress_ood. "
            "Use only after model/design freeze."
        ),
    )
    p.add_argument(
        "--with-persistence-common",
        action="store_true",
        help=(
            "Also evaluate all non-persistence baselines on the exact cohort "
            "where radar t-1h is available, enabling fair persistence comparison."
        ),
    )
    p.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="0 = full split; smoke-test only otherwise.",
    )
    p.add_argument("--overwrite-metrics", action="store_true")
    return p.parse_args()


def setup_logger(path: Path) -> logging.Logger:
    path.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("corrdiff.phase16.eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(path / "phase16_evaluate.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def event_metric_row(
    tp: float,
    fp: float,
    fn: float,
    tn: float,
) -> dict[str, float]:
    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp > 0 else np.nan
    pod = tp / (tp + fn) if tp + fn > 0 else np.nan
    far = fp / (tp + fp) if tp + fp > 0 else np.nan
    csi = tp / (tp + fp + fn) if tp + fp + fn > 0 else np.nan
    f1 = (
        2 * precision * pod / (precision + pod)
        if np.isfinite(precision)
        and np.isfinite(pod)
        and precision + pod > 0
        else np.nan
    )
    frequency_bias = (
        (tp + fp) / (tp + fn)
        if tp + fn > 0
        else np.nan
    )
    random_hits = (
        (tp + fp) * (tp + fn) / n
        if n > 0
        else np.nan
    )
    denom = tp + fp + fn - random_hits
    ets = (
        (tp - random_hits) / denom
        if np.isfinite(denom) and abs(denom) > 1e-12
        else np.nan
    )
    return {
        "precision": precision,
        "pod_recall": pod,
        "far": far,
        "csi": csi,
        "f1": f1,
        "frequency_bias": frequency_bias,
        "ets": ets,
    }


class Accumulator:
    def __init__(self):
        self.n_pixels = 0
        self.sum_sq_log = 0.0
        self.sum_abs_log = 0.0
        self.sum_bias_log = 0.0
        self.sum_sq_dbz = 0.0
        self.sum_abs_dbz = 0.0
        self.sum_bias_dbz = 0.0

        self.binary = {
            thr: dict(tp=0.0, fp=0.0, fn=0.0, tn=0.0, brier_sum=0.0)
            for thr in THRESHOLDS
        }
        self.fss = {
            (thr, k): dict(num=0.0, den=0.0, n=0)
            for thr in THRESHOLDS
            for k in FSS_SUPPORT_PIXELS
        }

        self.patch_rows = []
        self.gaussian_nll_sum = 0.0
        self.gaussian_crps_sum = 0.0
        self.gaussian_pixel_count = 0

        self.prob_bins = {
            thr: {
                "count": np.zeros(10, dtype=np.int64),
                "prob_sum": np.zeros(10, dtype=np.float64),
                "obs_sum": np.zeros(10, dtype=np.float64),
                "brier_sum": 0.0,
            }
            for thr in THRESHOLDS
        }

    def update(
        self,
        target_log: torch.Tensor,
        pred_log: torch.Tensor,
        patch_indices: np.ndarray,
        season_codes: np.ndarray,
        gaussian: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        y = target_log.float()
        p = pred_log.float().clamp_min(0.0)

        diff_log = p - y
        self.n_pixels += diff_log.numel()
        self.sum_sq_log += float(torch.square(diff_log).sum().cpu())
        self.sum_abs_log += float(diff_log.abs().sum().cpu())
        self.sum_bias_log += float(diff_log.sum().cpu())

        y_dbz = torch.expm1(y).clamp_min(0.0)
        p_dbz = torch.expm1(p).clamp_min(0.0)
        diff_dbz = p_dbz - y_dbz

        self.sum_sq_dbz += float(torch.square(diff_dbz).sum().cpu())
        self.sum_abs_dbz += float(diff_dbz.abs().sum().cpu())
        self.sum_bias_dbz += float(diff_dbz.sum().cpu())

        max_y = y_dbz.amax(dim=(1, 2, 3)).cpu().numpy()
        max_p = p_dbz.amax(dim=(1, 2, 3)).cpu().numpy()

        for idx, season, ty, py in zip(
            patch_indices,
            season_codes,
            max_y,
            max_p,
        ):
            self.patch_rows.append({
                "patch_index": int(idx),
                "season_code": str(season),
                "target_max_dbz": float(ty),
                "pred_max_dbz": float(py),
            })

        sigma = None
        mu = None
        if gaussian is not None:
            mu, log_sigma = gaussian
            sigma = torch.exp(log_sigma.float()).clamp_min(1e-6)
            nll = gaussian_nll(
                y,
                mu.float(),
                log_sigma.float(),
            )
            crps = gaussian_crps(
                y,
                mu.float(),
                sigma,
            )
            npx = y.numel()
            self.gaussian_nll_sum += float(nll.cpu()) * npx
            self.gaussian_crps_sum += float(crps.sum().cpu())
            self.gaussian_pixel_count += npx

        for thr in THRESHOLDS:
            yt = y_dbz >= thr
            pt = p_dbz >= thr

            tp = torch.logical_and(pt, yt).sum()
            fp = torch.logical_and(pt, ~yt).sum()
            fn = torch.logical_and(~pt, yt).sum()
            tn = torch.logical_and(~pt, ~yt).sum()

            b = self.binary[thr]
            b["tp"] += float(tp.cpu())
            b["fp"] += float(fp.cpu())
            b["fn"] += float(fn.cpu())
            b["tn"] += float(tn.cpu())
            b["brier_sum"] += float(
                torch.square(pt.float() - yt.float()).sum().cpu()
            )

            for k in FSS_SUPPORT_PIXELS:
                if k == 1:
                    ofrac = yt.float()
                    pfrac = pt.float()
                else:
                    ofrac = F.avg_pool2d(
                        yt.float(),
                        kernel_size=k,
                        stride=1,
                    )
                    pfrac = F.avg_pool2d(
                        pt.float(),
                        kernel_size=k,
                        stride=1,
                    )

                num = torch.square(pfrac - ofrac).sum()
                den = (torch.square(pfrac) + torch.square(ofrac)).sum()

                f = self.fss[(thr, k)]
                f["num"] += float(num.cpu())
                f["den"] += float(den.cpu())
                f["n"] += int(pfrac.numel())

            if gaussian is not None:
                threshold_log = float(np.log1p(thr))
                prob = normal_exceedance_probability(
                    threshold_log,
                    mu.float(),
                    sigma,
                )
                obs = yt.float()
                pr = self.prob_bins[thr]
                pr["brier_sum"] += float(
                    torch.square(prob - obs).sum().cpu()
                )

                prob_np = prob.detach().cpu().numpy().ravel()
                obs_np = obs.detach().cpu().numpy().ravel()
                bins = np.minimum(
                    (prob_np * 10).astype(np.int64),
                    9,
                )
                pr["count"] += np.bincount(
                    bins,
                    minlength=10,
                )
                pr["prob_sum"] += np.bincount(
                    bins,
                    weights=prob_np,
                    minlength=10,
                )
                pr["obs_sum"] += np.bincount(
                    bins,
                    weights=obs_np,
                    minlength=10,
                )

    def finalize(
        self,
        baseline: str,
        split: str,
        cohort: str,
    ) -> tuple[
        list[dict],
        list[dict],
        list[dict],
        list[dict],
        list[dict],
        list[dict],
    ]:
        point = []
        if self.n_pixels > 0:
            point.extend([
                {
                    "baseline": baseline,
                    "split": split,
                    "cohort": cohort,
                    "domain": "stored_log1p",
                    "rmse": math.sqrt(self.sum_sq_log / self.n_pixels),
                    "mae": self.sum_abs_log / self.n_pixels,
                    "bias": self.sum_bias_log / self.n_pixels,
                    "n_pixels": self.n_pixels,
                },
                {
                    "baseline": baseline,
                    "split": split,
                    "cohort": cohort,
                    "domain": "dbz_reconstructed",
                    "rmse": math.sqrt(self.sum_sq_dbz / self.n_pixels),
                    "mae": self.sum_abs_dbz / self.n_pixels,
                    "bias": self.sum_bias_dbz / self.n_pixels,
                    "n_pixels": self.n_pixels,
                },
            ])

        threshold_rows = []
        for thr, b in self.binary.items():
            n = b["tp"] + b["fp"] + b["fn"] + b["tn"]
            row = {
                "baseline": baseline,
                "split": split,
                "cohort": cohort,
                "threshold_dbz": thr,
                "tp": b["tp"],
                "fp": b["fp"],
                "fn": b["fn"],
                "tn": b["tn"],
                "brier_deterministic": (
                    b["brier_sum"] / n if n > 0 else np.nan
                ),
                "n_pixels": int(n),
            }
            row.update(
                event_metric_row(
                    b["tp"],
                    b["fp"],
                    b["fn"],
                    b["tn"],
                )
            )
            threshold_rows.append(row)

        fss_rows = []
        for (thr, k), v in self.fss.items():
            fss = (
                1.0 - v["num"] / v["den"]
                if v["den"] > 0
                else np.nan
            )
            fss_rows.append({
                "baseline": baseline,
                "split": split,
                "cohort": cohort,
                "threshold_dbz": thr,
                "support_pixels": k,
                "nominal_support_km": k * GRID_KM,
                "fss": fss,
                "fss_numerator": v["num"],
                "fss_denominator": v["den"],
                "n_neighborhood_values": v["n"],
            })

        patches = pd.DataFrame(self.patch_rows)
        patch_metric_rows = []
        group_rows = []

        if not patches.empty:
            for condition_name, condition in [
                ("all", np.ones(len(patches), dtype=bool)),
                ("target_ge30", patches["target_max_dbz"].to_numpy() >= 30),
                ("target_ge40", patches["target_max_dbz"].to_numpy() >= 40),
                ("target_ge45", patches["target_max_dbz"].to_numpy() >= 45),
            ]:
                t = patches.loc[condition]
                if t.empty:
                    continue
                diff = (
                    t["pred_max_dbz"].to_numpy()
                    - t["target_max_dbz"].to_numpy()
                )
                patch_metric_rows.append({
                    "baseline": baseline,
                    "split": split,
                    "cohort": cohort,
                    "condition": condition_name,
                    "n_patches": int(len(t)),
                    "rmse_max_dbz": float(
                        np.sqrt(np.mean(np.square(diff)))
                    ),
                    "mae_max_dbz": float(np.mean(np.abs(diff))),
                    "bias_max_dbz": float(np.mean(diff)),
                    "target_mean_max_dbz": float(
                        t["target_max_dbz"].mean()
                    ),
                    "pred_mean_max_dbz": float(
                        t["pred_max_dbz"].mean()
                    ),
                })

            for season in SEASONS:
                t = patches[patches["season_code"] == season]
                if t.empty:
                    continue
                diff = (
                    t["pred_max_dbz"].to_numpy()
                    - t["target_max_dbz"].to_numpy()
                )
                row = {
                    "baseline": baseline,
                    "split": split,
                    "cohort": cohort,
                    "group_type": "season",
                    "group_value": season,
                    "n_patches": int(len(t)),
                    "rmse_max_dbz": float(
                        np.sqrt(np.mean(np.square(diff)))
                    ),
                    "mae_max_dbz": float(np.mean(np.abs(diff))),
                    "bias_max_dbz": float(np.mean(diff)),
                }
                for thr in THRESHOLDS:
                    row[f"target_patch_event_rate_ge_{int(thr)}"] = float(
                        (t["target_max_dbz"] >= thr).mean()
                    )
                    row[f"pred_patch_event_rate_ge_{int(thr)}"] = float(
                        (t["pred_max_dbz"] >= thr).mean()
                    )
                group_rows.append(row)

        probabilistic_rows = []
        if self.gaussian_pixel_count > 0:
            probabilistic_rows.append({
                "baseline": baseline,
                "split": split,
                "cohort": cohort,
                "metric": "gaussian_nll_log1p",
                "threshold_dbz": np.nan,
                "value": (
                    self.gaussian_nll_sum
                    / self.gaussian_pixel_count
                ),
                "n_pixels": self.gaussian_pixel_count,
            })
            probabilistic_rows.append({
                "baseline": baseline,
                "split": split,
                "cohort": cohort,
                "metric": "gaussian_crps_log1p",
                "threshold_dbz": np.nan,
                "value": (
                    self.gaussian_crps_sum
                    / self.gaussian_pixel_count
                ),
                "n_pixels": self.gaussian_pixel_count,
            })

            for thr, b in self.prob_bins.items():
                n = int(b["count"].sum())
                nonzero = b["count"] > 0
                bin_prob = np.zeros(10)
                bin_obs = np.zeros(10)
                bin_prob[nonzero] = (
                    b["prob_sum"][nonzero] / b["count"][nonzero]
                )
                bin_obs[nonzero] = (
                    b["obs_sum"][nonzero] / b["count"][nonzero]
                )
                ece = float(
                    np.sum(
                        b["count"][nonzero]
                        / max(n, 1)
                        * np.abs(
                            bin_prob[nonzero] - bin_obs[nonzero]
                        )
                    )
                )
                probabilistic_rows.extend([
                    {
                        "baseline": baseline,
                        "split": split,
                        "cohort": cohort,
                        "metric": "brier_probability",
                        "threshold_dbz": thr,
                        "value": (
                            b["brier_sum"] / n if n else np.nan
                        ),
                        "n_pixels": n,
                    },
                    {
                        "baseline": baseline,
                        "split": split,
                        "cohort": cohort,
                        "metric": "ece_10bins",
                        "threshold_dbz": thr,
                        "value": ece,
                        "n_pixels": n,
                    },
                ])

        return (
            point,
            threshold_rows,
            fss_rows,
            patch_metric_rows,
            probabilistic_rows,
            group_rows,
        )


def infer_datetime_unit(values: np.ndarray) -> str:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "ns"
    mag = float(np.nanmedian(np.abs(finite.astype(np.float64))))
    if mag >= 1e17:
        return "ns"
    if mag >= 1e14:
        return "us"
    if mag >= 1e11:
        return "ms"
    return "s"


def timestamps_as_ns(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype("datetime64[ns]").astype(np.int64)
    unit = infer_datetime_unit(arr)
    return pd.to_datetime(
        arr,
        unit=unit,
        utc=True,
    ).asi8


def load_checkpoint_model(
    phase16_dir: Path,
    model_name: str,
    device: torch.device,
    base_channels: int,
):
    path = phase16_dir / "checkpoints" / f"{model_name}_best.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Train learned baselines first."
        )
    try:
        payload = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        payload = torch.load(
            path,
            map_location=device,
        )
    model = build_model(
        model_name,
        in_channels=len(CHANNELS),
        base_channels=int(
            payload.get("base_channels", base_channels)
        ),
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def persistence_prev_patch_map(
    root,
    patches_per_timestamp: int = 12,
) -> np.ndarray:
    ts = timestamps_as_ns(np.asarray(root["timestamps"][:]))
    if len(ts) % patches_per_timestamp != 0:
        raise RuntimeError("Unexpected patch count for persistence mapping.")

    groups = ts.reshape(-1, patches_per_timestamp)
    if not np.all(groups == groups[:, [0]]):
        raise RuntimeError(
            "Persistence baseline requires contiguous timestamp groups."
        )

    unique_ns = groups[:, 0]
    group_lookup = {
        int(t): i
        for i, t in enumerate(unique_ns)
    }
    hour_ns = int(pd.Timedelta(hours=1).value)

    prev_group = np.full(len(unique_ns), -1, dtype=np.int64)
    for i, t in enumerate(unique_ns):
        prev_group[i] = group_lookup.get(
            int(t - hour_ns),
            -1,
        )

    patch_group = np.arange(len(ts), dtype=np.int64) // patches_per_timestamp
    slot = np.arange(len(ts), dtype=np.int64) % patches_per_timestamp
    pg = prev_group[patch_group]

    prev_patch = np.full(len(ts), -1, dtype=np.int64)
    valid = pg >= 0
    prev_patch[valid] = (
        pg[valid] * patches_per_timestamp + slot[valid]
    )
    return prev_patch


def load_season_by_patch(
    phase16_dir: Path,
) -> np.ndarray:
    month = np.load(
        phase16_dir / "patch_month_local.npy"
    ).astype(np.int64)
    out = np.empty(len(month), dtype=object)
    out[np.isin(month, [12, 1, 2])] = "DJF"
    out[np.isin(month, [3, 4, 5])] = "MAM"
    out[np.isin(month, [6, 7, 8])] = "JJA"
    out[np.isin(month, [9, 10, 11])] = "SON"
    return out


def prediction_for_baseline(
    baseline: str,
    idx: np.ndarray,
    root,
    device: torch.device,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    climatology: np.ndarray,
    slot_mean: np.ndarray,
    global_mean: np.ndarray,
    month: np.ndarray,
    hour: np.ndarray,
    slot: np.ndarray,
    prev_patch: np.ndarray,
    learned_model=None,
    use_amp: bool = False,
):
    y = torch.from_numpy(
        np.asarray(
            zarr_take_first_axis(root["target"], idx),
            dtype=np.float32,
        )
    ).to(device)

    gaussian = None

    if baseline == "zero":
        pred = torch.zeros_like(y)

    elif baseline == "train_global_mean_field":
        arr = np.broadcast_to(
            global_mean[None, None, :, :],
            (len(idx), 1, global_mean.shape[0], global_mean.shape[1]),
        )
        pred = torch.from_numpy(
            np.asarray(arr, dtype=np.float32).copy()
        ).to(device)

    elif baseline == "train_slot_mean":
        arr = slot_mean[
            slot[idx].astype(np.int64)
        ][:, None, :, :]
        pred = torch.from_numpy(
            np.asarray(arr, dtype=np.float32)
        ).to(device)

    elif baseline == "climatology_month_hour_slot":
        arr = climatology[
            month[idx].astype(np.int64) - 1,
            hour[idx].astype(np.int64),
            slot[idx].astype(np.int64),
        ][:, None, :, :]
        pred = torch.from_numpy(
            np.asarray(arr, dtype=np.float32)
        ).to(device)

    elif baseline == "persistence_1h_radar_reference":
        prev = prev_patch[idx]
        if np.any(prev < 0):
            raise RuntimeError(
                "Persistence prediction called on patches without t-1h."
            )
        pred = torch.from_numpy(
            np.asarray(
                zarr_take_first_axis(root["target"], prev),
                dtype=np.float32,
            )
        ).to(device)

    elif baseline in LEARNED_MODELS:
        x = np.asarray(
            zarr_take_first_axis(root["input"], idx),
            dtype=np.float32,
        )
        x = (
            x
            - norm_mean[None, :, None, None]
        ) / norm_std[None, :, None, None]
        x = torch.from_numpy(x).to(device)

        amp_enabled = bool(use_amp and device.type == "cuda")
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            if baseline == "unet_gaussian":
                mu, log_sigma = learned_model(x)
                pred = mu.float()
                gaussian = (
                    mu.float(),
                    log_sigma.float(),
                )
            else:
                pred = learned_model(x).float()
    else:
        raise ValueError(baseline)

    return y, pred, gaussian


def evaluate_one(
    baseline: str,
    split: str,
    cohort: str,
    indices: np.ndarray,
    args: argparse.Namespace,
    root,
    device: torch.device,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    climatology: np.ndarray,
    slot_mean: np.ndarray,
    global_mean: np.ndarray,
    month: np.ndarray,
    hour: np.ndarray,
    slot: np.ndarray,
    seasons: np.ndarray,
    prev_patch: np.ndarray,
    learned_model,
    logger: logging.Logger,
):
    acc = Accumulator()

    n_batches = int(np.ceil(len(indices) / args.batch_size))
    for bi, start in enumerate(range(0, len(indices), args.batch_size)):
        if args.max_batches > 0 and bi >= args.max_batches:
            break

        idx = indices[start:start + args.batch_size]

        y, pred, gaussian = prediction_for_baseline(
            baseline=baseline,
            idx=idx,
            root=root,
            device=device,
            norm_mean=norm_mean,
            norm_std=norm_std,
            climatology=climatology,
            slot_mean=slot_mean,
            global_mean=global_mean,
            month=month,
            hour=hour,
            slot=slot,
            prev_patch=prev_patch,
            learned_model=learned_model,
            use_amp=args.amp,
        )

        acc.update(
            target_log=y,
            pred_log=pred,
            patch_indices=idx,
            season_codes=seasons[idx],
            gaussian=gaussian,
        )

        if bi == 0 or bi % 100 == 0:
            logger.info(
                "%s | %s | %s | batch %d/%d",
                split,
                cohort,
                baseline,
                bi + 1,
                n_batches,
            )

    return acc.finalize(
        baseline=baseline,
        split=split,
        cohort=cohort,
    )


def save_or_merge(
    df: pd.DataFrame,
    path: Path,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        old = pd.read_parquet(path)
        if not df.empty:
            key_cols = [
                c for c in [
                    "baseline",
                    "split",
                    "cohort",
                    "domain",
                    "threshold_dbz",
                    "support_pixels",
                    "condition",
                    "metric",
                    "group_type",
                    "group_value",
                ]
                if c in df.columns and c in old.columns
            ]
            if key_cols:
                new_keys = set(
                    map(
                        tuple,
                        df[key_cols].astype(str).to_numpy(),
                    )
                )
                old_mask = [
                    tuple(row) not in new_keys
                    for row in old[key_cols].astype(str).to_numpy()
                ]
                old = old.loc[old_mask]
        df = pd.concat([old, df], ignore_index=True)
    df.to_parquet(path, index=False)


def main() -> None:
    args = parse_args()
    logger = setup_logger(args.phase16_dir)

    splits = [
        x.strip()
        for x in args.splits.split(",")
        if x.strip()
    ]
    baselines = [
        x.strip()
        for x in args.baselines.split(",")
        if x.strip()
    ]

    allowed_splits = {
        "validation",
        "test_primary",
        "test_stress_ood",
    }
    if set(splits) - allowed_splits:
        raise ValueError(
            f"Allowed evaluation splits: {sorted(allowed_splits)}"
        )

    locked = {"test_primary", "test_stress_ood"}
    if locked.intersection(splits) and not args.allow_locked_splits:
        raise RuntimeError(
            "Locked test requested. Re-run with --allow-locked-splits only "
            "after the experimental design/model is frozen."
        )

    allowed_baselines = set(NONLEARNED + LEARNED_MODELS)
    unknown = sorted(set(baselines) - allowed_baselines)
    if unknown:
        raise ValueError(f"Unknown baselines: {unknown}")

    zarr_path = args.dataset_dir / "train.zarr"
    root = open_group(zarr_path)

    norm_path = args.phase16_dir / "train_input_normalization.npz"
    clim_path = (
        args.phase16_dir / "train_climatology_month_hour_slot.npz"
    )
    if not norm_path.exists() or not clim_path.exists():
        raise FileNotFoundError(
            "Run 16_prepare_baseline_artifacts.py first."
        )

    with np.load(norm_path, allow_pickle=True) as z:
        norm_mean = np.asarray(z["mean"], dtype=np.float32)
        norm_std = np.asarray(z["std"], dtype=np.float32)

    with np.load(clim_path, allow_pickle=True) as z:
        climatology = np.asarray(
            z["mean_target_log1p"],
            dtype=np.float32,
        )
        slot_mean = np.asarray(
            z["slot_mean_target_log1p"],
            dtype=np.float32,
        )
        global_mean = np.asarray(
            z["global_mean_target_log1p"],
            dtype=np.float32,
        )

    month = np.load(
        args.phase16_dir / "patch_month_local.npy"
    ).astype(np.uint8)
    hour = np.load(
        args.phase16_dir / "patch_hour_local.npy"
    ).astype(np.uint8)
    slot = np.load(
        args.phase16_dir / "patch_slot_codes.npy"
    ).astype(np.uint8)
    seasons = load_season_by_patch(args.phase16_dir)

    prev_patch = persistence_prev_patch_map(root)

    device = choose_device(args.device)

    learned_models = {}
    for name in baselines:
        if name in LEARNED_MODELS:
            learned_models[name], _ = load_checkpoint_model(
                args.phase16_dir,
                name,
                device,
                args.base_channels,
            )

    point_rows = []
    threshold_rows = []
    fss_rows = []
    patch_rows = []
    probabilistic_rows = []
    group_rows = []
    cohort_rows = []

    logger.info("=" * 92)
    logger.info("CORRDIFF PHASE 16 - EVALUATE BASELINES")
    logger.info("Splits: %s", splits)
    logger.info("Baselines: %s", baselines)
    logger.info("Device: %s", device)
    logger.info("=" * 92)

    for split in splits:
        idx_path = args.phase15_dir / f"{split}_patch_indices.npy"
        if not idx_path.exists():
            raise FileNotFoundError(idx_path)

        full_idx = np.load(idx_path).astype(np.int64)
        common_idx = full_idx[prev_patch[full_idx] >= 0]

        cohort_rows.append({
            "split": split,
            "cohort": "full_split",
            "n_patches": int(len(full_idx)),
            "fraction_of_full_split": 1.0,
        })
        cohort_rows.append({
            "split": split,
            "cohort": "persistence_common",
            "n_patches": int(len(common_idx)),
            "fraction_of_full_split": (
                len(common_idx) / len(full_idx)
                if len(full_idx) else np.nan
            ),
        })

        for baseline in baselines:
            if baseline == "persistence_1h_radar_reference":
                cohort_specs = [
                    ("persistence_common", common_idx),
                ]
            else:
                cohort_specs = [
                    ("full_split", full_idx),
                ]
                if args.with_persistence_common:
                    cohort_specs.append(
                        ("persistence_common", common_idx)
                    )

            for cohort, indices in cohort_specs:
                if len(indices) == 0:
                    continue

                result = evaluate_one(
                    baseline=baseline,
                    split=split,
                    cohort=cohort,
                    indices=indices,
                    args=args,
                    root=root,
                    device=device,
                    norm_mean=norm_mean,
                    norm_std=norm_std,
                    climatology=climatology,
                    slot_mean=slot_mean,
                    global_mean=global_mean,
                    month=month,
                    hour=hour,
                    slot=slot,
                    seasons=seasons,
                    prev_patch=prev_patch,
                    learned_model=learned_models.get(baseline),
                    logger=logger,
                )
                a, b, c, d, e, f = result
                point_rows.extend(a)
                threshold_rows.extend(b)
                fss_rows.extend(c)
                patch_rows.extend(d)
                probabilistic_rows.extend(e)
                group_rows.extend(f)

    outputs = {
        "point_metrics.parquet": pd.DataFrame(point_rows),
        "threshold_metrics.parquet": pd.DataFrame(threshold_rows),
        "fss_metrics.parquet": pd.DataFrame(fss_rows),
        "patch_max_metrics.parquet": pd.DataFrame(patch_rows),
        "probabilistic_metrics.parquet": pd.DataFrame(probabilistic_rows),
        "group_patch_metrics.parquet": pd.DataFrame(group_rows),
        "evaluation_cohorts.parquet": pd.DataFrame(cohort_rows),
    }

    for name, df in outputs.items():
        save_or_merge(
            df,
            args.phase16_dir / name,
            args.overwrite_metrics,
        )

    manifest = {
        "phase_version": PHASE_VERSION,
        "evaluated_splits": splits,
        "baselines": baselines,
        "locked_split_override_used": bool(
            locked.intersection(splits)
        ),
        "with_persistence_common": bool(
            args.with_persistence_common
        ),
        "metric_domains": [
            "stored_log1p",
            "reconstructed_dBZ",
            "threshold event skill",
            "FSS",
            "patch-max intensity",
            "probabilistic Gaussian metrics",
        ],
        "thresholds_dbz": THRESHOLDS,
        "fss_support_pixels": FSS_SUPPORT_PIXELS,
        "fss_nominal_support_km": [
            x * GRID_KM for x in FSS_SUPPORT_PIXELS
        ],
        "important_note": (
            "persistence_1h_radar_reference uses previous radar and is not an "
            "input-equivalent baseline for ERA5-only CorrDiff. Compare it only "
            "on persistence_common cohort."
        ),
    }
    (
        args.phase16_dir / "evaluation_manifest.json"
    ).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
