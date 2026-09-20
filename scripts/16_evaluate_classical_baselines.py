#!/usr/bin/env python3
"""
CorrDiff - Fase 16 v2
Avaliação dos baselines clássicos no split congelado da Fase 15.

Não treina modelos neurais.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from phase16_common import open_group, zarr_take_first_axis, infer_datetime_unit


PHASE_VERSION = "phase16-baselines-v2-classical-nvidia-handoff"

BASELINES = [
    "zero",
    "train_global_mean_field",
    "train_slot_mean",
    "climatology_month_hour_slot",
    "persistence_1h_radar_reference",
]

THRESHOLDS = [20.0, 30.0, 40.0, 45.0]
FSS_SUPPORT_PIXELS = [1, 2, 4, 8]
GRID_KM = 2.0
SEASONS = ["DJF", "MAM", "JJA", "SON"]


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
        help="validation,test_primary,test_stress_ood",
    )
    p.add_argument(
        "--baselines",
        default=",".join(BASELINES),
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument(
        "--with-persistence-common",
        action="store_true",
        help=(
            "Also evaluate non-persistence baselines on the exact subset "
            "where radar t-1h exists."
        ),
    )
    p.add_argument(
        "--allow-locked-splits",
        action="store_true",
    )
    p.add_argument("--overwrite-metrics", action="store_true")
    return p.parse_args()


def setup_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("corrdiff.phase16v2.evaluate")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(
        path / "phase16_evaluate.log",
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def timestamps_as_ns(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype("datetime64[ns]").astype(np.int64)
    return pd.to_datetime(
        arr,
        unit=infer_datetime_unit(arr),
        utc=True,
    ).asi8


def persistence_prev_patch_map(
    root,
    patches_per_timestamp: int = 12,
) -> np.ndarray:
    ts = timestamps_as_ns(np.asarray(root["timestamps"][:]))
    if len(ts) % patches_per_timestamp != 0:
        raise RuntimeError("Unexpected patch count.")

    groups = ts.reshape(-1, patches_per_timestamp)
    if not np.all(groups == groups[:, [0]]):
        raise RuntimeError(
            "Persistence requires contiguous timestamp groups."
        )

    unique_ns = groups[:, 0]
    lookup = {int(t): i for i, t in enumerate(unique_ns)}
    hour_ns = int(pd.Timedelta(hours=1).value)

    prev_group = np.full(len(unique_ns), -1, dtype=np.int64)
    for i, t in enumerate(unique_ns):
        prev_group[i] = lookup.get(int(t - hour_ns), -1)

    patch_group = np.arange(len(ts), dtype=np.int64) // patches_per_timestamp
    slot = np.arange(len(ts), dtype=np.int64) % patches_per_timestamp
    pg = prev_group[patch_group]

    out = np.full(len(ts), -1, dtype=np.int64)
    valid = pg >= 0
    out[valid] = (
        pg[valid] * patches_per_timestamp + slot[valid]
    )
    return out


def season_codes(month: np.ndarray) -> np.ndarray:
    month = np.asarray(month)
    out = np.empty(len(month), dtype=object)
    out[np.isin(month, [12, 1, 2])] = "DJF"
    out[np.isin(month, [3, 4, 5])] = "MAM"
    out[np.isin(month, [6, 7, 8])] = "JJA"
    out[np.isin(month, [9, 10, 11])] = "SON"
    return out


def fraction_field(binary: np.ndarray, k: int) -> np.ndarray:
    """
    Moving-window fraction with valid windows.
    input: [N,H,W]
    """
    if k == 1:
        return binary.astype(np.float32, copy=False)
    windows = sliding_window_view(
        binary,
        window_shape=(k, k),
        axis=(-2, -1),
    )
    return windows.mean(axis=(-2, -1), dtype=np.float64)


def categorical_metrics(tp, fp, fn, tn) -> dict[str, float]:
    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else np.nan
    pod = tp / (tp + fn) if tp + fn else np.nan
    far = fp / (tp + fp) if tp + fp else np.nan
    csi = tp / (tp + fp + fn) if tp + fp + fn else np.nan
    f1 = (
        2 * precision * pod / (precision + pod)
        if np.isfinite(precision)
        and np.isfinite(pod)
        and precision + pod > 0
        else np.nan
    )
    frequency_bias = (
        (tp + fp) / (tp + fn)
        if tp + fn else np.nan
    )
    random_hits = (
        (tp + fp) * (tp + fn) / n
        if n else np.nan
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
            thr: dict(tp=0, fp=0, fn=0, tn=0)
            for thr in THRESHOLDS
        }
        self.fss = {
            (thr, k): dict(num=0.0, den=0.0)
            for thr in THRESHOLDS
            for k in FSS_SUPPORT_PIXELS
        }
        self.patch_rows = []

    def update(
        self,
        target_log: np.ndarray,
        pred_log: np.ndarray,
        patch_indices: np.ndarray,
        seasons: np.ndarray,
    ) -> None:
        y = np.asarray(target_log, dtype=np.float64)
        p = np.maximum(
            np.asarray(pred_log, dtype=np.float64),
            0.0,
        )

        diff_log = p - y
        self.n_pixels += int(diff_log.size)
        self.sum_sq_log += float(np.square(diff_log).sum())
        self.sum_abs_log += float(np.abs(diff_log).sum())
        self.sum_bias_log += float(diff_log.sum())

        y_dbz = np.maximum(np.expm1(y), 0.0)
        p_dbz = np.maximum(np.expm1(p), 0.0)
        diff_dbz = p_dbz - y_dbz

        self.sum_sq_dbz += float(np.square(diff_dbz).sum())
        self.sum_abs_dbz += float(np.abs(diff_dbz).sum())
        self.sum_bias_dbz += float(diff_dbz.sum())

        max_y = y_dbz.max(axis=(1, 2))
        max_p = p_dbz.max(axis=(1, 2))

        for idx, season, ty, py in zip(
            patch_indices,
            seasons,
            max_y,
            max_p,
        ):
            self.patch_rows.append({
                "patch_index": int(idx),
                "season_code": str(season),
                "target_max_dbz": float(ty),
                "pred_max_dbz": float(py),
            })

        for thr in THRESHOLDS:
            yt = y_dbz >= thr
            pt = p_dbz >= thr

            b = self.binary[thr]
            b["tp"] += int(np.logical_and(pt, yt).sum())
            b["fp"] += int(np.logical_and(pt, ~yt).sum())
            b["fn"] += int(np.logical_and(~pt, yt).sum())
            b["tn"] += int(np.logical_and(~pt, ~yt).sum())

            for k in FSS_SUPPORT_PIXELS:
                ofrac = fraction_field(yt, k)
                pfrac = fraction_field(pt, k)
                v = self.fss[(thr, k)]
                v["num"] += float(
                    np.square(pfrac - ofrac).sum()
                )
                v["den"] += float(
                    (
                        np.square(pfrac)
                        + np.square(ofrac)
                    ).sum()
                )

    def finalize(
        self,
        baseline: str,
        split: str,
        cohort: str,
    ):
        point_rows = [
            {
                "baseline": baseline,
                "split": split,
                "cohort": cohort,
                "domain": "stored_log1p",
                "rmse": math.sqrt(
                    self.sum_sq_log / self.n_pixels
                ),
                "mae": self.sum_abs_log / self.n_pixels,
                "bias": self.sum_bias_log / self.n_pixels,
                "n_pixels": self.n_pixels,
            },
            {
                "baseline": baseline,
                "split": split,
                "cohort": cohort,
                "domain": "dbz_reconstructed",
                "rmse": math.sqrt(
                    self.sum_sq_dbz / self.n_pixels
                ),
                "mae": self.sum_abs_dbz / self.n_pixels,
                "bias": self.sum_bias_dbz / self.n_pixels,
                "n_pixels": self.n_pixels,
            },
        ]

        threshold_rows = []
        for thr, b in self.binary.items():
            row = {
                "baseline": baseline,
                "split": split,
                "cohort": cohort,
                "threshold_dbz": thr,
                "tp": b["tp"],
                "fp": b["fp"],
                "fn": b["fn"],
                "tn": b["tn"],
            }
            row.update(
                categorical_metrics(
                    b["tp"], b["fp"], b["fn"], b["tn"]
                )
            )
            threshold_rows.append(row)

        fss_rows = []
        for (thr, k), v in self.fss.items():
            fss_rows.append({
                "baseline": baseline,
                "split": split,
                "cohort": cohort,
                "threshold_dbz": thr,
                "support_pixels": k,
                "nominal_support_km": k * GRID_KM,
                "fss": (
                    1.0 - v["num"] / v["den"]
                    if v["den"] > 0
                    else np.nan
                ),
                "fss_numerator": v["num"],
                "fss_denominator": v["den"],
            })

        patches = pd.DataFrame(self.patch_rows)
        patch_rows = []
        seasonal_rows = []

        for condition_name, mask in [
            ("all", np.ones(len(patches), dtype=bool)),
            (
                "target_ge30",
                patches["target_max_dbz"].to_numpy() >= 30,
            ),
            (
                "target_ge40",
                patches["target_max_dbz"].to_numpy() >= 40,
            ),
            (
                "target_ge45",
                patches["target_max_dbz"].to_numpy() >= 45,
            ),
        ]:
            t = patches.loc[mask]
            if t.empty:
                continue
            diff = (
                t["pred_max_dbz"].to_numpy()
                - t["target_max_dbz"].to_numpy()
            )
            patch_rows.append({
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
                "season_code": season,
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
            seasonal_rows.append(row)

        return (
            point_rows,
            threshold_rows,
            fss_rows,
            patch_rows,
            seasonal_rows,
        )


def predict(
    baseline: str,
    idx: np.ndarray,
    root,
    artifacts: dict[str, np.ndarray],
    month: np.ndarray,
    hour: np.ndarray,
    slot: np.ndarray,
    prev_patch: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(
        zarr_take_first_axis(
            root["target"],
            idx,
            tail_selection=(0, slice(None), slice(None)),
        ),
        dtype=np.float32,
    )

    if baseline == "zero":
        pred = np.zeros_like(y)

    elif baseline == "train_global_mean_field":
        field = artifacts["global"]
        pred = np.broadcast_to(
            field[None, :, :],
            y.shape,
        ).copy()

    elif baseline == "train_slot_mean":
        pred = artifacts["slot"][slot[idx].astype(np.int64)]

    elif baseline == "climatology_month_hour_slot":
        pred = artifacts["clim"][
            month[idx].astype(np.int64) - 1,
            hour[idx].astype(np.int64),
            slot[idx].astype(np.int64),
        ]

    elif baseline == "persistence_1h_radar_reference":
        prev = prev_patch[idx]
        if np.any(prev < 0):
            raise RuntimeError(
                "Persistence called outside persistence_common."
            )
        pred = np.asarray(
            zarr_take_first_axis(
                root["target"],
                prev,
                tail_selection=(0, slice(None), slice(None)),
            ),
            dtype=np.float32,
        )

    else:
        raise ValueError(baseline)

    return y, np.asarray(pred, dtype=np.float32)


def save_or_merge(
    df: pd.DataFrame,
    path: Path,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        old = pd.read_parquet(path)
        if not df.empty and not old.empty:
            key_cols = [
                c for c in [
                    "baseline", "split", "cohort", "domain",
                    "threshold_dbz", "support_pixels",
                    "condition", "season_code",
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
                keep = [
                    tuple(row) not in new_keys
                    for row in old[key_cols].astype(str).to_numpy()
                ]
                old = old.loc[keep]
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
        raise ValueError("Unknown split.")
    if set(baselines) - set(BASELINES):
        raise ValueError("Unknown baseline.")

    locked = {"test_primary", "test_stress_ood"}
    if locked.intersection(splits) and not args.allow_locked_splits:
        raise RuntimeError(
            "Locked split requested. Use --allow-locked-splits only "
            "after the experimental protocol is frozen."
        )

    root = open_group(args.dataset_dir / "train.zarr")

    art_path = args.phase16_dir / "train_classical_baselines.npz"
    if not art_path.exists():
        raise FileNotFoundError(
            f"{art_path} not found; run prepare script first."
        )

    with np.load(art_path, allow_pickle=True) as z:
        artifacts = {
            "global": np.asarray(
                z["global_mean_target_log1p"],
                dtype=np.float32,
            ),
            "slot": np.asarray(
                z["slot_mean_target_log1p"],
                dtype=np.float32,
            ),
            "clim": np.asarray(
                z["month_hour_slot_mean_target_log1p"],
                dtype=np.float32,
            ),
        }

    month = np.load(
        args.phase16_dir / "patch_month_local.npy"
    )
    hour = np.load(
        args.phase16_dir / "patch_hour_local.npy"
    )
    slot = np.load(
        args.phase16_dir / "patch_slot_codes.npy"
    )
    seasons = season_codes(month)
    prev_patch = persistence_prev_patch_map(root)

    point_rows = []
    threshold_rows = []
    fss_rows = []
    patch_rows = []
    seasonal_rows = []
    cohort_rows = []

    logger.info("=" * 88)
    logger.info("CORRDIFF PHASE 16 v2 - CLASSICAL BASELINE EVALUATION")
    logger.info("Splits: %s", splits)
    logger.info("Baselines: %s", baselines)
    logger.info("=" * 88)

    for split in splits:
        idx_path = (
            args.phase15_dir / f"{split}_patch_indices.npy"
        )
        full_idx = np.load(idx_path).astype(np.int64)
        common_idx = full_idx[prev_patch[full_idx] >= 0]

        cohort_rows.extend([
            {
                "split": split,
                "cohort": "full_split",
                "n_patches": int(len(full_idx)),
                "fraction_of_full_split": 1.0,
            },
            {
                "split": split,
                "cohort": "persistence_common",
                "n_patches": int(len(common_idx)),
                "fraction_of_full_split": (
                    len(common_idx) / len(full_idx)
                    if len(full_idx) else np.nan
                ),
            },
        ])

        for baseline in baselines:
            if baseline == "persistence_1h_radar_reference":
                cohorts = [
                    ("persistence_common", common_idx),
                ]
            else:
                cohorts = [("full_split", full_idx)]
                if args.with_persistence_common:
                    cohorts.append(
                        ("persistence_common", common_idx)
                    )

            for cohort, indices in cohorts:
                if len(indices) == 0:
                    continue

                acc = Accumulator()
                n_batches = int(
                    np.ceil(len(indices) / args.batch_size)
                )

                for bi, start in enumerate(
                    range(0, len(indices), args.batch_size)
                ):
                    idx = indices[
                        start:start + args.batch_size
                    ]
                    y, p = predict(
                        baseline,
                        idx,
                        root,
                        artifacts,
                        month,
                        hour,
                        slot,
                        prev_patch,
                    )
                    acc.update(
                        y,
                        p,
                        idx,
                        seasons[idx],
                    )

                    if bi == 0 or bi % 100 == 0:
                        logger.info(
                            "%s | %s | %s | %d/%d",
                            split,
                            cohort,
                            baseline,
                            bi + 1,
                            n_batches,
                        )

                result = acc.finalize(
                    baseline,
                    split,
                    cohort,
                )
                a, b, c, d, e = result
                point_rows.extend(a)
                threshold_rows.extend(b)
                fss_rows.extend(c)
                patch_rows.extend(d)
                seasonal_rows.extend(e)

    outputs = {
        "point_metrics.parquet": pd.DataFrame(point_rows),
        "threshold_metrics.parquet": pd.DataFrame(threshold_rows),
        "fss_metrics.parquet": pd.DataFrame(fss_rows),
        "patch_max_metrics.parquet": pd.DataFrame(patch_rows),
        "seasonal_patch_metrics.parquet": pd.DataFrame(seasonal_rows),
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
        "persistence_note": (
            "Persistence uses observed radar t-1h and is not input-equivalent "
            "to ERA5-only CorrDiff."
        ),
        "no_neural_training": True,
    }
    (
        args.phase16_dir / "classical_evaluation_manifest.json"
    ).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
