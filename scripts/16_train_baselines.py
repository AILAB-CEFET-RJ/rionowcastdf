#!/usr/bin/env python3
"""
CorrDiff - Fase 16
Treino dos baselines aprendidos
===============================

Modelos:
- pixel_mlp            : baseline local 1x1, sem contexto espacial;
- unet_deterministic   : U-Net pequeno com MSE no target log1p;
- unet_gaussian        : U-Net heteroscedástico com Gaussian NLL no log1p.

Somente train e validation da Fase 15 são usados. Testes permanecem bloqueados.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError as exc:
    raise SystemExit("Phase 16 requires PyTorch.") from exc

from phase16_common import (
    CHANNELS,
    ZarrPatchDataset,
    build_model,
    gaussian_nll,
    set_seed,
)


PHASE_VERSION = "phase16-baselines-v1.1-zarr-oindex-frozen-split"


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
        "--models",
        default="pixel_mlp,unet_deterministic,unet_gaussian",
    )
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    p.add_argument(
        "--amp",
        action="store_true",
        help="Use mixed precision on CUDA.",
    )
    p.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help="0 = full epoch; useful only for smoke tests.",
    )
    p.add_argument(
        "--max-val-batches",
        type=int,
        default=0,
        help="0 = full validation.",
    )
    p.add_argument("--overwrite-checkpoints", action="store_true")
    return p.parse_args()


def setup_logger(path: Path) -> logging.Logger:
    path.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("corrdiff.phase16.train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(path / "phase16_train.log", encoding="utf-8")
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


def load_normalization(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 16_prepare_baseline_artifacts.py first."
        )
    with np.load(path, allow_pickle=True) as z:
        mean = np.asarray(z["mean"], dtype=np.float32)
        std = np.asarray(z["std"], dtype=np.float32)
    return mean, std


def model_loss(
    model_name: str,
    model,
    x: torch.Tensor,
    y: torch.Tensor,
):
    if model_name == "unet_gaussian":
        mu, log_sigma = model(x)
        loss = gaussian_nll(y, mu, log_sigma)
        point = mu
    else:
        point = model(x)
        loss = torch.mean(torch.square(point - y))
    return loss, point


@torch.no_grad()
def validate(
    model_name: str,
    model,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    max_batches: int,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_sq = 0.0
    total_abs = 0.0
    total_bias = 0.0
    total_pixels = 0
    total_batches = 0

    amp_enabled = bool(use_amp and device.type == "cuda")

    for batch_idx, (_, x, y) in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            loss, point = model_loss(
                model_name,
                model,
                x,
                y,
            )

        diff = point.float() - y.float()
        npx = diff.numel()

        total_loss += float(loss.detach().cpu()) * len(x)
        total_sq += float(torch.square(diff).sum().cpu())
        total_abs += float(diff.abs().sum().cpu())
        total_bias += float(diff.sum().cpu())
        total_pixels += npx
        total_batches += len(x)

    return {
        "val_loss": (
            total_loss / total_batches
            if total_batches else np.nan
        ),
        "val_rmse_log1p": (
            float(np.sqrt(total_sq / total_pixels))
            if total_pixels else np.nan
        ),
        "val_mae_log1p": (
            total_abs / total_pixels
            if total_pixels else np.nan
        ),
        "val_bias_log1p": (
            total_bias / total_pixels
            if total_pixels else np.nan
        ),
        "n_val_examples": int(total_batches),
    }


def train_one(
    model_name: str,
    args: argparse.Namespace,
    logger: logging.Logger,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> list[dict]:
    set_seed(args.seed)

    model = build_model(
        model_name,
        in_channels=len(CHANNELS),
        base_channels=args.base_channels,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    amp_enabled = bool(args.amp and device.type == "cuda")
    try:
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=amp_enabled,
        )
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(
            enabled=amp_enabled,
        )

    checkpoint_dir = args.phase16_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / f"{model_name}_best.pt"

    if best_path.exists() and not args.overwrite_checkpoints:
        raise FileExistsError(
            f"{best_path} exists; use --overwrite-checkpoints."
        )

    history = []
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_examples = 0
        t0 = time.time()

        for batch_idx, (_, x, y) in enumerate(train_loader):
            if (
                args.max_train_batches > 0
                and batch_idx >= args.max_train_batches
            ):
                break

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                loss, _ = model_loss(
                    model_name,
                    model,
                    x,
                    y,
                )

            scaler.scale(loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip,
                )

            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.detach().cpu()) * len(x)
            n_examples += len(x)

        train_loss = (
            epoch_loss / n_examples
            if n_examples else np.nan
        )

        val_metrics = validate(
            model_name,
            model,
            val_loader,
            device,
            args.amp,
            args.max_val_batches,
        )

        row = {
            "model": model_name,
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
            "elapsed_seconds": time.time() - t0,
            "device": str(device),
        }
        history.append(row)

        logger.info(
            "%s epoch=%d train_loss=%.6f val_loss=%.6f "
            "val_rmse_log1p=%.6f",
            model_name,
            epoch,
            train_loss,
            val_metrics["val_loss"],
            val_metrics["val_rmse_log1p"],
        )

        if val_metrics["val_loss"] < best_val:
            best_val = val_metrics["val_loss"]
            payload = {
                "phase_version": PHASE_VERSION,
                "model_name": model_name,
                "base_channels": args.base_channels,
                "channels": CHANNELS,
                "epoch": epoch,
                "best_val_loss": best_val,
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "training_args": vars(args),
            }
            torch.save(payload, best_path)

    return history


def main() -> None:
    args = parse_args()
    logger = setup_logger(args.phase16_dir)

    zarr_path = args.dataset_dir / "train.zarr"
    train_idx_path = args.phase15_dir / "train_patch_indices.npy"
    val_idx_path = args.phase15_dir / "validation_patch_indices.npy"
    norm_path = args.phase16_dir / "train_input_normalization.npz"

    for p in [
        zarr_path,
        train_idx_path,
        val_idx_path,
        norm_path,
    ]:
        if not p.exists():
            raise FileNotFoundError(p)

    input_mean, input_std = load_normalization(norm_path)
    train_idx = np.load(train_idx_path).astype(np.int64)
    val_idx = np.load(val_idx_path).astype(np.int64)

    train_ds = ZarrPatchDataset(
        zarr_path,
        train_idx,
        input_mean,
        input_std,
    )
    val_ds = ZarrPatchDataset(
        zarr_path,
        val_idx,
        input_mean,
        input_std,
    )

    device = choose_device(args.device)

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        generator=generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **loader_kwargs,
    )

    model_names = [
        x.strip()
        for x in args.models.split(",")
        if x.strip()
    ]

    allowed = {
        "pixel_mlp",
        "unet_deterministic",
        "unet_gaussian",
    }
    unknown = sorted(set(model_names) - allowed)
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")

    logger.info("=" * 88)
    logger.info("CORRDIFF PHASE 16 - TRAIN BASELINES")
    logger.info("Device: %s", device)
    logger.info("Train patches: %d", len(train_idx))
    logger.info("Validation patches: %d", len(val_idx))
    logger.info("Models: %s", model_names)
    logger.info("=" * 88)

    all_history = []

    for model_name in model_names:
        rows = train_one(
            model_name,
            args,
            logger,
            train_loader,
            val_loader,
            device,
        )
        all_history.extend(rows)

    hist = pd.DataFrame(all_history)

    hist_path = args.phase16_dir / "training_history.parquet"
    if hist_path.exists():
        old = pd.read_parquet(hist_path)
        hist = pd.concat([old, hist], ignore_index=True)
    hist.to_parquet(hist_path, index=False)

    manifest = {
        "phase_version": PHASE_VERSION,
        "training_split": "train",
        "checkpoint_selection_split": "validation",
        "locked_splits_used": [],
        "models": model_names,
        "losses": {
            "pixel_mlp": "MSE on stored log1p target",
            "unet_deterministic": "MSE on stored log1p target",
            "unet_gaussian": "heteroscedastic Gaussian NLL on stored log1p target",
        },
        "sampler": "natural training patch distribution; no target rebalance",
        "checkpoint_rule": "minimum validation loss",
        "normalization": "train-only channel mean/std from Phase 16 preparation",
        "seed": args.seed,
    }
    (
        args.phase16_dir / "learned_baseline_training_manifest.json"
    ).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
