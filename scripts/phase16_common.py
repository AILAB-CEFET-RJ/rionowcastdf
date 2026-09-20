"""Shared utilities for CorrDiff Phase 16 baselines."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset
except ImportError as exc:
    raise SystemExit("Phase 16 requires PyTorch.") from exc

try:
    import zarr
except ImportError as exc:
    raise SystemExit("Phase 16 requires zarr.") from exc


CHANNELS = [
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


def open_group(path: Path):
    try:
        return zarr.open_group(str(path), mode="r")
    except Exception:
        return zarr.open(str(path), mode="r")


def zarr_take_first_axis(arr, indices: np.ndarray, tail_selection=None):
    """
    Compatibility helper for Zarr v2/v3.

    NumPy-style ``arr[np.ndarray]`` is not supported by Zarr v2 basic
    indexing. For a list/array of row indices, use orthogonal indexing.

    Parameters
    ----------
    arr
        Zarr array.
    indices
        1-D integer indices for the first axis.
    tail_selection
        Optional tuple for remaining axes. When omitted, all remaining
        dimensions are selected with ``slice(None)``.

    Returns
    -------
    np.ndarray
        Materialized selection, preserving the order of ``indices``.
    """
    idx = np.asarray(indices, dtype=np.int64)
    if idx.ndim != 1:
        raise ValueError("indices must be a 1-D integer array")

    if tail_selection is None:
        tail_selection = tuple(
            slice(None) for _ in range(arr.ndim - 1)
        )

    selection = (idx,) + tuple(tail_selection)

    # Zarr v2 and v3 both expose orthogonal indexing in common releases.
    if hasattr(arr, "oindex"):
        return np.asarray(arr.oindex[selection])

    # Fallback for implementations exposing the method directly.
    if hasattr(arr, "get_orthogonal_selection"):
        return np.asarray(arr.get_orthogonal_selection(selection))

    raise RuntimeError(
        "This Zarr implementation does not expose orthogonal indexing "
        "(oindex/get_orthogonal_selection)."
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ZarrPatchDataset(Dataset):
    """
    Lazy Zarr-backed patch dataset.

    Returns:
        original_patch_index: int64
        normalized_input: float32 [C,H,W]
        target_log1p: float32 [1,H,W]
    """

    def __init__(
        self,
        zarr_path: Path,
        indices: np.ndarray,
        input_mean: np.ndarray,
        input_std: np.ndarray,
    ):
        self.zarr_path = Path(zarr_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.input_mean = np.asarray(input_mean, dtype=np.float32)[:, None, None]
        self.input_std = np.asarray(input_std, dtype=np.float32)[:, None, None]
        self._root = None

    def __len__(self) -> int:
        return len(self.indices)

    def _get_root(self):
        if self._root is None:
            self._root = open_group(self.zarr_path)
        return self._root

    def __getitem__(self, item: int):
        idx = int(self.indices[item])
        root = self._get_root()
        x = np.asarray(root["input"][idx], dtype=np.float32)
        y = np.asarray(root["target"][idx], dtype=np.float32)
        x = (x - self.input_mean) / self.input_std
        return (
            np.int64(idx),
            torch.from_numpy(x),
            torch.from_numpy(y),
        )


class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        groups = min(8, c_out)
        while c_out % groups != 0 and groups > 1:
            groups -= 1
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.GroupNorm(groups, c_out),
            nn.GELU(),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.GroupNorm(groups, c_out),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class PixelMLP(nn.Module):
    """1x1 conv baseline: no spatial context beyond each aligned pixel."""

    def __init__(self, in_channels: int = 12, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        nn.init.constant_(self.net[-1].bias, -2.0)

    def forward(self, x):
        raw = self.net(x)
        return F.softplus(raw)


class UNetBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int = 12,
        base_channels: int = 32,
        out_channels: int = 1,
    ):
        super().__init__()
        b = base_channels

        self.enc1 = ConvBlock(in_channels, b)
        self.down1 = nn.Conv2d(b, 2 * b, 4, stride=2, padding=1)
        self.enc2 = ConvBlock(2 * b, 2 * b)

        self.down2 = nn.Conv2d(2 * b, 4 * b, 4, stride=2, padding=1)
        self.enc3 = ConvBlock(4 * b, 4 * b)

        self.down3 = nn.Conv2d(4 * b, 8 * b, 4, stride=2, padding=1)
        self.bottleneck = ConvBlock(8 * b, 8 * b)

        self.up3 = nn.ConvTranspose2d(8 * b, 4 * b, 4, stride=2, padding=1)
        self.dec3 = ConvBlock(8 * b, 4 * b)

        self.up2 = nn.ConvTranspose2d(4 * b, 2 * b, 4, stride=2, padding=1)
        self.dec2 = ConvBlock(4 * b, 2 * b)

        self.up1 = nn.ConvTranspose2d(2 * b, b, 4, stride=2, padding=1)
        self.dec1 = ConvBlock(2 * b, b)

        self.head = nn.Conv2d(b, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        z = self.bottleneck(self.down3(e3))

        d3 = self.up3(z)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)


class DeterministicUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 12,
        base_channels: int = 32,
    ):
        super().__init__()
        self.backbone = UNetBackbone(
            in_channels=in_channels,
            base_channels=base_channels,
            out_channels=1,
        )
        nn.init.constant_(self.backbone.head.bias, -2.0)

    def forward(self, x):
        return F.softplus(self.backbone(x))


class GaussianUNet(nn.Module):
    """
    Heteroscedastic Gaussian baseline in stored log1p target space.

    This is intentionally simple. It is not expected to model the zero-inflated
    target perfectly; it is a probabilistic baseline for CorrDiff.
    """

    def __init__(
        self,
        in_channels: int = 12,
        base_channels: int = 32,
        min_log_sigma: float = -5.0,
        max_log_sigma: float = 2.0,
    ):
        super().__init__()
        self.backbone = UNetBackbone(
            in_channels=in_channels,
            base_channels=base_channels,
            out_channels=2,
        )
        self.min_log_sigma = float(min_log_sigma)
        self.max_log_sigma = float(max_log_sigma)

    def forward(self, x):
        raw = self.backbone(x)
        mu = F.softplus(raw[:, :1])
        log_sigma = raw[:, 1:2].clamp(
            self.min_log_sigma,
            self.max_log_sigma,
        )
        return mu, log_sigma


def build_model(
    model_name: str,
    in_channels: int = 12,
    base_channels: int = 32,
):
    if model_name == "pixel_mlp":
        return PixelMLP(in_channels=in_channels)
    if model_name == "unet_deterministic":
        return DeterministicUNet(
            in_channels=in_channels,
            base_channels=base_channels,
        )
    if model_name == "unet_gaussian":
        return GaussianUNet(
            in_channels=in_channels,
            base_channels=base_channels,
        )
    raise ValueError(f"Unknown model: {model_name}")


def gaussian_nll(
    y: torch.Tensor,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
) -> torch.Tensor:
    sigma2_inv = torch.exp(-2.0 * log_sigma)
    loss = (
        log_sigma
        + 0.5 * torch.square(y - mu) * sigma2_inv
        + 0.5 * math.log(2.0 * math.pi)
    )
    return loss.mean()


def gaussian_crps(
    y: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """
    Closed-form CRPS for Normal(mu, sigma).

    CRPS = sigma * [z(2Phi(z)-1) + 2phi(z) - 1/sqrt(pi)]
    """
    sigma = sigma.clamp_min(1e-6)
    z = (y - mu) / sigma
    phi = torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    Phi = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    return sigma * (
        z * (2.0 * Phi - 1.0)
        + 2.0 * phi
        - 1.0 / math.sqrt(math.pi)
    )


def normal_exceedance_probability(
    threshold_log1p: float,
    mu: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    sigma = sigma.clamp_min(1e-6)
    z = (threshold_log1p - mu) / sigma
    cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    return (1.0 - cdf).clamp(0.0, 1.0)
