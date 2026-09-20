"""Shared utilities for CorrDiff Phase 16 v2."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import zarr
except ImportError as exc:
    raise SystemExit("Phase 16 v2 requires zarr.") from exc


def open_group(path: Path):
    try:
        return zarr.open_group(str(path), mode="r")
    except Exception:
        return zarr.open(str(path), mode="r")


def zarr_take_first_axis(arr, indices: np.ndarray, tail_selection=None):
    """Zarr v2/v3 compatible first-axis gather using orthogonal indexing."""
    idx = np.asarray(indices, dtype=np.int64)
    if idx.ndim != 1:
        raise ValueError("indices must be 1-D")

    if tail_selection is None:
        tail_selection = tuple(
            slice(None) for _ in range(arr.ndim - 1)
        )

    selection = (idx,) + tuple(tail_selection)

    if hasattr(arr, "oindex"):
        return np.asarray(arr.oindex[selection])

    if hasattr(arr, "get_orthogonal_selection"):
        return np.asarray(arr.get_orthogonal_selection(selection))

    raise RuntimeError(
        "Zarr implementation does not expose orthogonal indexing."
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
