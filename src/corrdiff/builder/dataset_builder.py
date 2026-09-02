"""
CorrDiff dataset builder for ERA5 surface/pressure data and radar cache.

The pressure Parquet dataset is expected in long format: pressure variables
remain columns (for example ``u``, ``v`` and ``t``), while the atmospheric
level is stored in a separate ``pressure_level`` column. Logical CorrDiff
channel names such as ``u_850`` and ``u_500`` are created by this builder.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

from src.config.paths import ERA5_DIR, RADAR_CACHE_DIR


# =============================================================================
# Logging
# =============================================================================


def setup_logger(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("corrdiff.builder")

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        )
        logger.addHandler(handler)

    logger.setLevel(level)
    logger.propagate = False
    return logger


LOGGER = setup_logger()


# =============================================================================
# Utilities
# =============================================================================


def _csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _csv_int_list(value: str | None) -> list[int]:
    return [int(item) for item in _csv_list(value)]


def _normalize_timestamp(value: object) -> pd.Timestamp:
    """Return a timezone-naive UTC timestamp."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def _parse_end_timestamp(value: str) -> pd.Timestamp:
    """Interpret a date-only end value as the end of that day."""
    timestamp = pd.Timestamp(value)
    if len(value.strip()) <= 10:
        timestamp = timestamp + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return timestamp


# =============================================================================
# ERA5 dataset
# =============================================================================


class ERA5Dataset:
    """
    Monthly ERA5 reader that combines surface variables and pressure levels.

    Pressure data are read from one monthly Parquet partition containing a
    pressure-level column. Surface defaults mirror the current ERA5 single-level
    ETL. For a configuration such as::

        surface_variables = ["cape", "cin", "tcwv", "tcc", "z", "tp", "kx", "totalx"]
        pressure_variables = ["u", "v", "r", "q", "t", "w", "crwc"]
        pressure_levels = [850, 500]

    the returned tensor has channels in this order::

        cape, cin, tcwv, tcc, z, tp, kx, totalx,
        u_850, v_850, r_850, q_850, t_850, w_850, crwc_850,
        u_500, v_500, r_500, q_500, t_500, w_500, crwc_500

    The public method :meth:`get_tensor` returns an array with shape
    ``(C, H, W)`` interpolated onto the radar grid.
    """

    TIME_COLUMNS = ("valid_time", "time", "datetime", "date")
    LATITUDE_COLUMNS = ("latitude", "lat")
    LONGITUDE_COLUMNS = ("longitude", "lon")
    PRESSURE_COLUMNS = (
        "pressure_level",
        "level",
        "isobaricinhpa",
        "isobaricInhPa",
        "pressure",
    )

    VARIABLE_ALIASES: dict[str, tuple[str, ...]] = {
        # Current ERA5 single-level ETL
        "cape": ("cape", "convective_available_potential_energy"),
        "cin": ("cin", "convective_inhibition"),
        "tcwv": ("tcwv", "total_column_water_vapour"),
        "tcc": ("tcc", "total_cloud_cover"),
        "z": ("z", "geopotential"),
        "tp": ("tp", "total_precipitation"),
        "kx": ("kx", "k_index"),
        "totalx": ("totalx", "total_totals_index"),

        # Optional/future single-level variables kept for compatibility
        "t2m": ("t2m", "2m_temperature", "2_metre_temperature"),
        "u10": (
            "u10",
            "10m_u_component_of_wind",
            "10_metre_u_component_of_wind",
        ),
        "v10": (
            "v10",
            "10m_v_component_of_wind",
            "10_metre_v_component_of_wind",
        ),
        "sp": ("sp", "surface_pressure"),
        "msl": ("msl", "mean_sea_level_pressure"),
        "d2m": ("d2m", "2m_dewpoint_temperature", "2_metre_dewpoint_temperature"),

        # Pressure levels
        "u": ("u", "u_component_of_wind"),
        "v": ("v", "v_component_of_wind"),
        "r": ("r", "relative_humidity"),
        "q": ("q", "specific_humidity"),
        "t": ("t", "temperature"),
        "w": ("w", "vertical_velocity"),
        "crwc": ("crwc", "specific_rain_water_content"),
    }

    def __init__(
        self,
        era5_root: str | Path,
        surface_variables: Sequence[str],
        pressure_variables: Sequence[str],
        pressure_levels: Sequence[int],
        target_lat: np.ndarray,
        target_lon: np.ndarray,
        logger: logging.Logger | None = None,
    ) -> None:
        self.logger = logger or LOGGER
        self.root = Path(era5_root)
        self.surface_root = self.root / "single"
        self.pressure_root = self.root / "pressure"

        self.surface_variables = [
            self._canonical_variable(name) for name in surface_variables
        ]
        self.pressure_variables = [
            self._canonical_variable(name) for name in pressure_variables
        ]
        self.pressure_levels = sorted(
            {int(level) for level in pressure_levels}, reverse=True
        )

        self.target_lat = np.asarray(target_lat, dtype=np.float64)
        self.target_lon = np.asarray(target_lon, dtype=np.float64)
        if self.target_lat.shape != self.target_lon.shape:
            raise ValueError("target_lat and target_lon must have the same shape")
        if self.target_lat.ndim != 2:
            raise ValueError("target_lat and target_lon must be two-dimensional grids")

        self._target_points = np.column_stack(
            (self.target_lat.ravel(), self.target_lon.ravel())
        )

        self.channels = self._build_channels()

        self.loaded_month: str | None = None
        self.surface_df: pd.DataFrame | None = None
        self.pressure_df: pd.DataFrame | None = None
        self._surface_variable_columns: dict[str, str] = {}
        self._pressure_variable_columns: dict[str, str] = {}
        self._surface_lat = np.empty(0, dtype=np.float64)
        self._surface_lon = np.empty(0, dtype=np.float64)
        self._pressure_lat = np.empty(0, dtype=np.float64)
        self._pressure_lon = np.empty(0, dtype=np.float64)
        self._path_cache: dict[tuple[str, int, int], Path] = {}
        self.last_error: str | None = None

        self.logger.info("ERA5 channels (%d): %s", self.n_channels, self.channels)

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    def _build_channels(self) -> list[str]:
        channels = list(self.surface_variables)
        channels.extend(
            f"{variable}_{level}"
            for level in self.pressure_levels
            for variable in self.pressure_variables
        )
        return channels

    @classmethod
    def _canonical_variable(cls, name: str) -> str:
        normalized = name.strip().lower().replace(" ", "_")
        for canonical, aliases in cls.VARIABLE_ALIASES.items():
            if normalized == canonical or normalized in {
                alias.lower() for alias in aliases
            }:
                return canonical
        return normalized

    @staticmethod
    def _detect_column(
        columns: Iterable[str], candidates: Sequence[str], label: str
    ) -> str:
        column_map = {str(column).lower(): str(column) for column in columns}
        for candidate in candidates:
            match = column_map.get(candidate.lower())
            if match is not None:
                return match
        raise ValueError(
            f"Could not find {label} column. Available columns: {list(columns)}"
        )

    @classmethod
    def _resolve_variable_columns(
        cls, columns: Iterable[str], requested: Sequence[str], dataset_name: str
    ) -> dict[str, str]:
        column_map = {str(column).lower(): str(column) for column in columns}
        resolved: dict[str, str] = {}

        for canonical in requested:
            aliases = cls.VARIABLE_ALIASES.get(canonical, (canonical,))
            candidates = (canonical, *aliases)
            actual = next(
                (column_map[candidate.lower()] for candidate in candidates if candidate.lower() in column_map),
                None,
            )
            if actual is None:
                raise ValueError(
                    f"Variable '{canonical}' was not found in {dataset_name}. "
                    f"Available columns: {list(columns)}"
                )
            resolved[canonical] = actual

        return resolved

    def _collapse_duplicate_rows(
        self,
        dataframe: pd.DataFrame,
        key_columns: Sequence[str],
        value_columns: Sequence[str],
        dataset_name: str,
    ) -> pd.DataFrame:
        """Collapse duplicate coordinates only when values are non-conflicting.

        ERA5 files may expose auxiliary dimensions such as ``expver`` or
        ``number``. After those columns are intentionally discarded, more than
        one row can remain for the same logical coordinate. We do not silently
        average conflicting values. Duplicate groups are accepted only when
        each requested variable has at most one distinct non-null value; in that
        case ``groupby.first`` coalesces complementary NaNs safely.
        """
        duplicate_mask = dataframe.duplicated(list(key_columns), keep=False)
        if not duplicate_mask.any():
            return dataframe

        duplicate_rows = dataframe.loc[duplicate_mask, [*key_columns, *value_columns]]
        grouped = duplicate_rows.groupby(list(key_columns), dropna=False, sort=False)

        conflicts: list[str] = []
        for variable in value_columns:
            distinct_counts = grouped[variable].nunique(dropna=True)
            conflicting = distinct_counts[distinct_counts > 1]
            if not conflicting.empty:
                conflicts.append(f"{variable}: {len(conflicting)} groups")

        if conflicts:
            raise ValueError(
                f"{dataset_name} contains conflicting duplicate coordinate rows "
                f"after dropping auxiliary ERA5 dimensions (for example expver/number). "
                f"Conflicts: {', '.join(conflicts)}"
            )

        duplicate_count = int(duplicate_mask.sum())
        self.logger.warning(
            "%s contains %d duplicate rows after removing auxiliary dimensions; "
            "coalescing non-conflicting values by logical coordinate",
            dataset_name,
            duplicate_count,
        )

        return (
            dataframe.groupby(list(key_columns), as_index=False, dropna=False, sort=False)[
                list(value_columns)
            ]
            .first()
        )

    @staticmethod
    def _month_key(timestamp: pd.Timestamp) -> str:
        return timestamp.strftime("%Y-%m")

    def clear_cache(self) -> None:
        self.loaded_month = None
        self.surface_df = None
        self.pressure_df = None
        self._surface_variable_columns = {}
        self._pressure_variable_columns = {}
        self._surface_lat = np.empty(0, dtype=np.float64)
        self._surface_lon = np.empty(0, dtype=np.float64)
        self._pressure_lat = np.empty(0, dtype=np.float64)
        self._pressure_lon = np.empty(0, dtype=np.float64)
        gc.collect()

    def _candidate_month_paths(
        self, root: Path, timestamp: pd.Timestamp, dataset_kind: str
    ) -> list[Path]:
        year = timestamp.year
        month = timestamp.month
        month_2 = f"{month:02d}"

        candidates = [
            root / f"year={year}" / f"month={month}",
            root / f"year={year}" / f"month={month_2}",
            root / str(year) / month_2,
            root / str(year) / str(month),
        ]

        if dataset_kind == "surface":
            candidates.extend(
                [
                    root / str(year) / month_2 / f"era5_single_{year}_{month_2}.parquet",
                    root / f"era5_single_{year}_{month_2}.parquet",
                ]
            )
        else:
            candidates.extend(
                [
                    root / str(year) / month_2 / f"era5_pressure_{year}_{month_2}.parquet",
                    root / f"era5_pressure_{year}_{month_2}.parquet",
                ]
            )

        return candidates

    @staticmethod
    def _contains_parquet(path: Path) -> bool:
        if path.is_file() and path.suffix.lower() == ".parquet":
            return True
        if path.is_dir():
            try:
                next(path.rglob("*.parquet"))
                return True
            except StopIteration:
                return False
        return False

    def _locate_month_path(
        self, root: Path, timestamp: pd.Timestamp, dataset_kind: str
    ) -> Path:
        key = (dataset_kind, timestamp.year, timestamp.month)
        cached = self._path_cache.get(key)
        if cached is not None and self._contains_parquet(cached):
            return cached

        tried: list[str] = []
        for candidate in self._candidate_month_paths(root, timestamp, dataset_kind):
            tried.append(str(candidate))
            if self._contains_parquet(candidate):
                self._path_cache[key] = candidate
                return candidate

        # Last-resort filename search for non-partitioned local copies.
        month_token = f"{timestamp.month:02d}"
        patterns = (
            f"*{timestamp.year}*{month_token}*.parquet",
            f"*{timestamp.year}*{timestamp.month}*.parquet",
        )
        for pattern in patterns:
            matches = sorted(root.rglob(pattern)) if root.exists() else []
            if matches:
                parent = matches[0].parent
                self._path_cache[key] = parent
                return parent

        raise FileNotFoundError(
            f"No {dataset_kind} Parquet partition found for "
            f"{timestamp:%Y-%m}. Tried: {tried}"
        )

    def _read_month_dataframe(
        self, root: Path, timestamp: pd.Timestamp, dataset_kind: str
    ) -> pd.DataFrame:
        try:
            path = self._locate_month_path(root, timestamp, dataset_kind)
            self.logger.info("Loading ERA5 %s month from %s", dataset_kind, path)
            return pd.read_parquet(path)
        except FileNotFoundError as direct_error:
            # Hive-partition fallback. This handles a dataset whose only stable
            # entry point is the root directory.
            try:
                self.logger.info(
                    "Trying partition-filtered read for ERA5 %s %s",
                    dataset_kind,
                    timestamp.strftime("%Y-%m"),
                )
                return pd.read_parquet(
                    root,
                    filters=[
                        ("year", "==", timestamp.year),
                        ("month", "==", timestamp.month),
                    ],
                )
            except Exception as filtered_error:
                raise FileNotFoundError(
                    f"Unable to read ERA5 {dataset_kind} for {timestamp:%Y-%m}. "
                    f"Direct discovery error: {direct_error}. "
                    f"Filtered read error: {filtered_error}"
                ) from filtered_error

    @staticmethod
    def _normalize_time_series(series: pd.Series) -> pd.Series:
        values = pd.to_datetime(series, errors="coerce", utc=True)
        if values.isna().any():
            raise ValueError("ERA5 time column contains invalid timestamps")
        return values.dt.tz_convert(None)

    def _prepare_surface_month(
        self, dataframe: pd.DataFrame, timestamp: pd.Timestamp
    ) -> None:
        time_column = self._detect_column(
            dataframe.columns, self.TIME_COLUMNS, "surface time"
        )
        latitude_column = self._detect_column(
            dataframe.columns, self.LATITUDE_COLUMNS, "surface latitude"
        )
        longitude_column = self._detect_column(
            dataframe.columns, self.LONGITUDE_COLUMNS, "surface longitude"
        )

        dataframe = dataframe.rename(
            columns={
                time_column: "_time",
                latitude_column: "_latitude",
                longitude_column: "_longitude",
            }
        )
        dataframe["_time"] = self._normalize_time_series(dataframe["_time"])
        dataframe["_latitude"] = pd.to_numeric(dataframe["_latitude"])
        dataframe["_longitude"] = pd.to_numeric(dataframe["_longitude"])

        month_start = timestamp.to_period("M").start_time
        month_end = (timestamp.to_period("M") + 1).start_time
        dataframe = dataframe[
            (dataframe["_time"] >= month_start)
            & (dataframe["_time"] < month_end)
        ]

        self._surface_variable_columns = self._resolve_variable_columns(
            dataframe.columns, self.surface_variables, "surface dataset"
        )
        value_columns = list(self._surface_variable_columns.values())
        keep_columns = [
            "_time",
            "_latitude",
            "_longitude",
            *value_columns,
        ]
        dataframe = dataframe[keep_columns]
        dataframe = self._collapse_duplicate_rows(
            dataframe,
            key_columns=("_time", "_latitude", "_longitude"),
            value_columns=value_columns,
            dataset_name="surface dataset",
        ).sort_values(
            ["_time", "_latitude", "_longitude"], kind="mergesort"
        )
        self._surface_lat = np.sort(dataframe["_latitude"].unique())
        self._surface_lon = np.sort(dataframe["_longitude"].unique())
        self.surface_df = dataframe.set_index("_time", drop=True)

    def _prepare_pressure_month(
        self, dataframe: pd.DataFrame, timestamp: pd.Timestamp
    ) -> None:
        time_column = self._detect_column(
            dataframe.columns, self.TIME_COLUMNS, "pressure time"
        )
        latitude_column = self._detect_column(
            dataframe.columns, self.LATITUDE_COLUMNS, "pressure latitude"
        )
        longitude_column = self._detect_column(
            dataframe.columns, self.LONGITUDE_COLUMNS, "pressure longitude"
        )
        pressure_column = self._detect_column(
            dataframe.columns, self.PRESSURE_COLUMNS, "pressure level"
        )

        dataframe = dataframe.rename(
            columns={
                time_column: "_time",
                latitude_column: "_latitude",
                longitude_column: "_longitude",
                pressure_column: "_pressure_level",
            }
        )
        dataframe["_time"] = self._normalize_time_series(dataframe["_time"])
        dataframe["_latitude"] = pd.to_numeric(dataframe["_latitude"])
        dataframe["_longitude"] = pd.to_numeric(dataframe["_longitude"])
        dataframe["_pressure_level"] = (
            pd.to_numeric(dataframe["_pressure_level"], errors="coerce")
            .round()
            .astype("Int16")
        )
        if dataframe["_pressure_level"].isna().any():
            raise ValueError("Pressure-level column contains invalid values")
        dataframe["_pressure_level"] = dataframe["_pressure_level"].astype(int)

        month_start = timestamp.to_period("M").start_time
        month_end = (timestamp.to_period("M") + 1).start_time
        dataframe = dataframe[
            (dataframe["_time"] >= month_start)
            & (dataframe["_time"] < month_end)
            & (dataframe["_pressure_level"].isin(self.pressure_levels))
        ]

        available_levels = sorted(dataframe["_pressure_level"].unique().tolist())
        missing_levels = sorted(set(self.pressure_levels) - set(available_levels))
        if missing_levels:
            raise ValueError(
                f"Requested pressure levels are missing from {timestamp:%Y-%m}: "
                f"{missing_levels}. Available levels: {available_levels}"
            )

        self._pressure_variable_columns = self._resolve_variable_columns(
            dataframe.columns, self.pressure_variables, "pressure dataset"
        )
        value_columns = list(self._pressure_variable_columns.values())
        keep_columns = [
            "_time",
            "_pressure_level",
            "_latitude",
            "_longitude",
            *value_columns,
        ]
        dataframe = dataframe[keep_columns]
        dataframe = self._collapse_duplicate_rows(
            dataframe,
            key_columns=("_time", "_pressure_level", "_latitude", "_longitude"),
            value_columns=value_columns,
            dataset_name="pressure dataset",
        ).sort_values(
            ["_time", "_pressure_level", "_latitude", "_longitude"],
            kind="mergesort",
        )
        self._pressure_lat = np.sort(dataframe["_latitude"].unique())
        self._pressure_lon = np.sort(dataframe["_longitude"].unique())
        self.pressure_df = dataframe.set_index(
            ["_time", "_pressure_level"], drop=True
        )

    def load_month(self, timestamp: object) -> None:
        timestamp = _normalize_timestamp(timestamp)
        month_key = self._month_key(timestamp)
        if month_key == self.loaded_month:
            return

        self.clear_cache()
        self.logger.info("Loading ERA5 month %s", month_key)

        if self.surface_variables:
            surface = self._read_month_dataframe(
                self.surface_root, timestamp, "surface"
            )
            self._prepare_surface_month(surface, timestamp)

        if self.pressure_variables and self.pressure_levels:
            pressure = self._read_month_dataframe(
                self.pressure_root, timestamp, "pressure"
            )
            self._prepare_pressure_month(pressure, timestamp)

        self.loaded_month = month_key
        gc.collect()

    @staticmethod
    def _select_index_frame(
        dataframe: pd.DataFrame | None, key: object
    ) -> pd.DataFrame | None:
        if dataframe is None:
            return None
        try:
            selected = dataframe.loc[key]
        except KeyError:
            return None
        if isinstance(selected, pd.Series):
            selected = selected.to_frame().T
        return selected

    @staticmethod
    def _select_pressure_frame(
        dataframe: pd.DataFrame | None, timestamp: pd.Timestamp, level: int
    ) -> pd.DataFrame | None:
        if dataframe is None:
            return None
        try:
            selected = dataframe.xs(
                (timestamp, level),
                level=("_time", "_pressure_level"),
                drop_level=True,
            )
        except KeyError:
            return None
        if isinstance(selected, pd.Series):
            selected = selected.to_frame().T
        return selected

    @staticmethod
    def _frame_to_cube(
        frame: pd.DataFrame,
        variable_columns: Sequence[str],
        latitudes: np.ndarray,
        longitudes: np.ndarray,
    ) -> np.ndarray:
        expected_size = len(latitudes) * len(longitudes)
        ordered = frame.sort_values(
            ["_latitude", "_longitude"], kind="mergesort"
        )

        coordinate_duplicates = ordered.duplicated(
            ["_latitude", "_longitude"]
        ).any()
        if len(ordered) == expected_size and not coordinate_duplicates:
            data = ordered[list(variable_columns)].to_numpy(dtype=np.float32)
            return data.reshape(len(latitudes), len(longitudes), len(variable_columns))

        # Robust fallback for duplicated/missing coordinate rows.
        fields = []
        for variable in variable_columns:
            grid = ordered.pivot_table(
                index="_latitude",
                columns="_longitude",
                values=variable,
                aggfunc="mean",
            ).reindex(index=latitudes, columns=longitudes)
            fields.append(grid.to_numpy(dtype=np.float32))
        return np.stack(fields, axis=-1)

    def _interpolate_cube(
        self,
        cube: np.ndarray,
        latitudes: np.ndarray,
        longitudes: np.ndarray,
    ) -> np.ndarray:
        if len(latitudes) < 2 or len(longitudes) < 2:
            raise ValueError("ERA5 source grid must contain at least 2x2 points")

        interpolator = RegularGridInterpolator(
            (latitudes, longitudes),
            cube,
            bounds_error=False,
            fill_value=np.nan,
        )
        interpolated = interpolator(self._target_points)
        interpolated = interpolated.reshape(
            *self.target_lat.shape, cube.shape[-1]
        )
        return np.moveaxis(interpolated, -1, 0).astype(np.float32, copy=False)

    def get_tensor(self, timestamp: object) -> np.ndarray | None:
        """Return all configured ERA5 channels as ``(C, H, W)``."""
        timestamp = _normalize_timestamp(timestamp)
        self.last_error = None

        try:
            self.load_month(timestamp)
            channel_groups: list[np.ndarray] = []

            if self.surface_variables:
                surface_frame = self._select_index_frame(self.surface_df, timestamp)
                if surface_frame is None or surface_frame.empty:
                    self.last_error = f"Missing surface timestamp {timestamp}"
                    return None

                surface_columns = [
                    self._surface_variable_columns[name]
                    for name in self.surface_variables
                ]
                surface_cube = self._frame_to_cube(
                    surface_frame,
                    surface_columns,
                    self._surface_lat,
                    self._surface_lon,
                )
                channel_groups.append(
                    self._interpolate_cube(
                        surface_cube, self._surface_lat, self._surface_lon
                    )
                )

            if self.pressure_variables and self.pressure_levels:
                pressure_columns = [
                    self._pressure_variable_columns[name]
                    for name in self.pressure_variables
                ]
                for level in self.pressure_levels:
                    pressure_frame = self._select_pressure_frame(
                        self.pressure_df, timestamp, level
                    )
                    if pressure_frame is None or pressure_frame.empty:
                        self.last_error = (
                            f"Missing pressure timestamp {timestamp} at {level} hPa"
                        )
                        return None

                    pressure_cube = self._frame_to_cube(
                        pressure_frame,
                        pressure_columns,
                        self._pressure_lat,
                        self._pressure_lon,
                    )
                    channel_groups.append(
                        self._interpolate_cube(
                            pressure_cube, self._pressure_lat, self._pressure_lon
                        )
                    )

            if not channel_groups:
                raise ValueError("No ERA5 variables were configured")

            tensor = np.concatenate(channel_groups, axis=0)
            if tensor.shape[0] != self.n_channels:
                raise RuntimeError(
                    f"Unexpected ERA5 channel count: {tensor.shape[0]} "
                    f"instead of {self.n_channels}"
                )
            return tensor.astype(np.float32, copy=False)

        except Exception as error:
            self.last_error = str(error)
            self.logger.warning("ERA5 failed for %s: %s", timestamp, error)
            return None

    def summary(self) -> None:
        self.logger.info("=" * 72)
        self.logger.info("ERA5 DATASET")
        self.logger.info("Root               : %s", self.root)
        self.logger.info("Surface variables  : %s", self.surface_variables)
        self.logger.info("Pressure variables : %s", self.pressure_variables)
        self.logger.info("Pressure levels    : %s", self.pressure_levels)
        self.logger.info("Channels           : %s", self.channels)
        self.logger.info("Target grid        : %s", self.target_lat.shape)
        self.logger.info("=" * 72)


# =============================================================================
# Radar cache dataset
# =============================================================================


class RadarDataset:
    """Read preprocessed radar grids from ``YYYYMMDD_HH_MM.npy`` files."""

    def __init__(
        self,
        cache_dir: str | Path,
        lat_range: Sequence[float],
        lon_range: Sequence[float],
        resolution_km: float,
        logger: logging.Logger | None = None,
    ) -> None:
        self.logger = logger or LOGGER
        self.cache_dir = Path(cache_dir)
        self.lat_range = tuple(float(value) for value in lat_range)
        self.lon_range = tuple(float(value) for value in lon_range)
        self.resolution_km = float(resolution_km)

        if not self.cache_dir.exists():
            raise FileNotFoundError(f"Radar cache directory not found: {self.cache_dir}")

        degree_resolution = self.resolution_km / 111.0
        self.lat = np.arange(
            self.lat_range[0], self.lat_range[1], degree_resolution, dtype=np.float64
        )
        self.lon = np.arange(
            self.lon_range[0], self.lon_range[1], degree_resolution, dtype=np.float64
        )
        self.Lon, self.Lat = np.meshgrid(self.lon, self.lat)
        self.logger.info("Radar grid shape: %s", self.shape)

    @property
    def shape(self) -> tuple[int, int]:
        return self.Lat.shape

    def cache_path(self, timestamp: object) -> Path:
        timestamp = _normalize_timestamp(timestamp)
        return self.cache_dir / f"{timestamp:%Y%m%d_%H_%M}.npy"

    def exists(self, timestamp: object) -> bool:
        return self.cache_path(timestamp).exists()

    def get_grid(self, timestamp: object) -> np.ndarray | None:
        path = self.cache_path(timestamp)
        if not path.exists():
            return None
        try:
            grid = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
        except Exception as error:
            self.logger.warning("Could not read radar cache %s: %s", path, error)
            return None
        if grid.shape != self.shape:
            self.logger.warning(
                "Radar shape mismatch for %s: %s instead of %s",
                path.name,
                grid.shape,
                self.shape,
            )
            return None
        return grid

    def summary(self) -> None:
        self.logger.info("=" * 72)
        self.logger.info("RADAR DATASET")
        self.logger.info("Cache directory : %s", self.cache_dir)
        self.logger.info("Grid shape      : %s", self.shape)
        self.logger.info("Latitude range  : %.4f to %.4f", self.lat.min(), self.lat.max())
        self.logger.info("Longitude range : %.4f to %.4f", self.lon.min(), self.lon.max())
        self.logger.info("Resolution      : %.2f km", self.resolution_km)
        self.logger.info("=" * 72)


# =============================================================================
# CorrDiff dataset builder
# =============================================================================


class CorrDiffDatasetBuilder:
    """Create patch-based Zarr data for CorrDiff regression and diffusion."""

    def __init__(
        self,
        era5: ERA5Dataset,
        radar: RadarDataset,
        output_dir: str | Path,
        start_date: object,
        end_date: object,
        time_frequency: str = "1h",
        patch_size: int = 32,
        stride: int = 16,
        chunk_size: int = 256,
        write_batch_size: int = 2048,
        minimum_radar_valid_ratio: float = 0.05,
        minimum_input_valid_ratio: float = 1.0,
        overwrite: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        self.logger = logger or LOGGER
        self.era5 = era5
        self.radar = radar
        self.output_dir = Path(output_dir)
        self.start_date = _normalize_timestamp(start_date)
        self.end_date = _normalize_timestamp(end_date)
        self.time_frequency = time_frequency
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.chunk_size = int(chunk_size)
        self.write_batch_size = int(write_batch_size)
        self.minimum_radar_valid_ratio = float(minimum_radar_valid_ratio)
        self.minimum_input_valid_ratio = float(minimum_input_valid_ratio)
        self.overwrite = overwrite

        if self.start_date > self.end_date:
            raise ValueError("start_date must not be after end_date")
        if self.patch_size <= 0 or self.stride <= 0:
            raise ValueError("patch_size and stride must be positive")
        if self.patch_size > min(self.radar.shape):
            raise ValueError(
                f"patch_size {self.patch_size} exceeds radar grid {self.radar.shape}"
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.zarr_path = self.output_dir / "train.zarr"
        self.stats_dir = self.output_dir / "stats"

        self.root = None
        self.input_ds = None
        self.target_ds = None
        self.mask_ds = None
        self.timestamps_ds = None
        self.patch_rows_ds = None
        self.patch_cols_ds = None

        self.sample_count = 0
        self.written_count = 0
        self.input_sum = np.zeros(self.era5.n_channels, dtype=np.float64)
        self.input_sq_sum = np.zeros(self.era5.n_channels, dtype=np.float64)
        self.input_count = np.zeros(self.era5.n_channels, dtype=np.int64)
        self.target_sum = 0.0
        self.target_sq_sum = 0.0
        self.target_count = 0

        self._buffer_x: list[np.ndarray] = []
        self._buffer_y: list[np.ndarray] = []
        self._buffer_mask: list[np.ndarray] = []
        self._buffer_timestamp: list[int] = []
        self._buffer_patch_row: list[int] = []
        self._buffer_patch_col: list[int] = []

        self.counters = {
            "timestamps_total": 0,
            "timestamps_processed": 0,
            "missing_radar": 0,
            "invalid_radar": 0,
            "missing_era5": 0,
            "shape_mismatch": 0,
            "patches_rejected_radar": 0,
            "patches_rejected_input": 0,
            "patches_written": 0,
        }

        self.compressor = Blosc(cname="zstd", clevel=3, shuffle=2)

    def _timestamps(self) -> pd.DatetimeIndex:
        return pd.date_range(
            self.start_date, self.end_date, freq=self.time_frequency
        )

    def _prepare_output(self) -> None:
        if self.zarr_path.exists():
            if not self.overwrite:
                raise FileExistsError(
                    f"Output already exists: {self.zarr_path}. Use --overwrite."
                )
            shutil.rmtree(self.zarr_path)

        self.root = zarr.open(
            str(self.zarr_path), mode="w", zarr_version=2
        )
        spatial_shape = (self.patch_size, self.patch_size)

        self.input_ds = self.root.create_dataset(
            "input",
            shape=(0, self.era5.n_channels, *spatial_shape),
            chunks=(self.chunk_size, self.era5.n_channels, *spatial_shape),
            dtype="float32",
            compressor=self.compressor,
        )
        self.target_ds = self.root.create_dataset(
            "target",
            shape=(0, 1, *spatial_shape),
            chunks=(self.chunk_size, 1, *spatial_shape),
            dtype="float32",
            compressor=self.compressor,
        )
        self.mask_ds = self.root.create_dataset(
            "mask",
            shape=(0, 1, *spatial_shape),
            chunks=(self.chunk_size, 1, *spatial_shape),
            dtype="float32",
            compressor=self.compressor,
        )
        self.timestamps_ds = self.root.create_dataset(
            "timestamps",
            shape=(0,),
            chunks=(self.chunk_size,),
            dtype="int64",
            compressor=self.compressor,
        )
        self.patch_rows_ds = self.root.create_dataset(
            "patch_row",
            shape=(0,),
            chunks=(self.chunk_size,),
            dtype="int16",
            compressor=self.compressor,
        )
        self.patch_cols_ds = self.root.create_dataset(
            "patch_col",
            shape=(0,),
            chunks=(self.chunk_size,),
            dtype="int16",
            compressor=self.compressor,
        )

        self.root.attrs.update(
            {
                "channels": self.era5.channels,
                "surface_variables": self.era5.surface_variables,
                "pressure_variables": self.era5.pressure_variables,
                "pressure_levels_hpa": self.era5.pressure_levels,
                "target_transform": "log1p",
                "patch_size": self.patch_size,
                "stride": self.stride,
                "radar_resolution_km": self.radar.resolution_km,
            }
        )

    def _append_patch(
        self,
        x: np.ndarray,
        y: np.ndarray,
        mask: np.ndarray,
        timestamp: pd.Timestamp,
        patch_row: int,
        patch_col: int,
    ) -> None:
        self._buffer_x.append(x.astype(np.float32, copy=False))
        self._buffer_y.append(y.astype(np.float32, copy=False))
        self._buffer_mask.append(mask.astype(np.float32, copy=False))
        self._buffer_timestamp.append(int(timestamp.timestamp()))
        self._buffer_patch_row.append(patch_row)
        self._buffer_patch_col.append(patch_col)

        finite_input = np.isfinite(x)
        x64 = np.where(finite_input, x, 0.0).astype(np.float64, copy=False)
        self.input_sum += x64.sum(axis=(1, 2))
        self.input_sq_sum += np.square(x64).sum(axis=(1, 2))
        self.input_count += finite_input.sum(axis=(1, 2))

        valid_target = mask.astype(bool)
        target_values = y[valid_target]
        self.target_sum += float(target_values.sum(dtype=np.float64))
        self.target_sq_sum += float(
            np.square(target_values.astype(np.float64)).sum(dtype=np.float64)
        )
        self.target_count += int(target_values.size)

        self.sample_count += 1
        self.counters["patches_written"] += 1
        if len(self._buffer_x) >= self.write_batch_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer_x:
            return

        x_batch = np.stack(self._buffer_x)
        y_batch = np.stack(self._buffer_y)
        mask_batch = np.stack(self._buffer_mask)
        timestamp_batch = np.asarray(self._buffer_timestamp, dtype=np.int64)
        row_batch = np.asarray(self._buffer_patch_row, dtype=np.int16)
        col_batch = np.asarray(self._buffer_patch_col, dtype=np.int16)

        start = self.written_count
        end = start + len(x_batch)

        self.input_ds.resize((end, *self.input_ds.shape[1:]))
        self.target_ds.resize((end, *self.target_ds.shape[1:]))
        self.mask_ds.resize((end, *self.mask_ds.shape[1:]))
        self.timestamps_ds.resize((end,))
        self.patch_rows_ds.resize((end,))
        self.patch_cols_ds.resize((end,))

        self.input_ds[start:end] = x_batch
        self.target_ds[start:end] = y_batch
        self.mask_ds[start:end] = mask_batch
        self.timestamps_ds[start:end] = timestamp_batch
        self.patch_rows_ds[start:end] = row_batch
        self.patch_cols_ds[start:end] = col_batch

        self.written_count = end
        self._buffer_x.clear()
        self._buffer_y.clear()
        self._buffer_mask.clear()
        self._buffer_timestamp.clear()
        self._buffer_patch_row.clear()
        self._buffer_patch_col.clear()

    def _process_patches(
        self, x: np.ndarray, radar_grid: np.ndarray, timestamp: pd.Timestamp
    ) -> int:
        height, width = radar_grid.shape
        created = 0

        for row in range(0, height - self.patch_size + 1, self.stride):
            for col in range(0, width - self.patch_size + 1, self.stride):
                x_patch = x[
                    :,
                    row : row + self.patch_size,
                    col : col + self.patch_size,
                ]
                radar_patch = radar_grid[
                    row : row + self.patch_size,
                    col : col + self.patch_size,
                ]

                radar_mask = np.isfinite(radar_patch)
                if float(radar_mask.mean()) < self.minimum_radar_valid_ratio:
                    self.counters["patches_rejected_radar"] += 1
                    continue

                input_valid_ratio = float(np.isfinite(x_patch).mean())
                if input_valid_ratio < self.minimum_input_valid_ratio:
                    self.counters["patches_rejected_input"] += 1
                    continue

                radar_filled = np.nan_to_num(
                    radar_patch, nan=0.0, posinf=0.0, neginf=0.0
                )
                radar_filled = np.clip(radar_filled, 0.0, None)
                target = np.log1p(radar_filled)[None, ...].astype(np.float32)
                mask = radar_mask[None, ...].astype(np.float32)

                self._append_patch(
                    x_patch,
                    target,
                    mask,
                    timestamp,
                    row,
                    col,
                )
                created += 1

        return created

    def _save_statistics(self) -> None:
        if self.sample_count == 0:
            raise RuntimeError("No samples were generated; statistics cannot be saved")
        if np.any(self.input_count == 0) or self.target_count == 0:
            raise RuntimeError("Normalization statistics contain empty channels")

        self.stats_dir.mkdir(parents=True, exist_ok=True)

        input_mean = self.input_sum / self.input_count
        input_variance = np.maximum(
            self.input_sq_sum / self.input_count - np.square(input_mean), 0.0
        )
        input_std = np.sqrt(input_variance)
        target_mean = self.target_sum / self.target_count
        target_variance = max(
            self.target_sq_sum / self.target_count - target_mean**2, 0.0
        )
        target_std = np.sqrt(target_variance)

        input_std[input_std == 0] = 1.0
        if target_std == 0:
            target_std = 1.0

        input_mean_f32 = input_mean.astype(np.float32)
        input_std_f32 = input_std.astype(np.float32)
        target_mean_f32 = np.asarray([target_mean], dtype=np.float32)
        target_std_f32 = np.asarray([target_std], dtype=np.float32)

        np.save(self.stats_dir / "input_mean.npy", input_mean_f32)
        np.save(self.stats_dir / "input_std.npy", input_std_f32)
        np.save(self.stats_dir / "target_mean.npy", target_mean_f32)
        np.save(self.stats_dir / "target_std.npy", target_std_f32)
        np.savez(
            self.output_dir / "normalization.npz",
            input_mean=input_mean_f32,
            input_std=input_std_f32,
            target_mean=target_mean_f32,
            target_std=target_std_f32,
        )

        metadata = {
            "num_samples": self.sample_count,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "time_frequency": self.time_frequency,
            "patch_size": self.patch_size,
            "stride": self.stride,
            "input_channels": self.era5.n_channels,
            "channels": self.era5.channels,
            "surface_variables": self.era5.surface_variables,
            "pressure_variables": self.era5.pressure_variables,
            "pressure_levels_hpa": self.era5.pressure_levels,
            "radar_grid_shape": list(self.radar.shape),
            "radar_resolution_km": self.radar.resolution_km,
            "target_transform": "log1p",
            "counters": self.counters,
        }
        with (self.output_dir / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)

    def build(self) -> None:
        self._prepare_output()
        timestamps = self._timestamps()
        self.counters["timestamps_total"] = len(timestamps)

        self.logger.info("=" * 72)
        self.logger.info("CORRDIFF DATASET BUILD")
        self.logger.info("Period      : %s to %s", self.start_date, self.end_date)
        self.logger.info("Timestamps  : %d", len(timestamps))
        self.logger.info("Channels    : %d", self.era5.n_channels)
        self.logger.info("Output      : %s", self.zarr_path)
        self.logger.info("=" * 72)

        for timestamp in tqdm(timestamps, desc="Building CorrDiff dataset"):
            timestamp = _normalize_timestamp(timestamp)

            if not self.radar.exists(timestamp):
                self.counters["missing_radar"] += 1
                continue

            radar_grid = self.radar.get_grid(timestamp)
            if radar_grid is None or not np.isfinite(radar_grid).any():
                self.counters["invalid_radar"] += 1
                continue

            era5_tensor = self.era5.get_tensor(timestamp)
            if era5_tensor is None:
                self.counters["missing_era5"] += 1
                continue

            if era5_tensor.shape[1:] != radar_grid.shape:
                self.counters["shape_mismatch"] += 1
                self.logger.warning(
                    "Spatial mismatch at %s: ERA5 %s, radar %s",
                    timestamp,
                    era5_tensor.shape,
                    radar_grid.shape,
                )
                continue

            created = self._process_patches(era5_tensor, radar_grid, timestamp)
            if created > 0:
                self.counters["timestamps_processed"] += 1

        self._flush()
        self._save_statistics()

        self.logger.info("=" * 72)
        self.logger.info("DATASET COMPLETED")
        self.logger.info("Samples written : %d", self.written_count)
        for name, value in self.counters.items():
            self.logger.info("%-26s: %d", name, value)
        self.logger.info("=" * 72)


# =============================================================================
# CLI
# =============================================================================


def parameter_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a CorrDiff Zarr dataset from ERA5 surface/pressure Parquet "
            "partitions and preprocessed radar NPY grids."
        )
    )
    parser.add_argument("-b", "--begin", required=True, help="Start date/time")
    parser.add_argument("-e", "--end", required=True, help="End date/time")
    parser.add_argument(
        "--era5_root", type=Path, default=Path(ERA5_DIR), help="ERA5 root directory"
    )
    parser.add_argument(
        "--radar_cache",
        type=Path,
        default=Path(RADAR_CACHE_DIR),
        help="Directory containing YYYYMMDD_HH_MM.npy radar grids",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("datasets/corrdiff")
    )
    parser.add_argument(
        "--surface_variables",
        default="cape,cin,tcwv,tcc,z,tp,kx,totalx",
        help=(
            "Comma-separated ERA5 single-level variables. Default matches the "
            "current etl_era5_single pipeline."
        ),
    )
    parser.add_argument(
        "--pressure_variables",
        default="u,v,r,q,t,w,crwc",
        help="Comma-separated ERA5 pressure-level variables",
    )
    parser.add_argument("--pressure_levels", default="850,500")
    parser.add_argument(
        "--lat_range", nargs=2, type=float, default=[-23.5, -22.25]
    )
    parser.add_argument(
        "--lon_range", nargs=2, type=float, default=[-44.0, -42.5]
    )
    parser.add_argument("--radar_res", type=float, default=2.0)
    parser.add_argument("--time_frequency", default="1h")
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--write_batch_size", type=int, default=2048)
    parser.add_argument("--minimum_radar_valid_ratio", type=float, default=0.05)
    parser.add_argument("--minimum_input_valid_ratio", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    logger = setup_logger(logging.DEBUG if args.debug else logging.INFO)

    radar = RadarDataset(
        cache_dir=args.radar_cache,
        lat_range=args.lat_range,
        lon_range=args.lon_range,
        resolution_km=args.radar_res,
        logger=logger,
    )

    era5 = ERA5Dataset(
        era5_root=args.era5_root,
        surface_variables=_csv_list(args.surface_variables),
        pressure_variables=_csv_list(args.pressure_variables),
        pressure_levels=_csv_int_list(args.pressure_levels),
        target_lat=radar.Lat,
        target_lon=radar.Lon,
        logger=logger,
    )

    radar.summary()
    era5.summary()

    builder = CorrDiffDatasetBuilder(
        era5=era5,
        radar=radar,
        output_dir=args.output_dir,
        start_date=pd.Timestamp(args.begin),
        end_date=_parse_end_timestamp(args.end),
        time_frequency=args.time_frequency,
        patch_size=args.patch_size,
        stride=args.stride,
        chunk_size=args.chunk_size,
        write_batch_size=args.write_batch_size,
        minimum_radar_valid_ratio=args.minimum_radar_valid_ratio,
        minimum_input_valid_ratio=args.minimum_input_valid_ratio,
        overwrite=args.overwrite,
        logger=logger,
    )
    builder.build()


if __name__ == "__main__":
    main(parameter_parser())
