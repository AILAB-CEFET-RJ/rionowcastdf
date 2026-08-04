from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path
from typing import Dict, List, Optional

from pathlib import Path
import numpy as np
import pandas as pd

from scipy.interpolate import RegularGridInterpolator
import tqdm
import zarr

from src.config.paths import ERA5_DIR, RADAR_CACHE_DIR, RADAR_DIR

logger = logging.getLogger(__name__)
# =============================================================================
# LOGGER
# =============================================================================

def setup_logger():

    logger = logging.getLogger("corrdiff")

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s"
    )

    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger

# =============================================================================
# ERA5 DATASET
# =============================================================================
class ERA5Dataset:

    # -------------------------------------------------------------------------
    # Constructor
    # -------------------------------------------------------------------------

    def __init__(
        self,
        era5_root: str | Path,
        surface_variables: List[str],
        pressure_variables: List[str],
        pressure_levels: List[int],
        lat_range,
        lon_range,
        target_lat,
        target_lon,
    ):

        self.root = Path(era5_root)

        self.surface_root = self.root / "single"
        self.pressure_root = self.root / "pressure"

        self.surface_variables = list(surface_variables)

        self.pressure_variables = list(pressure_variables)

        self.pressure_levels = sorted(
            [int(x) for x in pressure_levels],
            reverse=True,
        )

        self.lat_range = lat_range
        self.lon_range = lon_range

        self.target_lat = target_lat
        self.target_lon = target_lon

        # --------------------------------------------------------------
        # Monthly cache
        # --------------------------------------------------------------

        self.loaded_month: Optional[str] = None

        self.surface_df: Optional[pd.DataFrame] = None

        self.pressure_df: Dict[int, pd.DataFrame] = {}

        # --------------------------------------------------------------
        # Build CorrDiff channels
        # --------------------------------------------------------------

        self.channels = self._build_channel_list()

        logger.info("========================================")
        logger.info("ERA5 Dataset initialized")
        logger.info("Surface variables : %s", self.surface_variables)
        logger.info("Pressure variables: %s", self.pressure_variables)
        logger.info("Pressure levels   : %s", self.pressure_levels)
        logger.info("Total channels    : %d", len(self.channels))
        logger.info("========================================")

    # -------------------------------------------------------------------------
    # Channels
    # -------------------------------------------------------------------------

    def _build_channel_list(self) -> List[str]:
        """
        Builds CorrDiff channel names.

        Example

        Surface

            t2m
            sp
            u10
            v10

        Pressure

            u_850
            v_850
            t_850
            q_850

            u_500
            v_500
            t_500
            q_500
        """

        channels = []

        channels.extend(self.surface_variables)

        for level in self.pressure_levels:

            for variable in self.pressure_variables:

                channels.append(
                    f"{variable}_{level}"
                )

        return channels

    @property
    def n_channels(self):

        return len(self.channels)

    # -------------------------------------------------------------------------
    # Cache
    # -------------------------------------------------------------------------

    def clear_cache(self):

        self.loaded_month = None

        self.surface_df = None

        self.pressure_df = {}

        gc.collect()

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def month_key(timestamp: pd.Timestamp) -> str:

        return timestamp.strftime("%Y-%m")
    
    
    # -------------------------------------------------------------------------
    # Carrega um mês inteiro
    # -------------------------------------------------------------------------

    def load_month(self, timestamp):

        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.Timestamp(timestamp)

        month = self.month_key(timestamp)

        if month == self.loaded_month:
            return

        logger.info(
            "Loading ERA5 month %s",
            month
        )

        self.clear_cache()

        self._load_surface(timestamp)

        self._load_pressure(timestamp)

        self.loaded_month = month

        gc.collect()

    # -------------------------------------------------------------------------
    # File discovery
    # -------------------------------------------------------------------------

    def surface_file(
        self,
        timestamp: pd.Timestamp,
    ) -> Path:

        year = timestamp.strftime("%Y")

        month = timestamp.strftime("%m")

        return (
            self.surface_root
            / year
            / month
            / f"era5_single_{year}_{month}.parquet"
        )

    def pressure_file(
        self,
        timestamp: pd.Timestamp,
        level: int,
    ) -> Path:

        year = timestamp.strftime("%Y")

        month = timestamp.strftime("%m")

        return (
            self.pressure_root
            / year
            / month
            / (
                f"era5_pressure_"
                f"{level}hPa_"
                f"{year}_{month}.parquet"
            )
        )
        
    # -------------------------------------------------------------------------
    # Recupera todos os dados de um timestamp
    # -------------------------------------------------------------------------

    def get_timestamp(self, timestamp):

        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.Timestamp(timestamp)

        surface = self.surface_row(timestamp)

        if surface is None:
            return None

        pressure = self.pressure_rows(timestamp)

        return {
            "surface": surface,
            "pressure": pressure
        }
        
    # -------------------------------------------------------------------------
    # Interpola uma variável para a grade do radar
    # -------------------------------------------------------------------------

    def _interpolate_variable(
        self,
        lat,
        lon,
        values,
    ):

        interp = self._build_interpolator(
            lat,
            lon,
            values,
        )

        pts = np.column_stack(

            (
                self.target_lat.ravel(),
                self.target_lon.ravel(),
            )

        )

        field = interp(pts)

        return field.reshape(
            self.target_lat.shape
        ).astype(np.float32)
        
        
    # -------------------------------------------------------------------------
    # Status do cache
    # -------------------------------------------------------------------------

    def cache_status(self):

        logger.info("")
        logger.info("=" * 60)
        logger.info("ERA5 CACHE STATUS")
        logger.info("=" * 60)

        logger.info(
            "Loaded month : %s",
            self.loaded_month
        )

        if self.surface_df is None:

            logger.info("Surface : not loaded")

        else:

            logger.info(
                "Surface rows : %d",
                len(self.surface_df)
            )

        for level in self.pressure_levels:

            if level not in self.pressure_df:

                logger.info(
                    "%dhPa : not loaded",
                    level
                )

                continue

            logger.info(
                "%dhPa rows : %d",
                level,
                len(self.pressure_df[level])
            )

        logger.info("=" * 60)


    # -------------------------------------------------------------------------
    # Information
    # -------------------------------------------------------------------------

    def summary(self):

        logger.info("")
        logger.info("========================================")
        logger.info("ERA5 DATASET")
        logger.info("========================================")

        logger.info("Root:")
        logger.info(self.root)

        logger.info("")

        logger.info("Surface variables")

        for variable in self.surface_variables:

            logger.info("   %s", variable)

        logger.info("")

        logger.info("Pressure variables")

        for variable in self.pressure_variables:

            logger.info("   %s", variable)

        logger.info("")

        logger.info("Pressure levels")

        for level in self.pressure_levels:

            logger.info("   %d hPa", level)

        logger.info("")

        logger.info("Channels")

        for i, channel in enumerate(self.channels):

            logger.info(
                "%02d  %s",
                i,
                channel,
            )

        logger.info("")
        logger.info("Total channels: %d", self.n_channels)
        logger.info("========================================")
        
    # -------------------------------------------------------------------------
    # Carrega arquivo de superfície
    # -------------------------------------------------------------------------

    def _load_surface(self, timestamp: pd.Timestamp):

        file = self.surface_file(timestamp)

        if not file.exists():
            raise FileNotFoundError(file)

        logger.info(
            "Loading surface: %s",
            file.name
        )

        df = pd.read_parquet(file)

        df = self._normalize_time(df)

        self.surface_df = df

    # -------------------------------------------------------------------------
    # Carrega todos os níveis de pressão
    # -------------------------------------------------------------------------

    def _load_pressure(self, timestamp: pd.Timestamp):

        self.pressure_df = {}

        for level in self.pressure_levels:

            file = self.pressure_file(
                timestamp,
                level
            )

            if not file.exists():
                raise FileNotFoundError(file)

            logger.info(
                "Loading pressure %dhPa",
                level
            )

            df = pd.read_parquet(file)

            df = self._normalize_time(df)

            self.pressure_df[level] = df



    # -------------------------------------------------------------------------
    # Linha da superfície
    # -------------------------------------------------------------------------

    def surface_row(self, timestamp):

        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.Timestamp(timestamp)

        self.load_month(timestamp)

        row = self.surface_df.loc[
            self.surface_df.time == timestamp
        ]

        if row.empty:
            return None

        return row.iloc[0]

    # -------------------------------------------------------------------------
    # Linhas dos níveis de pressão
    # -------------------------------------------------------------------------

    def pressure_rows(self, timestamp):

        if not isinstance(timestamp, pd.Timestamp):
            timestamp = pd.Timestamp(timestamp)

        self.load_month(timestamp)

        rows = {}

        for level, df in self.pressure_df.items():

            row = df.loc[
                df.time == timestamp
            ]

            if row.empty:
                rows[level] = None
            else:
                rows[level] = row.iloc[0]

        return rows

    # -------------------------------------------------------------------------
    # Descobre automaticamente os nomes das coordenadas
    # -------------------------------------------------------------------------

    @staticmethod
    def _latitude_name(df):

        for candidate in ["latitude", "lat"]:

            if candidate in df.columns:
                return candidate

        raise ValueError("Latitude column not found.")

    @staticmethod
    def _longitude_name(df):

        for candidate in ["longitude", "lon"]:

            if candidate in df.columns:
                return candidate

        raise ValueError("Longitude column not found.")


   
    # -------------------------------------------------------------------------
    # Cria um interpolador para uma variável
    # -------------------------------------------------------------------------

    def _build_interpolator(
        self,
        lat,
        lon,
        values,
    ):

        return RegularGridInterpolator(
            (lat, lon),
            values,
            bounds_error=False,
            fill_value=np.nan,
        )

   
    # -------------------------------------------------------------------------
    # Monta tensor de entrada CorrDiff
    # -------------------------------------------------------------------------

    def build_input_tensor(
        self,
        timestamp,
    ):

        timestamp = pd.Timestamp(timestamp)

        data = self.get_timestamp_data(timestamp)

        if data is None:
            return None

        channels = []

        ####################################################################
        # Variáveis de superfície
        ####################################################################

        surface = data["surface"]

        lat = np.asarray(surface["latitude"])
        lon = np.asarray(surface["longitude"])

        for variable in self.surface_variables:

            values = np.asarray(surface[variable])

            grid = self._interpolate_variable(
                lat,
                lon,
                values,
            )

            channels.append(grid)

        ####################################################################
        # Variáveis em níveis de pressão
        ####################################################################

        pressure = data["pressure"]

        for level in self.pressure_levels:

            row = pressure[level]

            if row is None:

                logger.warning(
                    "Missing %dhPa for %s",
                    level,
                    timestamp,
                )

                return None

            lat = np.asarray(row["latitude"])
            lon = np.asarray(row["longitude"])

            for variable in self.pressure_variables:

                values = np.asarray(row[variable])

                grid = self._interpolate_variable(
                    lat,
                    lon,
                    values,
                )

                channels.append(grid)

        ####################################################################
        # Tensor final
        ####################################################################

        tensor = np.stack(
            channels,
            axis=0,
        )

        return tensor.astype(np.float32)

        
# =============================================================================
# BLOCO 3
# RadarDataset
#
# Nesta versão o builder trabalha exclusivamente sobre o radar_cache.
# Os PNGs não são mais utilizados durante a geração do dataset CorrDiff.
# =============================================================================
class RadarDataset:

    """
    Leitor do radar cache.

    Cada arquivo representa um instante de tempo.

        YYYYMMDD_HH_MM.npy

    contendo um grid float32 já interpolado para a grade regular.

    Exemplo

        20240115_13_24.npy

    """

    # -------------------------------------------------------------------------
    # Constructor
    # -------------------------------------------------------------------------

    def __init__(

        self,

        cache_dir,

        lat_range,

        lon_range,

        resolution_km,

        start_date,

        end_date,

        time_resolution_minutes=2,

    ):

        self.cache_dir = Path(cache_dir)

        self.lat_range = lat_range

        self.lon_range = lon_range

        self.resolution_km = resolution_km

        self.start = pd.Timestamp(start_date)

        self.end = pd.Timestamp(end_date)

        self.time_resolution_minutes = time_resolution_minutes

        self.build_grid()

        logger.info("RadarDataset initialized")

        logger.info(
            "Cache directory : %s",
            self.cache_dir,
        )

    # -------------------------------------------------------------------------
    # Regular grid
    # -------------------------------------------------------------------------

    def build_grid(self):

        deg = self.resolution_km / 111.0

        self.lat = np.arange(

            self.lat_range[0],

            self.lat_range[1],

            deg,

        )

        self.lon = np.arange(

            self.lon_range[0],

            self.lon_range[1],

            deg,

        )

        self.Lon, self.Lat = np.meshgrid(

            self.lon,

            self.lat,

        )

        logger.info(

            "Radar grid shape: %s",

            self.Lat.shape,

        )

    # -------------------------------------------------------------------------
    # Timestamp sequence
    # -------------------------------------------------------------------------

    def timestamps(self):

        return pd.date_range(

            start=self.start,

            end=self.end,

            freq=f"{self.time_resolution_minutes}min",

        )

    # -------------------------------------------------------------------------
    # Cache filename
    # -------------------------------------------------------------------------

    def cache_path(

        self,

        timestamp,

    ):

        timestamp = pd.Timestamp(timestamp)

        return (

            self.cache_dir

            / f"{timestamp:%Y%m%d_%H_%M}.npy"

        )

    # -------------------------------------------------------------------------
    # Exists
    # -------------------------------------------------------------------------

    def exists(

        self,

        timestamp,

    ):

        return self.cache_path(

            timestamp

        ).exists()

    # -------------------------------------------------------------------------
    # Available timestamps
    # -------------------------------------------------------------------------

    def available_timestamps(self):

        files = sorted(

            self.cache_dir.glob("*.npy")

        )

        timestamps = []

        for file in files:

            try:

                timestamps.append(

                    pd.to_datetime(

                        file.stem,

                        format="%Y%m%d_%H_%M",

                    )

                )

            except Exception:

                continue

        logger.info(

            "Radar cache timestamps: %,d",

            len(timestamps),

        )

        return timestamps

    # -------------------------------------------------------------------------
    # Read one grid
    # -------------------------------------------------------------------------

    def get_grid(

        self,

        timestamp,

    ):

        file = self.cache_path(timestamp)

        if not file.exists():

            return None

        try:

            grid = np.load(file)

        except Exception as e:

            logger.warning(

                "Could not read %s (%s)",

                file.name,

                e,

            )

            return None

        return grid.astype(np.float32)

    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    @property
    def shape(self):

        return self.Lat.shape

    @property
    def height(self):

        return self.Lat.shape[0]

    @property
    def width(self):

        return self.Lat.shape[1]

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    def summary(self):

        logger.info("================================")

        logger.info("RADAR DATASET")

        logger.info("================================")

        logger.info(

            "Grid shape : %s",

            self.shape,

        )

        logger.info(

            "Latitude range : %.3f %.3f",

            self.lat.min(),

            self.lat.max(),

        )

        logger.info(

            "Longitude range : %.3f %.3f",

            self.lon.min(),

            self.lon.max(),

        )

        logger.info(

            "Resolution : %.1f km",

            self.resolution_km,

        )

        logger.info(

            "Cache files : %,d",

            len(

                list(

                    self.cache_dir.glob("*.npy")

                )

            ),

        )

        logger.info("================================")


# =============================================================================
# CorrDiff Dataset Builder
# =============================================================================
class CorrDiffDatasetBuilder:

    """
    Builder responsável apenas pela geração do dataset CorrDiff.

    Toda a leitura e interpolação do ERA5 fica encapsulada em Era5Dataset.
    """

    def __init__(

        self,

        era5: Era5Dataset,

        radar: RadarDataset,

        output_dir,

        patch_size=32,

        stride=16,

        chunk_size=256,

    ):

        self.logger = setup_logger()

        self.era5 = era5

        self.radar = radar

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(

            parents=True,

            exist_ok=True,

        )

        self.patch_size = patch_size

        self.stride = stride

        self.chunk_size = chunk_size

        self.sample_id = 0

        self.zarr_path = self.output_dir / "train.zarr"

        self.root = None

        self.input_ds = None

        self.target_ds = None

        self.mask_ds = None

        self.timestamps_ds = None

        self.running_sum = None

        self.running_sq_sum = None

        self.running_count = 0

        self.target_sum = 0.0

        self.target_sq_sum = 0.0

        self.target_count = 0

        self.compressor = Blosc(

            cname="zstd",

            clevel=3,

            shuffle=2,

        )

        self.logger.info("CorrDiff Builder initialized")
        
    ###########################################################################
    # Zarr initialization
    ###########################################################################

    def initialize_zarr(self):

        self.logger.info("Creating Zarr dataset...")

        self.root = zarr.open(

            str(self.zarr_path),

            mode="w",

            zarr_version=2,

        )

        channels = self.era5.n_channels

        self.input_ds = self.root.create_dataset(

            "input",

            shape=(1, channels,

                   self.patch_size,

                   self.patch_size),

            chunks=(

                self.chunk_size,

                channels,

                self.patch_size,

                self.patch_size,

            ),

            dtype=np.float32,

            compressor=self.compressor,

        )

        self.target_ds = self.root.create_dataset(

            "target",

            shape=(1,1,

                   self.patch_size,

                   self.patch_size),

            chunks=(

                self.chunk_size,

                1,

                self.patch_size,

                self.patch_size,

            ),

            dtype=np.float32,

            compressor=self.compressor,

        )

        self.mask_ds = self.root.create_dataset(

            "mask",

            shape=(1,1,

                   self.patch_size,

                   self.patch_size),

            chunks=(

                self.chunk_size,

                1,

                self.patch_size,

                self.patch_size,

            ),

            dtype=np.float32,

            compressor=self.compressor,

        )

        self.timestamps_ds = self.root.create_dataset(

            "timestamps",

            shape=(1,),

            chunks=(self.chunk_size,),

            dtype=np.int64,

            compressor=self.compressor,

        )
        
    ###########################################################################
    # Resize datasets
    ###########################################################################

    def _grow(self):

        idx = self.sample_id + 1

        self.input_ds.resize(

            (

                idx,

                self.era5.n_channels,

                self.patch_size,

                self.patch_size,

            )

        )

        self.target_ds.resize(

            (

                idx,

                1,

                self.patch_size,

                self.patch_size,

            )

        )

        self.mask_ds.resize(

            (

                idx,

                1,

                self.patch_size,

                self.patch_size,

            )

        )

        self.timestamps_ds.resize(

            (idx,)

        )
        
        
     ###########################################################################
    # Write sample
    ###########################################################################

    def append_sample(

        self,

        x,

        y,

        mask,

        timestamp,

    ):

        self._grow()

        idx = self.sample_id

        self.input_ds[idx] = x.astype(np.float32)

        self.target_ds[idx] = y.astype(np.float32)

        self.mask_ds[idx] = mask.astype(np.float32)

        self.timestamps_ds[idx] = int(

            pd.Timestamp(timestamp).timestamp()

        )

        if self.running_sum is None:

            self.running_sum = x.sum(axis=(1,2))

            self.running_sq_sum = (

                x**2

            ).sum(axis=(1,2))

        else:

            self.running_sum += x.sum(axis=(1,2))

            self.running_sq_sum += (

                x**2

            ).sum(axis=(1,2))

        self.running_count += (

            x.shape[1] * x.shape[2]

        )

        self.target_sum += y.sum()

        self.target_sq_sum += (

            y**2

        ).sum()

        self.target_count += y.size

        self.sample_id += 1
        
        
    ###########################################################################
    # Patch extraction
    ###########################################################################

    def process_patches(

        self,

        X,

        Y,

        timestamp,

    ):


        H, W = Y.shape


        created = 0


        for i in range(

            0,

            H - self.patch_size + 1,

            self.stride,

        ):


            for j in range(

                0,

                W - self.patch_size + 1,

                self.stride,

            ):


                ################################################################
                # Input patch
                ################################################################

                xp = X[

                    :,

                    i:i+self.patch_size,

                    j:j+self.patch_size,

                ]



                ################################################################
                # Target patch
                ################################################################

                yp = Y[

                    i:i+self.patch_size,

                    j:j+self.patch_size,

                ]



                ################################################################
                # Validity mask
                ################################################################

                mask = ~np.isnan(yp)



                valid_ratio = mask.mean()



                #
                # remove empty patches
                #
                if valid_ratio < 0.05:

                    continue



                ################################################################
                # CorrDiff preprocessing
                ################################################################

                yp = np.nan_to_num(

                    yp,

                    nan=0.0,

                )


                #
                # log transform
                #
                yp = np.log1p(

                    yp

                )


                yp = yp[None,:,:]


                mask = mask[None,:,:]



                ################################################################
                # Write sample
                ################################################################

                self.append_sample(

                    xp,

                    yp,

                    mask,

                    timestamp,

                )


                created += 1



        self.logger.debug(

            "Timestamp %s generated %d patches",

            timestamp,

            created,

        )
    
    ###########################################################################
    # Dataset generation
    ###########################################################################

    def build(self):

        """
        Build complete CorrDiff dataset.

        Pipeline:

        ERA5
          |
          v
        tensor(C,H,W)

        Radar
          |
          v
        precipitation field

        Both are split into patches
        and stored in Zarr.
        """

        self.logger.info(
            "Starting CorrDiff dataset generation"
        )


        #
        # ERA5 timestamps
        #
        timestamps = self.era5.timestamps()


        self.logger.info(

            "Available timestamps: %d",

            len(timestamps)

        )


        #
        # Initialize storage lazily
        #
        initialized = False


        for timestamp in tqdm(

            timestamps,

            desc="Building CorrDiff dataset"

        ):


            ###################################################################
            # ERA5 tensor
            ###################################################################

            X = self.era5.get_tensor(

                timestamp,

                target_lat=self.radar.lat,

                target_lon=self.radar.lon,

            )


            if X is None:

                continue



            ###################################################################
            # Radar target
            ###################################################################

            Y = self.radar.get_grid(

                timestamp

            )


            if Y is None:

                continue



            if np.isnan(Y).all():

                continue



            ###################################################################
            # Initialize Zarr
            ###################################################################

            if not initialized:

                self.initialize_zarr()

                initialized = True



            ###################################################################
            # Extract patches
            ###################################################################

            self.process_patches(

                X,

                Y,

                timestamp,

            )


        #######################################################################
        # Statistics
        #######################################################################

        self.save_stats()


        self.logger.info(

            "Dataset generation completed"

        )


        self.logger.info(

            "Total samples: %d",

            self.sample_id

        )
        
# =============================================================================
# ARGUMENT PARSER
# =============================================================================


def parameter_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-b",
        "--begin",
        required=True,
    )

    parser.add_argument(
        "-e",
        "--end",
        required=True,
    )

    parser.add_argument(
        "--era5_variables",
        type=str,
        default="u,v",
    )

    parser.add_argument(
        "--lat_range",
        nargs=2,
        type=float,
        default=[-23.5, -22.25],
    )

    parser.add_argument(
        "--lon_range",
        nargs=2,
        type=float,
        default=[-44.0, -42.5],
    )

    parser.add_argument(
        "--radar_res",
        type=int,
        default=2,
    )

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================
def main(args):

    era5 = ERA5Dataset(
        path=ERA5_DIR,
        surface_variables=args.era5_surface_variables.split(","),
        pressure_variables=args.era5_pressure_variables.split(","),
        pressure_levels=args.era5_pressure_levels.split(","),
        start_date=args.begin + " 00:00:00",
        end_date=args.end + " 23:00:00",
    )

    era5.load()

    radar = RadarDataset(
        radar_path=RADAR_DIR,
        cache_dir=RADAR_CACHE_DIR,
        resolution_km=args.radar_res,
        lat_range=args.lat_range,
        lon_range=args.lon_range,
        start_date=args.begin + " 00:00:00",
        end_date=args.end + " 23:59:00",
    )

    radar.build_grid()

    radar.precompute_pixel_map()

    builder = CorrDiffDatasetBuilder(
        era5=era5,
        radar=radar,
        output_dir="datasets/corrdiff",
        patch_size=32,
        stride=16,
        chunk_size=256,
    )

    builder.build()


if __name__ == "__main__":

    args = parameter_parser()

    main(args)
    