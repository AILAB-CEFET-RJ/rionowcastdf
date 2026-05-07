import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.interpolate import RegularGridInterpolator
import pyarrow.dataset as ds
import pyarrow.compute as pc
from src.config.paths import ERA5_DIR, RADAR_CACHE_DIR, RADAR_DIR
import logging
import sys
from PIL import Image
import argparse

# =========================================
# LOGGER UTILS DOMAIN
# =========================================
def setup_logger():
    
    logger = logging.getLogger("corrdiff")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    )
    handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(handler)

    return logger


# =========================================
# ERA5 DOMAIN
# =========================================
class ERA5Dataset:

    def __init__(self, path: str, variables, start_date, end_date, n_threads=8):
        self.logger = setup_logger()
        self.path = Path(path)
        self.variables = variables
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.n_threads = n_threads

        self.df = None
        self.lat = None
        self.lon = None

    def _load_file(self, f):
        try:
            self.logger.info("Loading ERA5...")
            df = pd.read_parquet(f)
    
            if "time" not in df.columns:
                df = df.reset_index()

            df["time"] = pd.to_datetime(df["valid_time"])
            return df
        except:
            return None

    def load(self):
        self.logger.info(f"Loading ERA5 from {self.path}")
        
        if not self.path.exists():
            raise ValueError(f"Path não existe: {self.path}")

        dataset = ds.dataset(self.path, format="parquet", partitioning="hive")


        # =========================================
        # TEMPORAL FILTER
        # =========================================
        time_col = None
        for col in dataset.schema.names:
            if col in ["time", "valid_time"]:
                time_col = col
                break

        if time_col is None:
            raise ValueError("No time column found in dataset")

        self.logger.info(f"Using time column: {time_col}")

        start = pd.Timestamp(self.start_date)
        end = pd.Timestamp(self.end_date)

        filter_expr = (
            (pc.field(time_col) >= pc.scalar(start)) &
            (pc.field(time_col) <= pc.scalar(end))
        )

        columns = ["latitude", "longitude", time_col] + self.variables

        
        table = dataset.to_table(
            columns=columns,
            filter=filter_expr
        )

        if table.num_rows == 0:
            self.logger.error("No data returned after filtering")
            raise ValueError("No data found for the specified date range")

        self.logger.info(f"Rows loaded: {table.num_rows}")
        df = table.to_pandas()

        if time_col != "time":
            df["time"] = df[time_col]

        df["time"] = pd.to_datetime(df["time"])

        df = df.sort_values(["time", "latitude", "longitude"])

        self.df = df
        self.lat = np.sort(df["latitude"].unique())
        self.lon = np.sort(df["longitude"].unique())

        self.logger.info(f"ERA5 loaded: {df.shape}")

    def interpolate(self, era5_t, target_lat, target_lon):
        grids = []

        for var in self.variables:
            grid = era5_t.pivot(
                index="latitude",
                columns="longitude",
                values=var
            ).values

            interp = RegularGridInterpolator(
                (self.lat, self.lon),
                grid,
                bounds_error=False,
                fill_value=np.nan
            )

            pts = np.stack([target_lat.ravel(), target_lon.ravel()], axis=-1)
            vals = interp(pts).reshape(target_lat.shape)

            grids.append(vals)

        return np.stack(grids, axis=0)


# =========================================
# RADAR DOMAIN
# =========================================
class RadarDataset:

    def __init__(self, radar_path, cache_dir, resolution_km, lat_range, lon_range,
                 start_date, end_date):
        
        self.logger = setup_logger()
        self.radar_path = Path(radar_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.start = pd.to_datetime(start_date)
        self.end = pd.to_datetime(end_date)

        self.res_km = resolution_km
        self.lat_range = lat_range
        self.lon_range = lon_range

        self.pos_sumare = (-22.955139, -43.248278)

        self.legend_values = np.array([50,45,40,35,30,25,20,0])
        self.legend_colors = np.array([
            (197,0,197),
            (227,6,5),
            (255,112,0),
            (195,230,0),
            (4,85,4),
            (19,122,19),
            (0,167,12),
            (0,0,0)
        ])

       
    # =========================
    # GRID
    # =========================
    def _build_grid(self):
        LAT_DEG_TO_KM = 111.0 # latitude degree is approximately 111 km
        deg = self.res_km / LAT_DEG_TO_KM

        self.lat = np.arange(self.lat_range[0], self.lat_range[1], deg)
        self.lon = np.arange(self.lon_range[0], self.lon_range[1], deg)

        self.Lon, self.Lat = np.meshgrid(self.lon, self.lat)

    # =========================
    # TIME
    # =========================
    def _build_time_index(self):
        self.times = pd.date_range(self.start, self.end, freq="1h")
        self.logger.info(f"Radar timesteps: {len(self.times)}")

    # =========================
    # FILE PATH
    # =========================
    def _filepath(self, t):
        return self.radar_path / t.strftime("%Y/%m/%d/%Y_%m_%d_%H_%M.png")

    # =========================
    # PIXEL MAP
    # =========================
    def _precompute_pixel_map(self):
        self.logger.info("Precomputing pixel mapping...")

        lat0, lon0 = self.pos_sumare

        self.px = ((self.Lon - lon0) * -32.5 + lon0)
        self.py = ((self.Lat - lat0) * 19.5 + lat0)

        self.px = self.px.astype(int)
        self.py = self.py.astype(int)

    # =========================
    # RGB → REFLECTIVITY
    # =========================
    def _rgb_to_reflectivity(self, rgb):

        flat = rgb.reshape(-1, 3)

        dists = np.sqrt(((flat[:, None, :] - self.legend_colors[None])**2).sum(axis=2))

        idx = np.argsort(dists, axis=1)[:, :2]

        c1 = self.legend_colors[idx[:,0]]
        c2 = self.legend_colors[idx[:,1]]
        v1 = self.legend_values[idx[:,0]]
        v2 = self.legend_values[idx[:,1]]

        dist = np.linalg.norm(c1 - c2, axis=1)
        t = np.divide(
            np.linalg.norm(flat - c1, axis=1),
            dist,
            out=np.zeros_like(dist),
            where=dist!=0
        )

        val = v1 + t * (v2 - v1)

        return val.reshape(rgb.shape[:2])

    # =========================
    # CACHE
    # =========================
    def _cache_path(self, t):
        return self.cache_dir / f"{t.strftime('%Y%m%d_%H')}.npy"

    # =========================
    # PROCESS BY TIMESTEP
    # =========================
    def _process_time(self, t):

        cache = self._cache_path(t)
        if cache.exists():
            return np.load(cache)

        path = self._filepath(t)

        if not path.exists():
            return None

        try:
            img = np.array(Image.open(path).convert("RGB"))

            reflect = self._rgb_to_reflectivity(img)

            # sample grid
            H, W = reflect.shape

            px = np.clip(self.px, 0, W-1)
            py = np.clip(self.py, 0, H-1)

            grid = reflect[py, px]

            np.save(cache, grid.astype(np.float32))

            return grid

        except Exception as e:
            self.logger.error(f"Radar error at {t}: {e}")
            return None

    # =========================
    # GET GRID
    # =========================
    def get_grid(self, t):
        return self._process_time(pd.to_datetime(t))


# =========================================
# DATASET BUILDER
# =========================================
class CorrDiffDatasetBuilder:

    def __init__(self, era5, radar, output_dir, patch_size=32, stride=16):
        self.logger = setup_logger()
        self.era5 = era5
        self.radar = radar
        self.output_dir = Path(output_dir)

        self.patch_size = patch_size
        self.stride = stride

        (self.output_dir / "train").mkdir(parents=True, exist_ok=True)

    def build(self):
        times = sorted(self.era5.df["time"].unique())
        self.logger.info(f"Total timesteps: {len(times)}")

        sample_id = 0

        for t in times:
            self.logger.info(f"Processing timestep: {t}")

            era5_t = self.era5.df[self.era5.df["time"] == t]

            if len(era5_t) == 0:
                self.logger.warning(f"No ERA5 data for {t}")
                continue

            X = self.era5.interpolate(
                era5_t,
                self.radar.Lat,
                self.radar.Lon
            )

            Y = self.radar.get_grid(t)

            if np.isnan(Y).all():
                self.logger.warning(f"All NaN radar grid at {t}")
                continue

            patches_created = 0

            H, W = Y.shape

            for i in range(0, H - self.patch_size + 1, self.stride):
                for j in range(0, W - self.patch_size + 1, self.stride):

                    xp = X[:, i:i+self.patch_size, j:j+self.patch_size]
                    yp = Y[i:i+self.patch_size, j:j+self.patch_size]

                    valid_ratio = np.mean(~np.isnan(yp))

                    if valid_ratio < 0.05:
                        continue

                    mask = ~np.isnan(yp)
                    yp = np.nan_to_num(yp, nan=0.0)

                    np.savez_compressed(
                        self.output_dir / "train" / f"{sample_id:06d}.npz",
                        input=xp.astype(np.float32),
                        target=yp[None, ...].astype(np.float32),
                        mask=mask[None, ...].astype(np.float32)
                    )

                    sample_id += 1
                    patches_created += 1

            self.logger.info(f"Patches created at {t}: {patches_created}")

        self.logger.info(f"Dataset final size: {sample_id}")

def parameter_parser():
    description = """
    Build dataset to use in corrdiff model, using ERA5 reanalysis and Sumaré Radar data.

    Example:
    
    One-day dataset with u,v variables, 2km reflectivity radar resolution and default lat/lon range:
        python3 -m src.spatiotemporal_builder.CorrdiffDatasetBuilder -b 2023-01-01 -e 2023-01-31 -e5vars u,v --lat_range -23.5 -22.25 --lon_range -44.0 -42.5 -radar_res 2
   """

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawTextHelpFormatter)


    parser.add_argument("-b", "--begin", required=True, help="Begin date (YYYY-MM-DD)")
    parser.add_argument("-e", "--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "-e5vars",
        "--era5_variables",
        type=str,
        default="u,v",
        help='Comma-separated list of variables (e.g. "v,u")',
    )
    
    parser.add_argument(
        "--lat_range",
        nargs=2,
        type=float,
        default=[-23.5, -22.25],
        help="Latitude range (default: -23.5, -22.25)",
    )
    
    parser.add_argument(
        "--lon_range",
        nargs=2,
        type=float,
        default=[-44.0, -42.5],
        help="Longitude range (default: -44.0, -42.5)",
    )
    parser.add_argument(
        "--radar_res",
        type=int,
        default=2,
        help="Radar resolution in km (default: 2)"
    )
    return parser.parse_args()

# =========================================
# BEGIN
# =========================================
def main(args):
    
    era5 = ERA5Dataset(
        path=ERA5_DIR,
        variables=args.era5_variables.split(","),
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
        end_date=args.end + " 23:58:00"
        )
    
    radar._build_grid()
    radar._build_time_index()
    radar._precompute_pixel_map()


    builder = CorrDiffDatasetBuilder(
        era5,
        radar,
        output_dir="datasets/corrdiff"
    )

    builder.build()


if __name__ == "__main__":
    args = parameter_parser()
    main(args)