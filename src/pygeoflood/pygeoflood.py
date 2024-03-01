import numpy as np
import os
import rasterio as rio
import shutil
import time
import toml

from . import tools as t
from pathlib import Path
from typing import Union


def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(
            f"{func.__name__} completed in {end_time - start_time:.4f} seconds"
        )
        return result

    return wrapper


class PyGeoFlood(object):
    def __init__(
        self,
        dem_path,
        project_dir=None,
        config_path=None,
    ):
        """
        Create a new pygeoflood model instance.

        Parameters
        ---------
        dem_path : `str`, `os.pathlike`
            Path to DEM in GeoTIFF format.
        project_dir : `str`, `os.pathlike`, optional
            Path to project directory. Default is current working directory.
        config_path : `str`, `os.pathlike`, optional
            Path to configuration file. If not provided, default configuration
            file will be copied to project directory.
        """

        self._dem_path = Path(dem_path)
        # if no project_dir is provided, use dir containing DEM
        if project_dir is not None:
            self._project_dir = Path(project_dir)
        else:
            self._project_dir = dem_path.parent
        # if no config_path is provided, use default config file
        if config_path is not None:
            self._config_path = Path(config_path)
            with open(self._config_path) as f:
                self._config = toml.load(f)
        else:
            default_config_path = Path(Path(__file__).parent, "config.toml")
            self._config_path = Path(self._project_dir, "config.toml")
            shutil.copy(default_config_path, self._config_path)
            with open(self._config_path) as f:
                self._config = toml.load(f)

    def __repr__(self):
        attrs = "\n    ".join(
            f'{k[1:]}="{v}"' if isinstance(v, (str, Path)) else f"{k[1:]}={v!r}"
            for k, v in self.__dict__.items()
            if v is not None and k != "_config"
        )
        return f"{self.__class__.__name__}(\n    {attrs}\n)"

    @property
    def dem_path(self) -> Union[str, os.PathLike]:
        return self._dem_path

    @dem_path.setter
    def dem_path(self, value: Union[str, os.PathLike]):
        if isinstance(value, (str, os.PathLike)):
            self._dem_path = value
        else:
            raise TypeError("dem_path must be a string or os.PathLike object")

    @property
    def project_dir(self) -> Union[str, os.PathLike]:
        return self._project_dir

    @project_dir.setter
    def project_dir(self, value: Union[str, os.PathLike]):
        if isinstance(value, (str, os.PathLike)):
            self._project_dir = value
        else:
            raise TypeError(
                "project_dir must be a string or os.PathLike object"
            )

    @property
    def filtered_dem_path(self) -> Union[str, os.PathLike]:
        return self._filtered_dem_path

    @filtered_dem_path.setter
    def filtered_dem_path(self, value: Union[str, os.PathLike]):
        if isinstance(value, (str, os.PathLike)):
            self._filtered_dem_path = value
        else:
            raise TypeError(
                "filtered_dem_path must be a string or os.PathLike object"
            )

    @property
    def slope_path(self) -> Union[str, os.PathLike]:
        return self._slope_path

    @slope_path.setter
    def slope_path(self, value: Union[str, os.PathLike]):
        if isinstance(value, (str, os.PathLike)):
            self._slope_path = value
        else:
            raise TypeError("slope_path must be a string or os.PathLike object")

    @property
    def curvature_path(self) -> Union[str, os.PathLike]:
        return self._curvature_path

    @slope_path.setter
    def curvature_path(self, value: Union[str, os.PathLike]):
        if isinstance(value, (str, os.PathLike)):
            self._curvature_path = value
        else:
            raise TypeError(
                "curvature_path must be a string or os.PathLike object"
            )

    @property
    def config_path(self) -> Union[str, os.PathLike]:
        return self._config_path

    @config_path.setter
    def config_path(self, value: Union[str, os.PathLike]):
        if (
            isinstance(value, (str, os.PathLike))
            and Path(value).suffix == ".toml"
        ):
            if Path(value).is_file():
                self._config_path = value
                with open(self._config_path) as f:
                    self._config = toml.load(f)
            else:
                default_config_path = Path(Path(__file__).parent, "config.toml")
                self._config_path = Path(self._project_dir, "config.toml")
                shutil.copy(default_config_path, self._config_path)
                with open(self._config_path) as f:
                    self._config = toml.load(f)
        else:
            raise TypeError(
                "config must be a string or os.PathLike object with .toml extension"
            )

    @time_it
    def nonlinear_filter(
        self,
        filtered_dem_path: Union[str, os.PathLike] = None,
    ):
        """Run nonlinear filter on DEM."""
        # read in DEM
        with rio.open(self._dem_path) as ds:
            dem = ds.read(1)
            dem_profile = ds.profile

        pixel_scale = dem_profile["transform"][0]
        dem = t.set_nan(dem, dem_profile["nodata"])
        edgeThresholdValue = t.lambda_nonlinear_filter(dem, pixel_scale)
        filteredDemArray = t.anisodiff(
            img=dem,
            niter=self._config["filter"]["n_iter"],
            kappa=edgeThresholdValue,
            gamma=self._config["filter"]["time_increment"],
            step=(pixel_scale, pixel_scale),
            option=self._config["filter"]["method"],
        )

        # write filtered DEM with lzw compression
        dem_profile.update(compress="lzw")

        if filtered_dem_path is not None:
            filtered_dem = Path(filtered_dem_path)
        else:
            # append to DEM filename, save in same directory
            filtered_dem = Path(
                self._project_dir,
                f"{self._dem_path.stem}_PM_filtered.tif",
            )
        self._filtered_dem_path = filtered_dem

        with rio.open(self._filtered_dem_path, "w", **dem_profile) as ds:
            ds.write(filteredDemArray, 1)

    @time_it
    def slope_curvature(
        self,
        slope_path: Union[str, os.PathLike] = None,
        curvature_path: Union[str, os.PathLike] = None,
    ):
        """Calculate slope and curvature of DEM."""
        if self._filtered_dem_path is None:
            raise ValueError(
                "Filtered DEM must be created before calculating slope and curvature"
            )
        with rio.open(self._filtered_dem_path) as ds:
            filtered_dem = ds.read(1)
            filtered_dem_profile = ds.profile

        print("computing slope")
        pixel_scale = filtered_dem_profile["transform"][0]
        slope_array = t.compute_dem_slope(filtered_dem, pixel_scale)
        slope_array[np.isnan(filtered_dem)] = np.nan

        # write slope array
        if slope_path is not None:
            slope = Path(slope_path)
        else:
            slope = Path(
                self._project_dir,
                f"{self._dem_path.stem}_slope.tif",
            )
        self._slope_path = slope
        with rio.open(self._slope_path, "w", **filtered_dem_profile) as ds:
            ds.write(slope_array, 1)

        print("computing curvature")
        curvature_array = t.compute_dem_curvature(
            filtered_dem,
            pixel_scale,
            self._config["curvature"]["method"],
        )

        # write curvature array
        if curvature_path is not None:
            curvature = Path(curvature_path)
        else:
            curvature = Path(
                self._project_dir,
                f"{self._dem_path.stem}_curvature.tif",
            )
        self._curvature_path = curvature
        with rio.open(self._curvature_path, "w", **filtered_dem_profile) as ds:
            ds.write(curvature_array, 1)
