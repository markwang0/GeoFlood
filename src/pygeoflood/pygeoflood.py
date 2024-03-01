import numpy as np
import os
import rasterio as rio
import time
import toml
from typing import Union
from .tools import anisodiff, lambda_nonlinear_filter
from pathlib import Path

# put this into project_dir and keep path
# as an attribute of the class
# with open("config.toml") as f:
#     config = toml.load(f)


class PyGeoFlood(object):
    def __init__(
        self,
        dem_path,
        project_dir=None,
        dem_smoothing_quantile=0.9,
    ):
        """
        Create a new pygeoflood model instance.

        Parameters
        ---------
        dem_path : `str`, `os.pathlike`
            Path to DEM in GeoTIFF format.
        project_dir : `str`, `os.pathlike`, optional
            Path to project directory. Default is current working directory.
        dem_smoothing_quantile : `float`, optional
            Quantile of landscape to smooth. Typically ranges from 0.5-0.9, default is 0.9.
        nan_flag : int
        """

        self._dem_path = Path(dem_path)
        # if no project_dir is provided, use dir containing DEM
        if project_dir:
            self._project_dir = Path(project_dir)
        else:
            self._project_dir = dem_path.parent
        self._dem_smoothing_quantile = dem_smoothing_quantile
        self._filtered_dem_path = None

    def __repr__(self):
        attrs = "\n    ".join(
            f"{k[1:]}='{str(v)}'" if isinstance(v, Path) else f"{k[1:]}={v!r}"
            for k, v in self.__dict__.items()
            if v is not None
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
    def dem_smoothing_quantile(self) -> float:
        return self._dem_smoothing_quantile

    @dem_smoothing_quantile.setter
    def dem_smoothing_quantile(self, value: float):
        if isinstance(value, float) and (0 <= value <= 1):
            self._dem_smoothing_quantile = value
        else:
            raise ValueError(
                "dem_smoothing_quantile must be a float between 0 and 1."
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

    def nonlinear_filter(
        self,
        filtered_dem_path: Union[str, os.PathLike] = None,
    ):
        """Run nonlinear filter on DEM."""
        start_time = time.time()
        # read in DEM
        with rio.open(self._dem_path) as ds:
            dem = ds.read(1)
            dem_profile = ds.profile
        # set NaN values on DEM
        demPixelScale = dem_profile["transform"][0]
        dem[dem < config["general"]["nan_floor"]] = np.nan
        dem[dem == dem_profile["nodata"]] = np.nan
        edgeThresholdValue = lambda_nonlinear_filter(dem, demPixelScale)
        filteredDemArray = anisodiff(
            img=dem,
            niter=config["filter"]["n_iter"],
            kappa=edgeThresholdValue,
            gamma=config["filter"]["time_increment"],
            step=(demPixelScale, demPixelScale),
            option=config["filter"]["method"],
        )
        # write filtered DEM with lzw compression
        dem_profile.update(compress="lzw")
        # append to DEM filename, save in same directory
        filtered_dem = Path(
            self._project_dir,
            f"{self._dem_path.stem}_PM_filtered.tif",
        )
        if filtered_dem_path:
            filtered_dem = Path(filtered_dem_path)
        else:
            self._filtered_dem_path = filtered_dem
        with rio.open(filtered_dem, "w", **dem_profile) as ds:
            ds.write(filteredDemArray, 1)

        run_time = time.time() - start_time  # seconds
        print(
            f"Time taken to complete nonlinear filtering: {round(run_time,0)} seconds"
        )

    def slope_curvature(self):
        """Calculate slope and curvature of DEM."""
        start_time = time.time()
