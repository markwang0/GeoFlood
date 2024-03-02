import numpy as np
import os
import rasterio as rio
import shutil
import time


from . import tools as t
from pathlib import Path
from typing import Union


def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time

        # Check if duration is over 60 minutes (3600 seconds)
        if duration > 3600:
            print(f"{func.__name__} completed in {duration / 3600:.4f} hours")
        # Check if duration is over 60 seconds
        elif duration > 60:
            print(f"{func.__name__} completed in {duration / 60:.4f} minutes")
        else:
            print(f"{func.__name__} completed in {duration:.4f} seconds")
        return result

    return wrapper


class PyGeoFlood(object):
    def __init__(
        self,
        dem_path,
        project_dir=None,
    ):
        """
        Create a new pygeoflood model instance.

        Parameters
        ---------
        dem_path : `str`, `os.pathlike`
            Path to DEM in GeoTIFF format.
        project_dir : `str`, `os.pathlike`, optional
            Path to project directory. Default is current working directory.
        """

        self._dem_path = Path(dem_path)
        # if no project_dir is provided, use dir containing DEM
        if project_dir is not None:
            self._project_dir = Path(project_dir)
        else:
            self._project_dir = dem_path.parent

    def __repr__(self):
        attrs = "\n    ".join(
            f'{k[1:]}="{v}"' if isinstance(v, (str, Path)) else f"{k[1:]}={v!r}"
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

    @curvature_path.setter
    def curvature_path(self, value: Union[str, os.PathLike]):
        if isinstance(value, (str, os.PathLike)):
            self._curvature_path = value
        else:
            raise TypeError(
                "curvature_path must be a string or os.PathLike object"
            )

    @time_it
    def nonlinear_filter(
        self,
        filtered_dem_path: Union[str, os.PathLike] = None,
        method: str = "PeronaMalik2",
        smoothing_quantile: float = 0.9,
        time_increment: float = 0.1,
        n_iter: int = 50,
        sigma_squared: float = 0.05,
    ):
        """
        Run nonlinear filter on DEM.

        Parameters
        ---------
        dem_path : `str`, `os.pathlike`
            Path to DEM in GeoTIFF format.
        filtered_dem_path : `str`, `os.pathlike`, optional
            Path to save filtered DEM. If not provided, filtered DEM will be
            saved in project directory.
        method : `str`, optional
            Filter method to apply to DEM. Options include:
            - "PeronaMalik1": TODO: detailed description
            - "PeronaMalik2": TODO: detailed description
            - "Gaussian": Smoothes DEM with a Gaussian filter.
            Default is "PeronaMalik2".
        smoothing_quantile : `float`, optional
            Quantile for calculating Perona-Malik nonlinear filter
            edge threshold value (kappa). Default is 0.9.
        time_increment : `float`, optional
            Time increment for Perona-Malik nonlinear filter. Default is 0.1.
            AKA gamma, a higher makes diffusion process faster but can lead to
            instability.
        n_iter : `int`, optional
            Number of iterations for Perona-Malik nonlinear filter. Default is 50.
        sigma_squared : `float`, optional
            Variance of Gaussian filter. Default is 0.05.
        """
        # read in DEM
        with rio.open(self._dem_path) as ds:
            dem = ds.read(1)
            dem_profile = ds.profile

        pixel_scale = dem_profile["transform"][0]
        dem[dem == dem_profile["nodata"]] = np.nan
        edgeThresholdValue = t.lambda_nonlinear_filter(
            dem, pixel_scale, smoothing_quantile
        )
        filteredDemArray = t.anisodiff(
            img=dem,
            niter=n_iter,
            kappa=edgeThresholdValue,
            gamma=time_increment,
            step=(pixel_scale, pixel_scale),
            option=method,
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
        method: str = "geometric",
    ):
        """
        Calculate slope and curvature of DEM.

        Parameters
        ---------
        slope_path : `str`, `os.pathlike`, optional
            Path to save slope raster. If not provided, slope raster will be
            saved in project directory.
        curvature_path : `str`, `os.pathlike`, optional
            Path to save curvature raster. If not provided, curvature raster
            will be saved in project directory.
        method : `str`, optional
            Method for calculating curvature. Options include:
            - "geometric": TODO: detailed description
            - "laplacian": TODO: detailed description
            Default is "geometric".
        """
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
            method,
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
