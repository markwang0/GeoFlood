import numpy as np
import os

from . import tools as t
from pathlib import Path


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
        dem_path : `str`, `os.PathLike`
            Path to DEM in GeoTIFF format.
        project_dir : `str`, `os.PathLike`, optional
            Path to project directory. Default is current working directory.
        """

        self._dem_path = Path(dem_path)
        # if no project_dir is provided, use dir containing DEM
        if project_dir is not None:
            self._project_dir = Path(project_dir)
        else:
            self._project_dir = dem_path.parent

    # String representation of class
    def __repr__(self):
        attrs = "\n    ".join(
            f'{k[1:]}="{v}"' if isinstance(v, (str, Path)) else f"{k[1:]}={v!r}"
            for k, v in self.__dict__.items()
            if v is not None
        )
        return f"{self.__class__.__name__}(\n    {attrs}\n)"

    # create properties for class with getters and setters
    dem_path = t.create_property("dem_path")
    project_dir = t.create_property("project_dir")
    filtered_dem_path = t.create_property("filtered_dem_path")
    slope_path = t.create_property("slope_path")
    curvature_path = t.create_property("curvature_path")

    @t.time_it
    def nonlinear_filter(
        self,
        filtered_dem_path: str | os.PathLike = None,
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
        dem_path : `str`, `os.PathLike`
            Path to DEM in GeoTIFF format.
        filtered_dem_path : `str`, `os.PathLike`, optional
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

        # read original DEM
        dem, dem_profile, pixel_scale = t.read_raster(self._dem_path)

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

        # write filtered DEM
        self._filtered_dem_path = t.write_raster(
            raster=filteredDemArray,
            profile=dem_profile,
            write_path=filtered_dem_path,
            project_dir=self._project_dir,
            dem_name=self._dem_path.stem,
            suffix="filtered",
        )
        print(f"Filtered DEM written to {self._filtered_dem_path}")

    @t.time_it
    def slope_curvature(
        self,
        slope_path: str | os.PathLike = None,
        curvature_path: str | os.PathLike = None,
        method: str = "geometric",
    ):
        """
        Calculate slope and curvature of DEM.

        Parameters
        ---------
        slope_path : `str`, `os.PathLike`, optional
            Path to save slope raster. If not provided, slope raster will be
            saved in project directory.
        curvature_path : `str`, `os.PathLike`, optional
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

        # read filtered DEM
        filtered_dem, filtered_dem_profile, pixel_scale = t.read_raster(
            self._filtered_dem_path
        )

        print("Computing slope")
        slope_array = t.compute_dem_slope(filtered_dem, pixel_scale)
        slope_array[np.isnan(filtered_dem)] = np.nan

        # write slope array
        self._slope_path = t.write_raster(
            raster=slope_array,
            profile=filtered_dem_profile,
            write_path=slope_path,
            project_dir=self._project_dir,
            dem_name=self._dem_path.stem,
            suffix="slope",
        )
        print(f"Slope raster written to {self._slope_path}")

        print("Computing curvature")
        curvature_array = t.compute_dem_curvature(
            filtered_dem,
            pixel_scale,
            method,
        )
        curvature_array[np.isnan(filtered_dem)] = np.nan

        # write curvature array
        self._curvature_path = t.write_raster(
            raster=curvature_array,
            profile=filtered_dem_profile,
            write_path=curvature_path,
            project_dir=self._project_dir,
            dem_name=self._dem_path.stem,
            suffix="curvature",
        )
        print(f"Curvature raster written to {self._curvature_path}")
