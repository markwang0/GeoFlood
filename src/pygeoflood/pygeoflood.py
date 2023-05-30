import numpy as np
import rasterio
import time
from . import defaults
from .nonlinear_filter import anisodiff, lambda_nonlinear_filter


class Pygeoflood(object):
    def __init__(
        self,
        dem_path=".",
    ):
        """Create a new pygeoflood model instance.

        Parameters
        ---------
        dem_path : string
            File path to DEM in GeoTIFF format. e.g. "some/path/dem.tif"
        """
        self._dem_path = dem_path

    def __repr__(self):
        return f"Pygeoflood(dem_path='{self._dem_path}')"

    @property
    def dem_path(self):
        """File path to DEM in GeoTIFF format."""
        return self._dem_path

    def run_nonlinear_filter(self):
        """Run nonlinear filter on DEM."""
        start_time = time.time()

        # read in DEM
        with rasterio.open(self._dem_path) as ds:
            nanDemArray = ds.read(1)
            # https://rasterio.readthedocs.io/en/stable/topics/switch.html#geotransforms
            demPixelScale = ds.transform[0]
            dem_profile = ds.profile

        nanDemArray[nanDemArray < defaults.demNanFlag] = np.nan
        edgeThresholdValue = lambda_nonlinear_filter(nanDemArray, demPixelScale)
        if defaults.diffusionMethod == "PeronaMalik2":
            filt_opt = 2
        elif defaults.diffusionMethod == "PeronaMalik1":
            filt_opt = 1
        filteredDemArray = anisodiff(
            nanDemArray,
            defaults.nFilterIterations,
            edgeThresholdValue,
            defaults.diffusionTimeIncrement,
            (demPixelScale, demPixelScale),
            option=filt_opt,
        )

        # write filtered DEM
        # dem_profile.update(compress="lzw")
        filtered_dem_path = (
            self._dem_path[:-4] + "_PM_filtered.tif"
        )  # append to DEM filename
        with rasterio.open(filtered_dem_path, "w", **dem_profile) as ds:
            ds.write(filteredDemArray, 1)

        run_time = time.time() - start_time  # seconds
        print(
            f"time taken to complete nonlinear filtering: {round(run_time,0)} seconds"
        )
