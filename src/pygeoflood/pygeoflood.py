import geopandas as gpd
import fiona
import numpy as np
import pandas as pd
import rasterio as rio
import sys

from . import tools as t
from os import PathLike
from pathlib import Path
from rasterio.warp import transform_bounds


class PyGeoFlood(object):

    # make these attributes properties with getters and setters
    # t.path_property() ensures attribute is a pathlib.Path object
    # dem_path and project_dir are not set to None initially
    dem_path = t.path_property("dem_path")
    project_dir = t.path_property("project_dir")
    # below are set to None initially
    filtered_dem_path = t.path_property("filtered_dem_path")
    slope_path = t.path_property("slope_path")
    curvature_path = t.path_property("curvature_path")
    filled_path = t.path_property("filled_path")
    mfd_fac_path = t.path_property("mfd_fac_path")
    d8_fdr_path = t.path_property("d8_fdr_path")
    basins_path = t.path_property("basins_path")
    outlets_path = t.path_property("outlets_path")
    flow_skeleton_path = t.path_property("flow_skeleton_path")
    curvature_skeleton_path = t.path_property("curvature_skeleton_path")
    combined_skeleton_path = t.path_property("combined_skeleton_path")
    cost_function_geodesic_path = t.path_property("cost_function_geodesic_path")
    geodesic_distance_path = t.path_property("geodesic_distance_path")
    channel_heads_path = t.path_property("channel_heads_path")
    flowline_path = t.path_property("flowline_path")
    endpoints_path = t.path_property("endpoints_path")
    binary_hand_path = t.path_property("binary_hand_path")
    custom_flowline_path = t.path_property("custom_flowline_path")
    custom_flowline_raster_path = t.path_property("custom_flowline_raster_path")
    channel_network_path = t.path_property("channel_network_path")
    cost_function_channel_path = t.path_property("cost_function_channel_path")
    streamcell_path = t.path_property("streamcell_path")

    catchment_path = t.path_property("catchment_path")

    def __init__(
        self,
        dem_path,
        project_dir=None,
        filtered_dem_path=None,
        slope_path=None,
        curvature_path=None,
        filled_path=None,
        mfd_fac_path=None,
        d8_fdr_path=None,
        basins_path=None,
        outlets_path=None,
        flow_skeleton_path=None,
        curvature_skeleton_path=None,
        combined_skeleton_path=None,
        cost_function_geodesic_path=None,
        geodesic_distance_path=None,
        channel_heads_path=None,
        flowline_path=None,
        endpoints_path=None,
        binary_hand_path=None,
        custom_flowline_path=None,
        custom_flowline_raster_path=None,
        channel_network_path=None,
        cost_function_channel_path=None,
        streamcell_path=None,
        catchment_path=None,
    ):
        """
        Create a new pygeoflood model instance.

        Parameters
        ---------
        dem_path : `str`, `os.PathLike`
            Path to DEM in GeoTIFF format.
        project_dir : `str`, `os.PathLike`, optional
            Path to project directory. Default is the directory containing the
            DEM. All outputs will be saved to this directory.
        **kwargs : `dict`, optional
        """

        # automatically becomes a pathlib.Path object if not aleady
        self.dem_path = dem_path
        # if no project_dir is provided, use dir containing DEM
        if project_dir is not None:
            self.project_dir = project_dir
        else:
            self.project_dir = dem_path.parent
        self.filtered_dem_path = filtered_dem_path
        self.slope_path = slope_path
        self.curvature_path = curvature_path
        self.filled_path = filled_path
        self.mfd_fac_path = mfd_fac_path
        self.d8_fdr_path = d8_fdr_path
        self.basins_path = basins_path
        self.outlets_path = outlets_path
        self.flow_skeleton_path = flow_skeleton_path
        self.curvature_skeleton_path = curvature_skeleton_path
        self.combined_skeleton_path = combined_skeleton_path
        self.cost_function_geodesic_path = cost_function_geodesic_path
        self.geodesic_distance_path = geodesic_distance_path
        self.channel_heads_path = channel_heads_path
        self.flowline_path = flowline_path
        self.endpoints_path = endpoints_path
        self.binary_hand_path = binary_hand_path
        self.custom_flowline_path = custom_flowline_path
        self.custom_flowline_raster_path = custom_flowline_raster_path
        self.channel_network_path = channel_network_path
        self.cost_function_channel_path = cost_function_channel_path
        self.streamcell_path = streamcell_path

        self.catchment_path = catchment_path

    # string representation of class
    # output can be used to recreate instance
    def __repr__(self):
        attrs = "\n    ".join(
            (
                f'{k[1:]}="{v}",'
                if isinstance(v, (str, Path))
                else f"{k[1:]}={v!r},"
            )
            for k, v in self.__dict__.items()
            if v is not None
        )
        return f"{self.__class__.__name__}(\n    {attrs}\n)"

    @t.time_it
    def nonlinear_filter(
        self,
        custom_path: str | PathLike = None,
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
        dem_path : `str`, `os.PathLike`, optional
            Path to DEM in GeoTIFF format. If not provided, DEM path provided
            when creating the PyGeoFlood instance will be used.
        custom_path : `str`, `os.PathLike`, optional
            Custom path to save filtered DEM. If not provided, filtered DEM
            will be saved in project directory.
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
        dem, dem_profile, pixel_scale = t.read_raster(self.dem_path)

        edgeThresholdValue = t.lambda_nonlinear_filter(
            dem, pixel_scale, smoothing_quantile
        )

        filtered_dem = t.anisodiff(
            img=dem,
            niter=n_iter,
            kappa=edgeThresholdValue,
            gamma=time_increment,
            step=(pixel_scale, pixel_scale),
            option=method,
        )

        # get file path for filtered DEM
        self.filtered_dem_path = t.get_file_path(
            custom_path=custom_path,
            project_dir=self.project_dir,
            dem_name=self.dem_path.stem,
            suffix="filtered",
        )

        # write filtered DEM
        t.write_raster(
            raster=filtered_dem,
            profile=dem_profile,
            file_path=self.filtered_dem_path,
        )
        print(f"Filtered DEM written to {str(self.filtered_dem_path)}")

    @t.time_it
    def slope(
        self,
        custom_path: str | PathLike = None,
    ):
        """
        Calculate slope of DEM.

        Parameters
        ---------
        custom_path : `str`, `os.PathLike`, optional
            Custom path to save slope raster. If not provided, slope raster
            will be saved in project directory.
        """

        t.check_attributes([("Filtered DEM", self.filtered_dem_path)], "slope")

        # read filtered DEM
        filtered_dem, filtered_dem_profile, pixel_scale = t.read_raster(
            self.filtered_dem_path
        )

        slope_array = t.compute_dem_slope(filtered_dem, pixel_scale)

        # get file path for slope array
        self.slope_path = t.get_file_path(
            custom_path=custom_path,
            project_dir=self.project_dir,
            dem_name=self.dem_path.stem,
            suffix="slope",
        )

        # write slope array
        t.write_raster(
            raster=slope_array,
            profile=filtered_dem_profile,
            file_path=self.slope_path,
        )
        print(f"Slope raster written to {str(self.slope_path)}")

    @t.time_it
    def curvature(
        self,
        custom_path: str | PathLike = None,
        method: str = "geometric",
    ):
        """
        Calculate curvature of DEM.

        Parameters
        ---------
        custom_path : `str`, `os.PathLike`, optional
            Custom path to save curvature raster. If not provided, curvature
            raster will be saved in project directory.
        method : `str`, optional
            Method for calculating curvature. Options include:
            - "geometric": TODO: detailed description
            - "laplacian": TODO: detailed description
            Default is "geometric".
        """

        t.check_attributes(
            [("Filtered DEM", self.filtered_dem_path)], "curvature"
        )

        # read filtered DEM
        filtered_dem, filtered_dem_profile, pixel_scale = t.read_raster(
            self.filtered_dem_path
        )

        curvature_array = t.compute_dem_curvature(
            filtered_dem,
            pixel_scale,
            method,
        )

        # get file path for curvature array
        self.curvature_path = t.get_file_path(
            custom_path=custom_path,
            project_dir=self.project_dir,
            dem_name=self.dem_path.stem,
            suffix="curvature",
        )

        # write curvature array
        t.write_raster(
            raster=curvature_array,
            profile=filtered_dem_profile,
            file_path=self.curvature_path,
        )
        print(f"Curvature raster written to {str(self.curvature_path)}")

    @t.time_it
    def fill_depressions(
        self,
        custom_path: str | PathLike = None,
        **wbt_args,
    ):
        """
        Fill filtered DEM depressions. This is a wrapper for the WhiteboxTools
        `fill_depressions` function.

        Parameters
        ---------
        custom_path : `str`, `os.PathLike`, optional
            Path to save filled DEM. If not provided, filled DEM will be saved
            in project directory.
        wbt_args : `dict`, optional
            Additional arguments to pass to the WhiteboxTools `fill_depressions`
            function. See WhiteboxTools documentation for details.
        """

        t.check_attributes(
            [("Filtered DEM", self.filtered_dem_path)], "fill_depressions"
        )

        # get file path for filled DEM
        self.filled_path = t.get_file_path(
            custom_path=custom_path,
            project_dir=self.project_dir,
            dem_name=self.dem_path.stem,
            suffix="filled",
        )

        # get instance of WhiteboxTools
        wbt = t.get_WhiteboxTools()

        # fill DEM depressions
        # use absolute paths to avoid errors
        wbt.fill_depressions(
            dem=self.filtered_dem_path.resolve(),
            output=self.filled_path.resolve(),
            fix_flats=True,
            **wbt_args,
        )

        print(f"Filled DEM written to {str(self.filled_path)}")

    @t.time_it
    def mfd_flow_accumulation(
        self,
        custom_path: str | PathLike = None,
        **wbt_args,
    ):
        """
        Calculate MFD flow accumulation. This is a wrapper for the WhiteboxTools
        `quinn_flow_accumulation` function.

        Parameters
        ---------
        mfd_fac_path : `str`, `os.PathLike`, optional
            Path to save MFD flow accumulation raster. If not provided, MFD flow
            accumulation raster will be saved in project directory.
        wbt_args : `dict`, optional
            Additional arguments to pass to the WhiteboxTools `quinn_flow_accumulation`
            function. See WhiteboxTools documentation for details.
        """

        t.check_attributes(
            [("Filled DEM", self.filled_path)], "MFD flow accumulation"
        )

        # get file path for MFD flow accumulation
        self.mfd_fac_path = t.get_file_path(
            custom_path=custom_path,
            project_dir=self.project_dir,
            dem_name=self.dem_path.stem,
            suffix="mfd_fac",
        )

        # get instance of WhiteboxTools
        wbt = t.get_WhiteboxTools()

        # calculate MFD flow accumulation
        # use absolute paths to avoid errors
        wbt.quinn_flow_accumulation(
            dem=self.filled_path.resolve(),
            output=self.mfd_fac_path.resolve(),
            out_type="cells",
            **wbt_args,
        )

        print(
            f"MFD flow accumulation raster written to {str(self.mfd_fac_path)}"
        )

    @t.time_it
    def d8_flow_direction(
        self,
        custom_path: str | PathLike = None,
        **wbt_args,
    ):
        """
        Calculate D8 flow direction. This is a wrapper for the WhiteboxTools
        `d8_pointer` function.

        Parameters
        ---------
        custom_path : `str`, `os.PathLike`, optional
            Path to save D8 flow direction raster. If not provided, D8 flow
            direction raster will be saved in project directory.
        wbt_args : `dict`, optional
            Additional arguments to pass to the WhiteboxTools `d8_pointer`
            function. See WhiteboxTools documentation for details.
        """

        t.check_attributes(
            [("Filled DEM", self.filled_path)], "D8 flow direction"
        )

        # get file path for D8 flow direction
        self.d8_fdr_path = t.get_file_path(
            custom_path=custom_path,
            project_dir=self.project_dir,
            dem_name=self.dem_path.stem,
            suffix="d8_fdr",
        )

        # get instance of WhiteboxTools
        wbt = t.get_WhiteboxTools()

        # calculate D8 flow direction
        # use absolute paths to avoid errors
        wbt.d8_pointer(
            dem=self.filled_path.resolve(),
            output=self.d8_fdr_path.resolve(),
            **wbt_args,
        )

        # for some reason WBT assigns D8 values to nodata cells
        # add back nodata cells from filtered DEM
        filtered_dem, filtered_profile, _ = t.read_raster(
            self.filtered_dem_path
        )
        filtered_dem[filtered_dem == filtered_profile["nodata"]] = np.nan
        # read D8 flow direction raster
        d8_fdr, d8_profile, _ = t.read_raster(self.d8_fdr_path)
        d8_fdr[np.isnan(filtered_dem)] = d8_profile["nodata"]
        # write D8 flow direction raster
        t.write_raster(
            raster=d8_fdr,
            profile=d8_profile,
            file_path=self.d8_fdr_path,
        )

        print(f"D8 flow direction raster written to {str(self.d8_fdr_path)}")

    @t.time_it
    def outlets(
        self,
        custom_path: str | PathLike = None,
    ):
        """
        Create outlets raster. Outlets are cells which have no downslope neighbors
        according to the D8 flow direction. Outlets are designated by 1, all other
        cells are designated by 0.

        Parameters
        ---------
        custom_path : `str`, `os.PathLike`, optional
            Path to save outlets raster. If not provided, outlets raster will be
            saved in project directory.
        """

        t.check_attributes(
            [("D8 flow direction raster", self.d8_fdr_path)], "outlets"
        )

        # read D8 flow direction raster, outlets designated by WBT as 0
        outlets, profile, _ = t.read_raster(self.d8_fdr_path)
        nan_mask = outlets == profile["nodata"]
        # get outlets as 1, all else as 0
        # make all cells 1 that are not outlets
        outlets[outlets != 0] = 1
        # flip to get outlets as 1, all else as 0
        outlets = 1 - outlets
        # reset nodata cells, which were set to 0 above
        outlets[nan_mask] = profile["nodata"]

        # get file path for outlets raster
        self.outlets_path = t.get_file_path(
            custom_path=custom_path,
            project_dir=self.project_dir,
            dem_name=self.dem_path.stem,
            suffix="outlets",
        )

        # write outlets raster
        t.write_raster(
            raster=outlets,
            profile=profile,
            file_path=self.outlets_path,
        )

        print(f"Outlets raster written to {str(self.outlets_path)}")

    @t.time_it
    def basins(
        self,
        custom_path: str | PathLike = None,
        **wbt_args,
    ):
        """
        Delineate basins. This is a wrapper for the WhiteboxTools `basins` function.

        Parameters
        ---------
        custom_path : `str`, `os.PathLike`, optional
            Path to save basins raster. If not provided, basins raster will be
            saved in project directory.
        wbt_args : `dict`, optional
            Additional arguments to pass to the WhiteboxTools `basins` function.
            See WhiteboxTools documentation for details.
        """

        t.check_attributes(
            [("D8 flow direction raster", self.d8_fdr_path)], "basins"
        )

        # get file path for basins
        self.basins_path = t.get_file_path(
            custom_path=custom_path,
            project_dir=self.project_dir,
            dem_name=self.dem_path.stem,
            suffix="basins",
        )

        # get instance of WhiteboxTools
        wbt = t.get_WhiteboxTools()

        # delineate basins
        # use absolute paths to avoid errors
        wbt.basins(
            d8_pntr=self.d8_fdr_path.resolve(),
            output=self.basins_path.resolve(),
            **wbt_args,
        )

        print(f"Basins raster written to {str(self.basins_path)}")

    @t.time_it
    def skeleton_definition(
        self,
        custom_path: str | PathLike = None,
        fac_threshold: float = 3000,
        write_flow_skeleton: bool = False,
        write_curvature_skeleton: bool = False,
    ):
        """
        Define skeleton from flow and curvature.

        Parameters
        ---------
        custom_path : `str`, `os.PathLike`, optional
            Custom path to save combined skeleton. If not provided, combined
            skeleton will be saved in project directory.
        fac_threshold : `float`, optional
            Flow accumlulation threshold for defining flow skeleton. Default is 3000.
        write_flow_skeleton : `bool`, optional
            Whether to write flow skeleton to file. Default is False.
        write_curvature_skeleton : `bool`, optional
            Whether to write curvature skeleton to file. Default is False.
        """

        check_rasters = [
            ("Curvature raster", self.curvature_path),
            ("Flow accumulation raster", self.mfd_fac_path),
        ]

        t.check_attributes(check_rasters, "skeleton_definition")

        # get skeleton from curvature only
        curvature, curvature_profile, _ = t.read_raster(self.curvature_path)
        finite_curvature = curvature[np.isfinite(curvature)]
        curvature_mean = np.nanmean(finite_curvature)
        curvature_std = np.nanstd(finite_curvature)
        print("Curvature mean: ", curvature_mean)
        print("Curvature standard deviation: ", curvature_std)
        print(f"Curvature Projection: {str(curvature_profile['crs'])}")
        thresholdCurvatureQQxx = 1.5
        curvature_threshold = (
            curvature_mean + thresholdCurvatureQQxx * curvature_std
        )
        curvature_skeleton = t.get_skeleton(curvature, curvature_threshold)

        # get skeleton from flow only
        mfd_fac, _, _ = t.read_raster(self.mfd_fac_path)
        mfd_fac[np.isnan(curvature)] = np.nan
        mfd_fac_mean = np.nanmean(mfd_fac)
        print("Mean upstream flow: ", mfd_fac_mean)
        fac_skeleton = t.get_skeleton(mfd_fac, fac_threshold)

        # get skeleton from flow and curvature
        combined_skeleton = t.get_skeleton(
            curvature, curvature_threshold, mfd_fac, fac_threshold
        )

        skeleton_profile = curvature_profile.copy()
        skeleton_profile.update(dtype="int16", nodata=-32768)

        if write_flow_skeleton:
            # get file path for flow skeleton
            self.flow_skeleton_path = t.get_file_path(
                custom_path=None,
                project_dir=self.project_dir,
                dem_name=self.dem_path.stem,
                suffix="flow_skeleton",
            )
            t.write_raster(
                raster=fac_skeleton,
                profile=skeleton_profile,
                file_path=self.flow_skeleton_path,
            )
            print(f"Flow skeleton written to {str(self.flow_skeleton_path)}")

        if write_curvature_skeleton:
            # get file path for curvature skeleton
            self.curvature_skeleton_path = t.get_file_path(
                custom_path=None,
                project_dir=self.project_dir,
                dem_name=self.dem_path.stem,
                suffix="curvature_skeleton",
            )
            t.write_raster(
                raster=curvature_skeleton,
                profile=skeleton_profile,
                file_path=self.curvature_skeleton_path,
            )
            print(
                f"Curvature skeleton written to {str(self.curvature_skeleton_path)}"
            )

        # write combined skeleton
        self.combined_skeleton_path = t.get_file_path(
            custom_path=custom_path,
            project_dir=self.project_dir,
            dem_name=self.dem_path.stem,
            suffix="combined_skeleton",
        )
        t.write_raster(
            raster=combined_skeleton,
            profile=skeleton_profile,
            file_path=self.combined_skeleton_path,
        )
        print(
            f"Combined skeleton written to {str(self.combined_skeleton_path)}"
        )

    @t.time_it
    def geodesic_distance(
        self,
        custom_path: str | PathLike = None,
        write_cost_function: bool = False,
        basin_elements: int = 2,
        area_threshold: float = 0.1,
        normalize_curvature: bool = True,
        local_cost_min: float | None = None,
    ):
        """
        Calculate geodesic distance. This is a wrapper for the WhiteboxTools
        `geodesic_distance` function.

        Parameters
        ---------
        custom_path : `str`, `os.PathLike`, optional
            Path to save geodesic distance raster. If not provided, geodesic
            distance raster will be saved in project directory.
        write_cost_function : `bool`, optional
            Whether to write cost function raster to file. Default is False.
        basin_elements : `int`, optional
            Number of basin elements. Default is 2.
        area_threshold : `float`, optional
            Area threshold for fast marching method. Default is 0.1.
        normalize_curvature : `bool`, optional
            Whether to normalize curvature. Default is True.
        local_cost_min : `float`, optional
            Minimum local cost. Default is None.
        """

        check_rasters = [
            ("Curvature raster", self.curvature_path),
            ("Flow accumulation raster", self.mfd_fac_path),
            ("Outlets raster", self.outlets_path),
            ("Basins raster", self.basins_path),
            ("Combined skeleton raster", self.combined_skeleton_path),
        ]

        t.check_attributes(check_rasters, "geodesic_distance")

        outlets, o_profile, _ = t.read_raster(self.outlets_path)
        outlets = outlets.astype(np.float32)
        outlets[(outlets == 0) | (outlets == o_profile["nodata"])] = np.nan
        outlets = np.transpose(np.argwhere(~np.isnan(outlets)))
        basins, _, _ = t.read_raster(self.basins_path)
        curvature, _, _ = t.read_raster(self.curvature_path)
        mfd_fac, _, _ = t.read_raster(self.mfd_fac_path)
        filtered_dem, filt_profile, _ = t.read_raster(self.filtered_dem_path)
        mfd_fac[np.isnan(filtered_dem)] = np.nan
        del filtered_dem
        combined_skeleton, _, _ = t.read_raster(self.combined_skeleton_path)

        # get start points for Fast Marching Method
        fmm_start_points = t.get_fmm_points(
            basins, outlets, basin_elements, area_threshold
        )

        # Computing the local cost function
        # min-max normalization of curvature (0 to 1)
        if normalize_curvature:
            curvature = t.minmax_scale(curvature)
        curvature[np.isnan(curvature)] = 0

        # calculate cost function
        # Calculate the local reciprocal cost (weight, or propagation speed
        # in the eikonal equation sense)
        # (mfd_fac + flowMean * combined_skeleton + flowMean * curvature)
        flowMean = np.nanmean(mfd_fac)
        weights_arrays = [
            (1, mfd_fac),
            (flowMean, combined_skeleton),
            (flowMean, curvature),
        ]
        cost_function_geodesic = t.get_combined_cost(
            weights_arrays, return_reciprocal=True
        )
        if local_cost_min is not None:
            cost_function_geodesic[cost_function_geodesic < local_cost_min] = (
                1.0
            )
        # print("1/cost min: ", np.nanmin(cost_function))
        # print("1/cost max: ", np.nanmax(cost_function))
        del curvature, combined_skeleton

        # Compute the geodesic distance using Fast Marching Method
        geodesic_distance = t.fast_marching(
            fmm_start_points, basins, mfd_fac, cost_function_geodesic
        )

        if write_cost_function:
            self.cost_function_geodesic_path = t.get_file_path(
                custom_path=None,
                project_dir=self.project_dir,
                dem_name=self.dem_path.stem,
                suffix="cost_function_geodesic",
            )
            t.write_raster(
                raster=cost_function_geodesic,
                profile=filt_profile,
                file_path=self.cost_function_geodesic_path,
            )
            print(
                f"Cost function written to {str(self.cost_function_geodesic_path)}"
            )

        # get file path for geodesic distance
        self.geodesic_distance_path = t.get_file_path(
            custom_path=custom_path,
            project_dir=self.project_dir,
            dem_name=self.dem_path.stem,
            suffix="geodesic_distance",
        )

        # write geodesic distance
        t.write_raster(
            raster=geodesic_distance,
            profile=filt_profile,
            file_path=self.geodesic_distance_path,
        )

        print(
            f"Geodesic distance raster written to {str(self.geodesic_distance_path)}"
        )

    @t.time_it
    def channel_heads(
        self,
        custom_path: str | PathLike = None,
        channel_head_median_dist: int = 30,
        vector_extension: str = "shp",
        max_channel_heads: int = 10000,
    ):
        """
        Define channel heads.

        Parameters
        ---------
        custom_path : `str`, `os.PathLike`, optional
            Custom path to save channel heads shapefile. If not provided,
            channel heads shapefile will be saved in project directory.
        channel_head_median_dist : `int`, optional
            Median hillslope of the input DEM, i.e. the distance between
            each pixel and the first channelized downslope pixel. Default is 30.
        vector_extension : `str`, optional
            Extension for vector file. Default is "shp".
        max_channel_heads : `int`, optional
            Maximum number of channel heads to extract. Default is 10000.
            (useful for pre-allocation of memory for large rasters)
        """

        check_rasters = [
            ("Combined skeleton raster", self.combined_skeleton_path),
            ("Geodesic distance raster", self.geodesic_distance_path),
        ]

        t.check_attributes(check_rasters, "channel_head_definition")

        # read combined skeleton and geodesic distance rasters
        combined_skeleton, _, _ = t.read_raster(self.combined_skeleton_path)

        geodesic_distance, geo_profile, _ = t.read_raster(
            self.geodesic_distance_path
        )

        # get channel heads
        ch_rows, ch_cols = t.get_channel_heads(
            combined_skeleton,
            geodesic_distance,
            channel_head_median_dist,
            max_channel_heads,
        )

        # get file path for channel heads
        self.channel_heads_path = t.get_file_path(
            custom_path=custom_path,
            project_dir=self.project_dir,
            dem_name=self.dem_path.stem,
            suffix="channel_heads",
            extension=vector_extension,
        )

        # write channel heads points shapefile
        t.write_vector_points(
            rows=ch_rows,
            cols=ch_cols,
            profile=geo_profile,
            dataset_name="channel_heads",
            file_path=self.channel_heads_path,
        )

        print(
            f"Channel heads shapefile written to {str(self.channel_heads_path)}"
        )

    @t.time_it
    def endpoints(
        self,
        custom_path: str | PathLike = None,
    ):
        """
        Save flowline endpoints in a csv file.

        Parameters
        ---------
        custom_path : `str`, `os.PathLike`, optional
            Custom path to save endpoints csv. If not provided, endpoints
            csv will be saved in project directory.
        """

        t.check_attributes(
            [("Flowline vector file", self.flowline_path)],
            "endpoints",
        )

        flowline = gpd.read_file(self.flowline_path)
        endpoints = t.get_endpoints(flowline)

        # get file path for endpoints
        self.endpoints_path = t.get_file_path(
            custom_path=custom_path,
            project_dir=self.project_dir,
            dem_name=self.dem_path.stem,
            suffix="endpoints",
            extension="csv",
        )

        # write endpoints csv
        endpoints.to_csv(self.endpoints_path, index=False)

        print(f"Endpoints csv written to {str(self.endpoints_path)}")

    @t.time_it
    def binary_hand(
        self,
        custom_path: str | PathLike = None,
    ):
        """
        Creates binary HAND raster with values of 1 given to pixels at a lower
        elevation than the elevation associated with NHD MR Flowline pixels.
        A value of zero is given to all other pixels in the image, i.e. pixels
        at a higher elevation than the NHD MR Flowlines.

        Parameters
        ---------
        custom_path : `str`, `os.PathLike`, optional
            Custom path to save binary HAND raster. If not provided, binary HAND
            raster will be saved in project directory.
        """

        required_files = [
            ("DEM", self.dem_path),
            ("Flowline vector file", self.flowline_path),
        ]

        t.check_attributes(required_files, "endpoints")

        flowline = gpd.read_file(self.flowline_path)
        dem, dem_profile, _ = t.read_raster(self.dem_path)
        binary_hand = t.get_binary_hand(flowline, dem, dem_profile)
        out_profile = dem_profile.copy()
        out_profile.update(dtype="int16", nodata=-32768)
        binary_hand[dem == dem_profile["nodata"]] = out_profile["nodata"]
        binary_hand[np.isnan(dem)] = out_profile["nodata"]

        # get file path for binary hand
        self.binary_hand_path = t.get_file_path(
            custom_path=custom_path,
            project_dir=self.project_dir,
            dem_name=self.dem_path.stem,
            suffix="binary_hand",
        )

        # write binary hand
        t.write_raster(
            raster=binary_hand,
            profile=out_profile,
            file_path=self.binary_hand_path,
        )

        print(f"Binary HAND raster written to {str(self.binary_hand_path)}")

    @t.time_it
    def rasterize_custom_flowline(
        self,
        custom_path: str | PathLike = None,
        layer: str | int = 0,
    ):
        """
        Create custom flowline raster from user-provided flowline vector file.
        This flowline could be obtained from the NHD HR dataset, for example.
        The attribute `custom_flowline_path` must be set before running this method.
        The flowline is buffered by 5 units and cropped to the extent of the DEM.
        The crs of the flowline will be reprojected to the crs of the DEM.
        Note: if you already have a custom flowline raster, you can skip this step
        and set the `custom_flowline_raster_path` attribute directly.

        Parameters
        ---------
        custom_path : `str`, `os.PathLike`, optional
            Custom path to save custom flowline raster. If not provided, custom
            flowline raster will be saved in project directory. The flowline
            raster has data type int16 with 1=channel and 0=non-channel.
        layer : `str` or `int`, optional
            Layer name or number in flowline vector file. Default is 0.
        """

        t.check_attributes(
            [("Custom flowline vector file", self.custom_flowline_path)],
            "custom_flowline",
        )

        # get bounding box and crs from DEM to clip flowline
        with rio.open(self.dem_path) as ds:
            bbox = ds.bounds
            dem_profile = ds.profile

        # transform bounding box to crs of flowline
        with fiona.open(self.custom_flowline_path, layer=layer) as ds:
            out_crs = ds.crs

        bbox = transform_bounds(dem_profile["crs"], out_crs, *bbox)

        # read custom flowline within bounding box and specified layer
        custom_flowline = gpd.read_file(
            self.custom_flowline_path,
            bbox=bbox,
            layer=layer,
        )

        # will reproject to dem_profile crs if necessary
        custom_flowline_raster = t.rasterize_flowline(
            flowline_gdf=custom_flowline,
            ref_profile=dem_profile,
            buffer=5,
        )

        # get file path for custom flowline raster
        self.custom_flowline_raster_path = t.get_file_path(
            custom_path=custom_path,
            project_dir=self.project_dir,
            dem_name=self.dem_path.stem,
            suffix="custom_flowline",
        )

        # write custom flowline raster
        out_profile = dem_profile.copy()
        out_profile.update(dtype="int16", nodata=-32768)
        t.write_raster(
            raster=custom_flowline_raster,
            profile=out_profile,
            file_path=self.custom_flowline_raster_path,
        )

        print(
            f"Custom flowline raster written to {str(self.custom_flowline_raster_path)}"
        )

    @t.time_it
    def extract_channel_network(
        self,
        custom_path: str | PathLike = None,
        write_cost_function: bool = False,
        use_custom_flowline: bool = False,
        no_flowline: bool = False,
        custom_weight_curvature: float | None = None,
        custom_weight_mfd_fac: float | None = None,
        custom_weight_binary_hand: float | None = None,
        custom_weight_custom_flowline: float | None = None,
    ):
        """
        Extract channel network. By default, curvature, flow accumulation, and
        binary HAND (information from NHD MR flowline) are used to calculate the
        cost function. A custom flowline such as the NHD HR flowline can be
        included in the cost function with use_custom_flowline=True. Only curvature
        and flow accumulataion will be considered if no_flowline=True. The cost
        function is calculated as the reciprocal of the weighted sum of these
        of these rasters. The cost function is thresholded and used to extract
        the channel network.

        Parameters
        ---------
        custom_path : `str`, `os.PathLike`, optional
            Custom path to save channel network raster. If not provided, channel
            network raster will be saved in project directory. The channel network
            vector file will have an identical name and path, but with the extension
            ".shp".
        write_cost_function : `bool`, optional
            Whether to write cost function raster to file. Default is False.
        use_custom_flowline : `bool`, optional
            Whether to use custom flowline raster in cost function. Default is False.
        no_flowline : `bool`, optional
            Whether to use the NHD MR flowline information (binary HAND raster)
            in the cost function. Default is False, so binary HAND raster will
            be used by default.
        custom_weight_curvature : `float`, optional
            Custom weight for curvature in cost function. Default is None.
        custom_weight_mfd_fac : `float`, optional
            Custom weight for flow accumulation in cost function. Default is None.
        custom_weight_binary_hand : `float`, optional
            Custom weight for binary HAND in cost function. Default is None.
        custom_weight_custom_flowline : `float`, optional
            Custom weight for custom flowline in cost function. Default is None.
        """

        required_files = [
            ("Curvature raster", self.curvature_path),
            ("Flow accumulation raster", self.mfd_fac_path),
            ("Endpoints csv", self.endpoints_path),
        ]

        t.check_attributes(required_files, "extract_channel_network")

        # read and prepare required rasters
        mfd_fac, fac_profile, _ = t.read_raster(self.mfd_fac_path)
        mfd_fac[mfd_fac == fac_profile["nodata"]] = np.nan
        mfd_fac = np.log(mfd_fac)
        mfd_fac = t.minmax_scale(mfd_fac)

        curvature, _, _ = t.read_raster(self.curvature_path)
        curvature[(curvature < -10) | (curvature > 10)] = np.nan
        curvature = t.minmax_scale(curvature)

        binary_hand, _, _ = t.read_raster(self.binary_hand_path)

        ### get cost surface array
        # use custom (likely NHD HR) flowline, NHD MR flowlines (binary HAND),
        # curvature, and flow accumulation in cost function
        if use_custom_flowline:
            required_files = [
                ("Custom flowline raster", self.custom_flowline_raster_path),
                ("Binary HAND raster", self.binary_hand_path),
            ]
            t.check_attributes(
                required_files,
                "extract_channel_network with use_custom_flowline=True",
            )
            # int16, 1 channel, 0 not
            custom_flowline_raster, _, _ = t.read_raster(
                self.custom_flowline_raster_path
            )
            weight_binary_hand = 0.75
            weight_custom_flowline = 1

        # use NHD MR flowlines (binary HAND), curvature, and flow accumulation
        # in cost function (no custom flowline)
        elif not no_flowline:
            required_files = [
                ("Binary HAND raster", self.binary_hand_path),
            ]
            t.check_attributes(required_files, "extract_channel_network")
            custom_flowline_raster = None
            weight_binary_hand = 0.75
            weight_custom_flowline = 0

        # use curvature and flow accumulation in cost function
        # (no NHD MR or custom flowlines)
        else:
            custom_flowline_raster = None
            weight_binary_hand = 0
            weight_custom_flowline = 0

        # default curvature and flow accumulation weights
        curv_weight_str = " (mean flow accumulation)"
        weight_curvature = np.nanmean(mfd_fac)
        weight_mfd_fac = 1

        # set custom weights if provided
        if custom_weight_curvature is not None:
            weight_curvature = custom_weight_curvature
            curv_weight_str = ""
        if custom_weight_mfd_fac is not None:
            weight_mfd_fac = custom_weight_mfd_fac
        if custom_weight_binary_hand is not None:
            weight_binary_hand = custom_weight_binary_hand
        if custom_weight_custom_flowline is not None:
            weight_custom_flowline = custom_weight_custom_flowline

        print("Cost function weights:")
        print(f"curvature          {weight_curvature:.4f}{curv_weight_str}")
        print(f"mfd_fac            {weight_mfd_fac:.4f}")
        print(f"binary_hand        {weight_binary_hand:.4f}")
        print(f"custom_flowline    {weight_custom_flowline:.4f}\n")

        weights_arrays = [
            (weight_curvature, curvature),
            (weight_mfd_fac, mfd_fac),
            (weight_binary_hand, binary_hand),
            (weight_custom_flowline, custom_flowline_raster),
        ]

        cost = t.get_combined_cost(weights_arrays)

        print(f"Cost min: {np.nanmin(cost)}")
        print(f"Cost max: {np.nanmax(cost)}")
        print(f"cost shape: {cost.shape}")

        if write_cost_function:
            self.cost_function_channel_path = t.get_file_path(
                custom_path=None,
                project_dir=self.project_dir,
                dem_name=self.dem_path.stem,
                suffix="cost_function_channel",
            )
            t.write_raster(
                raster=cost,
                profile=fac_profile,
                file_path=self.cost_function_channel_path,
            )
            print(
                f"Cost function written to {str(self.cost_function_channel_path)}"
            )
        # threshold cost surface
        # get 2.5% quantile
        cost_quantile = np.quantile(cost[~np.isnan(cost)], 0.025)
        artificial_high_cost = 100000
        cost[(cost >= cost_quantile) | np.isnan(cost)] = artificial_high_cost
        channel_network, stream_rowcol, stream_keys = t.get_channel_network(
            cost,
            self.endpoints_path,
            fac_profile["transform"],
        )

        ### write channel network raster and shapefile
        # raster
        self.channel_network_path = t.get_file_path(
            custom_path=custom_path,
            project_dir=self.project_dir,
            dem_name=self.dem_path.stem,
            suffix="channel_network",
        )
        out_profile = fac_profile.copy()
        out_profile.update(dtype="int16", nodata=-32768)
        t.write_raster(
            raster=channel_network,
            profile=out_profile,
            file_path=self.channel_network_path,
        )
        print(
            f"Channel network raster written to {str(self.channel_network_path)}"
        )
        # shapefile
        shp_path = self.channel_network_path.with_suffix(".shp")
        t.write_vector_lines(
            rowcol_list=stream_rowcol,
            keys=stream_keys,
            profile=fac_profile,
            dataset_name="ChannelNetwork",
            file_path=shp_path,
        )
        print(f"Channel network vector written to {str(shp_path)}")
