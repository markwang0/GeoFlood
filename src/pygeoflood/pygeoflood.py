import numpy as np

from . import tools as t
from os import PathLike
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
            Path to project directory. Default is the directory containing the
            DEM. All outputs will be saved to this directory.
        """

        # automatically becomes a pathlib.Path object if not aleady
        self.dem_path = dem_path
        # if no project_dir is provided, use dir containing DEM
        if project_dir is not None:
            self.project_dir = project_dir
        else:
            self.project_dir = dem_path.parent

    # String representation of class
    def __repr__(self):
        attrs = "\n    ".join(
            (f'{k[1:]}="{v}",' if isinstance(v, (str, Path)) else f"{k[1:]}={v!r},")
            for k, v in self.__dict__.items()
            if v is not None
        )
        return f"{self.__class__.__name__}(\n    {attrs}\n)"

    # create properties for class with getters and setters
    # setters ensure each path property is a pathlib.Path object
    dem_path = t.path_property("dem_path")
    project_dir = t.path_property("project_dir")
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

        filteredDemArray = t.anisodiff(
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
            raster=filteredDemArray,
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
        if self.filtered_dem_path is None:
            raise ValueError(
                "Filtered DEM must be created before calculating slope and curvature"
            )

        # read filtered DEM
        filtered_dem, filtered_dem_profile, pixel_scale = t.read_raster(
            self.filtered_dem_path
        )

        print("Computing slope")
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
        if self.filtered_dem_path is None:
            raise ValueError(
                "Filtered DEM must be created before calculating curvature"
            )

        # read filtered DEM
        filtered_dem, filtered_dem_profile, pixel_scale = t.read_raster(
            self.filtered_dem_path
        )

        print("Computing curvature")
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
        if self.filtered_dem_path is None:
            raise ValueError("Filtered DEM must be created before filling depressions")

        print("Filling depressions on the filtered DEM")

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
        if self.filled_path is None:
            raise ValueError(
                "Filled DEM must be created before calculating MFD flow accumulation"
            )

        print("Calculating MFD flow accumulation")

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

        print(f"MFD flow accumulation raster written to {str(self.mfd_fac_path)}")

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
        if self.filled_path is None:
            raise ValueError(
                "Filled DEM must be created before calculating D8 flow direction"
            )

        print("Calculating D8 flow direction")

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
        filtered_dem, filtered_profile, _ = t.read_raster(self.filtered_dem_path)
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
        if self.d8_fdr_path is None:
            raise ValueError(
                "D8 flow direction raster must be created before creating outlets raster"
            )

        print("Creating outlets raster")

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
        if self.d8_fdr_path is None:
            raise ValueError(
                "D8 flow direction raster must be created before delineating basins"
            )

        print("Delineating basins")

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

        required_rasters = [
            ("Curvature", self.curvature_path),
            ("Flow accumulation", self.mfd_fac_path),
        ]

        for raster, path in required_rasters:
            if path is None:
                raise ValueError(
                    f"{raster} raster must be created before defining skeleton"
                )

        # get skeleton from curvature only
        curvature, curvature_profile, _ = t.read_raster(self.curvature_path)
        finite_curvature = curvature[np.isfinite(curvature)]
        curvature_mean = np.nanmean(finite_curvature)
        curvature_std = np.nanstd(finite_curvature)
        print("Curvature mean: ", curvature_mean)
        print("Curvature standard deviation: ", curvature_std)
        print(f"Curvature Projection: {str(curvature_profile['crs'])}")
        thresholdCurvatureQQxx = 1.5
        curvature_threshold = curvature_mean + thresholdCurvatureQQxx * curvature_std
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
            print(f"Curvature skeleton written to {str(self.curvature_skeleton_path)}")

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
        print(f"Combined skeleton written to {str(self.combined_skeleton_path)}")
