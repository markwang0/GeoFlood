import geopandas as gpd
import numpy as np
import os
import psutil
import time
import rasterio as rio
import skfmm
import warnings

from functools import wraps
from numba import jit, prange
from pathlib import Path
from scipy import ndimage
from scipy.signal import convolve2d
from scipy.stats.mstats import mquantiles
from shapely.geometry import Point
from whitebox.whitebox_tools import WhiteboxTools

warnings.filterwarnings(
    action="ignore",
    message="Invalid value encountered",
    category=RuntimeWarning,
)


def path_property(name: str) -> property:
    """
    Create a path property with a storage name prefixed by an underscore.

    Parameters
    ----------
    name : `str`
        Name of path property.

    Returns
    -------
    prop : `property`
        Path property with storage name prefixed by an underscore.
    """
    storage_name = f"_{name}"

    @property
    def prop(self) -> str | os.PathLike:
        return getattr(self, storage_name)

    @prop.setter
    def prop(self, value: str | os.PathLike):
        # convert to Path object unless None
        if value is None:
            setattr(self, storage_name, value)
        else:
            if isinstance(value, (str, os.PathLike)):
                setattr(self, storage_name, Path(value))
            else:
                raise TypeError(
                    f"{name} must be a string or os.PathLike object"
                )

    return prop


# time function dectorator
def time_it(func: callable) -> callable:
    """
    Decorator function to time the execution of a function

    Parameters
    ----------
    func : `function`
        Function to time.

    Returns
    -------
    wrapper : `function`
        Wrapped function.
    """

    @wraps(func)
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


def check_attributes(
    attr_list: list[tuple[str, str | os.PathLike]], method
) -> None:
    """
    Check if required attributes are present.

    Parameters
    ----------
    attr_list : `list`
        List of (dataset name, path) tuples.
    """

    for name, path in attr_list:
        if path is None:
            raise ValueError(f"{name} must be created before running {method}")


def read_raster(
    raster_path: str,
) -> tuple[np.ndarray, rio.profiles.Profile | dict, float | int]:
    """
    Read a raster file and return the array, rasterio profile, and pixel scale.

    Parameters
    ----------
    raster_path : `str`
        Path to raster file.

    Returns
    -------
    raster : `np.ndarray`
        Array of raster values.
    profile : `rio.profiles.Profile` | `dict`
        Raster profile.
    pixel_scale : `float` | `int`
        Pixel scale of raster.
    """
    with rio.open(raster_path) as ds:
        raster = ds.read(1)
        profile = ds.profile
    pixel_scale = profile["transform"][0]
    # convert nodata to np.nan if dtype is float
    if "float" in profile["dtype"].lower():
        raster[raster == profile["nodata"]] = np.nan
    return raster, profile, pixel_scale


def write_raster(
    raster: np.ndarray,
    profile: rio.profiles.Profile | dict,
    file_path: str | os.PathLike,
    compression: str = "lzw",
) -> str | os.PathLike:
    """
    Write a raster file.

    Parameters
    ----------
    raster : `np.ndarray`
        Array of raster values.
    profile : `rio.profiles.Profile` | `dict`
        Raster profile.
    compression : `str`, optional
        Compression method. Default is "lzw".
    """

    profile.update(compress=compression)

    with rio.open(file_path, "w", **profile) as ds:
        ds.write(raster, 1)


def write_vector(
    rows: np.ndarray,
    cols: np.ndarray,
    profile: rio.profiles.Profile | dict,
    dataset_name: str,
    file_path: str | os.PathLike,
):
    """
    Write a vector file.

    Parameters
    ----------
    rows : `np.ndarray`
        Array of row values.
    cols : `np.ndarray`
        Array of column values.
    profile : `rio.profiles.Profile` | `dict`
        Raster profile.
    dataset_name : `str`
        Name of dataset.
    file_path : `str` | `os.PathLike`
        Path to write file.
    """

    transform = profile["transform"]
    crs = profile["crs"]

    # Use rasterio.transform.xy to project xx, yy points
    rows, cols = np.array(rows), np.array(cols)
    xy_proj = [
        rio.transform.xy(transform, rows[i], cols[i], offset="center")
        for i in range(len(rows))
    ]

    # Unpack the projected coordinates to easting and northing for UTM
    easting, northing = zip(*xy_proj)

    # Create Point geometries
    geometry = [Point(x, y) for x, y in zip(easting, northing)]

    # Create a GeoDataFrame with Northing and Easting fields
    gdf = gpd.GeoDataFrame(
        {
            "Type": [dataset_name] * len(geometry),
            "Northing": northing,
            "Easting": easting,
        },
        geometry=geometry,
    )

    # Set the CRS for the GeoDataFrame from rasterio CRS
    gdf.crs = crs

    # Write the GeoDataFrame to a shapefile
    gdf.to_file(file_path)


def get_file_path(
    custom_path: str | os.PathLike,
    project_dir: str | os.PathLike,
    dem_name: str,
    suffix: str,
    extension: str = "tif",
) -> str | os.PathLike:
    """
    Get file path.

    Parameters
    ----------
    custom_path : `str` | `os.PathLike`
        Optional custom path to save file.
    project_dir : `str` | `os.PathLike`
        Path to project directory.
    dem_name : `str`
        Name of DEM file.
    suffix : `str`
        Suffix to append to DEM filename. Only used if `write_path` is not
        provided.
    extension : `str`, optional
        File extension. Default is ".tif".

    Returns
    -------
    file_path : `str` | `os.PathLike`
        Path to write file.
    """
    if custom_path is not None:
        file_path = Path(custom_path)
    else:
        # append to DEM filename, save in project directory
        file_path = Path(
            project_dir,
            f"{dem_name}_{suffix}.{extension}",
        )

    return file_path


def get_WhiteboxTools(
    verbose: bool = False,
    compress: bool = True,
):
    """
    Get preconfigured WhiteboxTools instance.

    Parameters
    ----------
    verbose : `bool`, optional
        Verbose mode. Default is False.
    compress : `bool`, optional
        Compress rasters. Default is True.

    Returns
    -------
    wbt : `WhiteboxTools`
        WhiteboxTools instance.
    """
    wbt = WhiteboxTools()
    wbt.set_verbose_mode(verbose)
    wbt.set_compress_rasters(compress)
    return wbt


# Gaussian Filter
def simple_gaussian_smoothing(
    inputDemArray: np.ndarray,
    kernelWidth,
    diffusionSigmaSquared: float,
) -> np.ndarray:
    """
    smoothing input array with gaussian filter
    Code is vectorized for efficiency Harish Sangireddy

    Parameters
    ----------
    inputDemArray : `np.ndarray`
        Array of input DEM values.
    kernelWidth :
        Width of Gaussian kernel.
    diffusionSigmaSquared : `float`
        Diffusion sigma squared.

    Returns
    -------
    smoothedDemArray : `np.ndarray`
        Array of smoothed DEM values.
    """
    [Ny, Nx] = inputDemArray.shape
    halfKernelWidth = int((kernelWidth - 1) / 2)
    # Make a ramp array with 5 rows each containing [-2, -1, 0, 1, 2]
    x = np.linspace(-halfKernelWidth, halfKernelWidth, kernelWidth)
    y = x
    xv, yv = np.meshgrid(x, y)
    gaussianFilter = np.exp(
        -(xv**2 + yv**2) / (2 * diffusionSigmaSquared)
    )  # 2D Gaussian
    gaussianFilter = gaussianFilter / np.sum(gaussianFilter)  # Normalize
    print(inputDemArray[0, 0:halfKernelWidth])
    xL = np.nanmean(inputDemArray[:, 0:halfKernelWidth], axis=1)
    print(f"xL: {xL}")
    xR = np.nanmean(inputDemArray[:, Nx - halfKernelWidth : Nx], axis=1)
    print(f"xR: {xR}")
    part1T = np.vstack((xL, xL))
    part1 = part1T.T
    part2T = np.vstack((xR, xR))
    part2 = part2T.T
    eI = np.hstack((part1, inputDemArray, part2))
    xU = np.nanmean(eI[0:halfKernelWidth, :], axis=0)
    xD = np.nanmean(eI[Ny - halfKernelWidth : Ny, :], axis=0)
    part3 = np.vstack((xU, xU))
    part4 = np.vstack((xD, xD))
    # Generate the expanded DTM array, 4 pixels wider in both x,y directions
    eI = np.vstack((part3, eI, part4))
    # The 'valid' option forces the 2d convolution to clip 2 pixels off
    # the edges NaNs spread from one pixel to a 5x5 set centered on
    # the NaN
    fillvalue = np.nanmean(inputDemArray)
    smoothedDemArray = convolve2d(eI, gaussianFilter, "valid")
    del inputDemArray, eI
    return smoothedDemArray


def anisodiff(
    img: np.ndarray,
    niter: int,
    kappa: float,
    gamma: float,
    step: tuple[float, float] = (1.0, 1.0),
    option: str = "PeronaMalik2",
) -> np.ndarray:
    """
    Anisotropic diffusion.

    Parameters
    ----------
    img : `np.ndarray`
        Array of input image values.
    niter : `int`
        Number of iterations.
    kappa : `float`
        Edge threshold value.
    gamma : `float`
        Time increment.
    step : `tuple`, optional
        Step size. Default is (1.0, 1.0).
    option : `str`, optional
        Diffusion option. Default is "PeronaMalik2".

    Returns
    -------
    imgout : `np.ndarray`
        Array of filtered image values.
    """

    # initialize output array
    img = img.astype("float32")
    imgout = img.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()
    for _ in range(niter):
        # calculate the diffs
        deltaS[:-1, :] = np.diff(imgout, axis=0)
        deltaE[:, :-1] = np.diff(imgout, axis=1)
        if option == "PeronaMalik2":
            # gS = gs_diff(deltaS,kappa,step1)
            # gE = ge_diff(deltaE,kappa,step2)
            gS = 1.0 / (1.0 + (deltaS / kappa) ** 2.0) / step[0]
            gE = 1.0 / (1.0 + (deltaE / kappa) ** 2.0) / step[1]
        elif option == "PeronaMalik1":
            gS = np.exp(-((deltaS / kappa) ** 2.0)) / step[0]
            gE = np.exp(-((deltaE / kappa) ** 2.0)) / step[1]
        # update matrices
        E = gE * deltaE
        S = gS * deltaS
        # subtract a copy that has been shifted 'North/West' by one
        # pixel. don't ask questions. just do it. trust me.
        # **above comments from original GeoNet code**
        NS[:] = S
        EW[:] = E
        NS[1:, :] -= S[:-1, :]
        EW[:, 1:] -= E[:, :-1]
        # update the image
        mNS = np.isnan(NS)
        mEW = np.isnan(EW)
        NS[mNS] = 0
        EW[mEW] = 0
        NS += EW
        mNS &= mEW
        NS[mNS] = np.nan
        imgout += gamma * NS
    return imgout


def lambda_nonlinear_filter(
    nanDemArray: np.ndarray,
    demPixelScale: float,
    smoothing_quantile: float,
) -> float:
    """
    Compute the threshold lambda used in Perona-Malik nonlinear filtering.

    Parameters
    ----------
    nanDemArray : `np.ndarray`
        Array of input DEM values.
    demPixelScale : `float`
        Pixel scale of DEM.
    smoothing_quantile : `float`
        Quantile for calculating Perona-Malik nonlinear filter edge threshold
        value (kappa).

    Returns
    -------
    edgeThresholdValue : `float`
        Edge threshold value.
    """

    print("Computing slope of raw DTM")
    slopeXArray, slopeYArray = np.gradient(nanDemArray, demPixelScale)
    slopeMagnitudeDemArray = np.sqrt(slopeXArray**2 + slopeYArray**2)
    print(("DEM slope array shape:"), slopeMagnitudeDemArray.shape)

    # plot the slope DEM array
    # if defaults.doPlot == 1:
    #    raster_plot(slopeMagnitudeDemArray, 'Slope of unfiltered DEM')

    # Computation of the threshold lambda used in Perona-Malik nonlinear
    # filtering. The value of lambda (=edgeThresholdValue) is given by the 90th
    # quantile of the absolute value of the gradient.

    print("Computing lambda = q-q-based nonlinear filtering threshold")
    slopeMagnitudeDemArray = slopeMagnitudeDemArray.flatten()
    slopeMagnitudeDemArray = slopeMagnitudeDemArray[
        ~np.isnan(slopeMagnitudeDemArray)
    ]
    print("DEM smoothing Quantile:", smoothing_quantile)
    edgeThresholdValue = (
        mquantiles(
            np.absolute(slopeMagnitudeDemArray),
            smoothing_quantile,
        )
    ).item()
    print("Edge Threshold Value:", edgeThresholdValue)
    return edgeThresholdValue


def compute_dem_slope(
    filteredDemArray: np.ndarray,
    pixelDemScale: float,
) -> np.ndarray:
    """
    Compute slope of DEM.

    Parameters
    ----------
    filteredDemArray : `np.ndarray`
        Array of filtered DEM values.
    pixelDemScale : `float`
        Pixel scale of DEM.

    Returns
    -------
    slopeDemArray : `np.ndarray`
        Array of DEM slope values.
    """

    slopeYArray, slopeXArray = np.gradient(filteredDemArray, pixelDemScale)
    slopeDemArray = np.sqrt(slopeXArray**2 + slopeYArray**2)
    slopeMagnitudeDemArrayQ = slopeDemArray
    slopeMagnitudeDemArrayQ = np.reshape(
        slopeMagnitudeDemArrayQ,
        np.size(slopeMagnitudeDemArrayQ),
    )
    slopeMagnitudeDemArrayQ = slopeMagnitudeDemArrayQ[
        ~np.isnan(slopeMagnitudeDemArrayQ)
    ]
    # Computation of statistics of slope
    print(" slope statistics")
    print(
        " angle min:",
        np.arctan(np.percentile(slopeMagnitudeDemArrayQ, 0.1)) * 180 / np.pi,
    )
    print(
        " angle max:",
        np.arctan(np.percentile(slopeMagnitudeDemArrayQ, 99.9)) * 180 / np.pi,
    )
    print(" mean slope:", np.nanmean(slopeDemArray))
    print(" stdev slope:", np.nanstd(slopeDemArray))
    slopeDemArray[np.isnan(filteredDemArray)] = np.nan
    return slopeDemArray


def compute_dem_curvature(
    filteredDemArray: np.ndarray,
    pixelDemScale: float,
    curvatureCalcMethod: str,
) -> np.ndarray:
    """
    Compute curvature of DEM.

    Parameters
    ----------
    filteredDemArray : `np.ndarray`
        Array of DEM values.
    pixelDemScale : `float`
        Pixel scale of DEM.
    curvatureCalcMethod : `str`, optional
        Method for calculating curvature. Options include:
        - "geometric": TODO: detailed description
        - "laplacian": TODO: detailed description
        Default is "geometric".

    Returns
    -------
    curvatureDemArray : `np.ndarray`
        Array of DEM curvature values.
    """

    # OLD:
    # gradXArray, gradYArray = np.gradient(demArray, pixelDemScale)
    # NEW:
    gradYArray, gradXArray = np.gradient(filteredDemArray, pixelDemScale)

    slopeArrayT = np.sqrt(gradXArray**2 + gradYArray**2)
    if curvatureCalcMethod == "geometric":
        # Geometric curvature
        print(" using geometric curvature")
        gradXArrayT = np.divide(gradXArray, slopeArrayT)
        gradYArrayT = np.divide(gradYArray, slopeArrayT)
    elif curvatureCalcMethod == "laplacian":
        # do nothing..
        print(" using laplacian curvature")
        gradXArrayT = gradXArray
        gradYArrayT = gradYArray

    # NEW:
    tmpy, gradGradXArray = np.gradient(gradXArrayT, pixelDemScale)
    gradGradYArray, tmpx = np.gradient(gradYArrayT, pixelDemScale)

    curvatureDemArray = gradGradXArray + gradGradYArray
    curvatureDemArray[np.isnan(curvatureDemArray)] = 0
    del tmpy, tmpx
    # Computation of statistics of curvature
    print(" curvature statistics")
    tt = curvatureDemArray[~np.isnan(curvatureDemArray)]
    print(" non-nan curvature cell number:", tt.shape[0])
    finiteCurvatureDemList = curvatureDemArray[np.isfinite(curvatureDemArray)]
    print(" non-nan finite curvature cell number:", end=" ")
    finiteCurvatureDemList.shape[0]
    curvatureDemMean = np.nanmean(finiteCurvatureDemList)
    curvatureDemStdDevn = np.nanstd(finiteCurvatureDemList)
    print(" mean: ", curvatureDemMean)
    print(" standard deviation: ", curvatureDemStdDevn)
    curvatureDemArray[np.isnan(filteredDemArray)] = np.nan
    return curvatureDemArray


def get_skeleton(
    inputArray1: np.ndarray,
    threshold1: float,
    inputArray2: np.ndarray = None,
    threshold2: float = None,
) -> np.ndarray:
    """
    Creates a channel skeleton by thresholding grid measures such as flow or curvature.
    Can operate on single or dual thresholds depending on input.

    Parameters
    ----------
    inputArray1 : `np.ndarray`
        First array of input values.
    threshold1 : float
        Threshold value for the first input array.
    inputArray2 : `np.ndarray`, optional
        Second array of input values. If provided, dual thresholding will be applied.
    threshold2 : `float`, optional
        Threshold value for the second input array, required if inputArray2 is provided.

    Returns
    -------
    skeletonArray : `np.ndarray`
        Binary array (dtype: int) of skeleton values.
    """

    mask1 = inputArray1 > threshold1

    if inputArray2 is not None and threshold2 is not None:
        mask2 = inputArray2 > threshold2
        skeletonArray = (mask1 & mask2).astype(int)
    else:
        skeletonArray = mask1.astype(int)

    return skeletonArray


@jit(nopython=True, parallel=True)
def get_fmm_points(
    basins,
    outlets,
    basin_elements,
    area_threshold,
):
    fmmX = []
    fmmY = []
    basins_ravel = basins.ravel()
    n_pixels = basins.size
    for label in prange(outlets.shape[1]):
        numelements = np.sum(basins_ravel == (label + 1))
        percentBasinArea = numelements * 100.00001 / n_pixels
        if (percentBasinArea > area_threshold) and (
            numelements > basin_elements
        ):
            fmmX.append(outlets[1, label])
            fmmY.append(outlets[0, label])

    return np.array([fmmY, fmmX])


def get_ram_usage() -> str:
    """
    Get the current system RAM usage and return it in a human-readable format.

    Returns:
    --------
    str
        A string representing the current RAM usage in GB with 2 decimal places.
    """
    # Fetch RAM usage information
    mem = psutil.virtual_memory()
    # Convert bytes to GB for more human-friendly reading
    avail_memory_gb = mem.available / (1024**3)
    total_memory_gb = mem.total / (1024**3)
    total_less_avail = total_memory_gb - avail_memory_gb

    return f"RAM usage: {total_less_avail:.2f}/{total_memory_gb:.0f} GB ({mem.percent}%)"


def fast_marching(
    fmm_start_points,
    basins,
    mfd_fac,
    cost_function,
):
    # Fast marching
    print("Performing fast marching")
    # Do fast marching for each sub basin
    geodesic_distance = np.zeros_like(basins)
    geodesic_distance[geodesic_distance == 0] = np.inf
    fmm_total_iter = len(fmm_start_points[0])
    for i in range(fmm_total_iter):
        basinIndexList = basins[fmm_start_points[0, i], fmm_start_points[1, i]]
        # print("start point :", fmm_start_points[:, i])
        maskedBasin = np.zeros_like(basins)
        maskedBasin[basins == basinIndexList] = 1
        maskedBasinFAC = np.zeros_like(basins)
        maskedBasinFAC[basins == basinIndexList] = mfd_fac[
            basins == basinIndexList
        ]

        phi = np.zeros_like(cost_function)
        speed = np.zeros_like(cost_function)
        phi[maskedBasinFAC != 0] = 1
        speed[maskedBasinFAC != 0] = cost_function[maskedBasinFAC != 0]
        phi[fmm_start_points[0, i], fmm_start_points[1, i]] = -1
        del maskedBasinFAC
        # print RAM usage per iteration
        print(f"FMM {i+1}/{fmm_total_iter}: {get_ram_usage()}")
        try:
            travel_time = skfmm.travel_time(phi, speed, dx=0.01)
        except IOError as e:
            print("Error in calculating skfmm travel time")
            print("Error in catchment: ", basinIndexList)
            print("I/O error({0}): {1}".format(e.errno, e.strerror))
            # setting travel time to empty array
            travel_time = np.nan * np.zeros_like(cost_function)
            # if defaults.doPlot == 1:
            #    raster_point_plot(speed, fmm_start_points[:,i],
            #                      'speed basin Index'+str(basinIndexList))
            # plt.contour(speed,cmap=cm.coolwarm)
            #    raster_point_plot(phi, fmm_start_points[:,i],
            #                      'phi basin Index'+str(basinIndexList))
        except ValueError:
            print("Error in calculating skfmm travel time")
            print("Error in catchment: ", basinIndexList)
            print("That was not a valid number")
        geodesic_distance[maskedBasin == 1] = travel_time[maskedBasin == 1]
    geodesic_distance[maskedBasin == 1] = travel_time[maskedBasin == 1]
    geodesic_distance[geodesic_distance == np.inf] = np.nan
    # Plot the geodesic array
    # if defaults.doPlot == 1:
    #    geodesic_contour_plot(geodesic_distance,
    #                          'Geodesic distance array (travel time)')
    return geodesic_distance


def get_channel_heads(
    combined_skeleton: np.ndarray,
    geodesic_distance: np.ndarray,
    channel_head_median_dist: int,
    max_channel_heads: int,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """
    Through the histogram of sorted_label_counts
    (skeletonNumElementsList minus the maximum value which
    corresponds to the largest connected element of the skeleton) we get the
    size of the smallest elements of the skeleton, which will likely
    correspond to small isolated convergent areas. These elements will be
    excluded from the search of end points.
    """
    # Locating end points
    print("Locating skeleton end points")
    structure = np.ones((3, 3))
    labeled, num_labels = ndimage.label(combined_skeleton, structure=structure)
    print("Counting the number of elements of each connected component")
    lbls = np.arange(1, num_labels + 1)
    label_counts = ndimage.labeled_comprehension(
        input=combined_skeleton,
        labels=labeled,
        index=lbls,
        func=np.count_nonzero,
        out_dtype=int,
        default=0,
    )
    sorted_label_counts = np.sort(label_counts)
    num_bins = int(np.floor(np.sqrt(len(sorted_label_counts))))
    histarray, bin_edges = np.histogram(sorted_label_counts[:-1], num_bins)
    # if defaults.doPlot == 1:
    #     raster_plot(labeled, "Skeleton Labeled Array elements Array")
    # Create skeleton gridded array
    labeled_set, label_indices = np.unique(labeled, return_inverse=True)
    skeleton_gridded_array = np.array(
        [label_counts[x - 1] for x in labeled_set]
    )[label_indices].reshape(labeled.shape)
    # if defaults.doPlot == 1:
    #     raster_plot(
    #         skeleton_gridded_array, "Skeleton Num elements Array"
    #     )
    # Elements smaller than count_threshold are not considered in the
    # channel_heads detection
    count_threshold = bin_edges[2]
    print(f"Skeleton region size threshold: {str(count_threshold)}")
    # Scan the array for finding the channel heads
    print("Continuing to locate skeleton endpoints")
    channel_heads = []
    nrows, ncols = combined_skeleton.shape
    channel_heads = jit_channel_heads(
        labeled,
        skeleton_gridded_array,
        geodesic_distance,
        nrows,
        ncols,
        count_threshold,
        channel_head_median_dist,
        max_channel_heads,
    )
    channel_heads = np.transpose(channel_heads)
    ch_rows = channel_heads[0]
    ch_cols = channel_heads[1]
    return ch_rows, ch_cols


@jit(nopython=True)
def jit_channel_heads(
    labeled,
    skeleton_gridded_array,
    geodesic_distance,
    nrows,
    ncols,
    count_threshold,
    channel_head_median_dist,
    max_channel_heads,
):
    """
    Numba JIT-compiled function for finding channel heads.
    """
    # pre-allocate array of channel heads
    channel_heads = np.zeros((max_channel_heads, 2), dtype=np.int32)
    ch_count = 0

    for i in range(nrows):
        for j in range(ncols):
            if (
                labeled[i, j] != 0
                and skeleton_gridded_array[i, j] >= count_threshold
            ):
                my, py, mx, px = i - 1, nrows - i, j - 1, ncols - j
                xMinus, xPlus = min(channel_head_median_dist, mx), min(
                    channel_head_median_dist, px
                )
                yMinus, yPlus = min(channel_head_median_dist, my), min(
                    channel_head_median_dist, py
                )

                search_geodesic_box = geodesic_distance[
                    i - yMinus : i + yPlus + 1, j - xMinus : j + xPlus + 1
                ]
                search_skeleton_box = labeled[
                    i - yMinus : i + yPlus + 1, j - xMinus : j + xPlus + 1
                ]

                v = search_skeleton_box == labeled[i, j]
                v1 = v & (search_geodesic_box > geodesic_distance[i, j])

                if not np.any(v1):
                    channel_heads[ch_count] = [i, j]
                    ch_count += 1
                # Trim to the actual number of channel heads found
                # warn if max_channel_heads was exceeded
                if ch_count > max_channel_heads:
                    print(
                        f"Warning: max_channel_heads ({max_channel_heads}) exceeded. "
                        "Consider increasing max_channel_heads"
                    )
    return channel_heads[:ch_count]


def get_endpoints(in_shp: str | os.PathLike) -> gpd.GeoDataFrame:
    # Read the shapefile into a GeoDataFrame
    gdf = gpd.read_file(in_shp)

    # Extract start and end points directly into the GeoDataFrame
    gdf["START_X"] = gdf.geometry.apply(lambda geom: geom.coords[0][0])
    gdf["START_Y"] = gdf.geometry.apply(lambda geom: geom.coords[0][1])
    gdf["END_X"] = gdf.geometry.apply(lambda geom: geom.coords[-1][0])
    gdf["END_Y"] = gdf.geometry.apply(lambda geom: geom.coords[-1][1])

    # Create a RiverID column
    gdf["RiverID"] = range(1, len(gdf) + 1)

    # Select and order the columns for the output CSV
    endpoints = gdf[["RiverID", "START_X", "START_Y", "END_X", "END_Y"]]

    return endpoints
