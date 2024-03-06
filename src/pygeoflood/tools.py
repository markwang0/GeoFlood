import numpy as np
import os
import time
from pathlib import Path
import rasterio as rio
import warnings

from functools import wraps
from scipy.signal import convolve2d
from scipy.stats.mstats import mquantiles
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
        # always convert to Path object
        if isinstance(value, (str, os.PathLike)):
            setattr(self, storage_name, Path(value))
        else:
            raise TypeError(f"{name} must be a string or os.PathLike object")

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


def get_file_path(
    custom_path: str | os.PathLike,
    project_dir: str | os.PathLike,
    dem_name: str,
    suffix: str,
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
            f"{dem_name}_{suffix}.tif",
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
    gaussianFilter = gaussianFilter / np.sum(gaussianFilter[:])  # Normalize
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
    fillvalue = np.nanmean(inputDemArray[:])
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
    slopeMagnitudeDemArray = slopeMagnitudeDemArray[~np.isnan(slopeMagnitudeDemArray)]
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
    print(" mean slope:", np.nanmean(slopeDemArray[:]))
    print(" stdev slope:", np.nanstd(slopeDemArray[:]))
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
    tt = curvatureDemArray[~np.isnan(curvatureDemArray[:])]
    print(" non-nan curvature cell number:", tt.shape[0])
    finiteCurvatureDemList = curvatureDemArray[np.isfinite(curvatureDemArray[:])]
    print(" non-nan finite curvature cell number:", end=" ")
    finiteCurvatureDemList.shape[0]
    curvatureDemMean = np.nanmean(finiteCurvatureDemList)
    curvatureDemStdDevn = np.nanstd(finiteCurvatureDemList)
    print(" mean: ", curvatureDemMean)
    print(" standard deviation: ", curvatureDemStdDevn)
    curvatureDemArray[np.isnan(filteredDemArray)] = np.nan
    return curvatureDemArray
