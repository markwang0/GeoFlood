import numpy as np
import scipy.signal as conv2
from pathlib import Path


from scipy.stats.mstats import mquantiles


# Gaussian Filter
def simple_gaussian_smoothing(
    inputDemArray, kernelWidth, diffusionSigmaSquared
) -> np.ndarray:
    """
    smoothing input array with gaussian filter
    Code is vectorized for efficiency Harish Sangireddy
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
    smoothedDemArray = conv2.convolve2d(eI, gaussianFilter, "valid")
    del inputDemArray, eI
    return smoothedDemArray


def anisodiff(
    img,
    niter,
    kappa,
    gamma,
    step=(1.0, 1.0),
    option="PeronaMalik2",
) -> np.ndarray:
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
    nanDemArray,
    demPixelScale,
    smoothing_quantile,
) -> float:
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


def compute_dem_slope(filteredDemArray, pixelDemScale) -> np.ndarray:
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
    return slopeDemArray


def compute_dem_curvature(
    demArray,
    pixelDemScale,
    curvatureCalcMethod,
) -> np.ndarray:
    # OLD:
    # gradXArray, gradYArray = np.gradient(demArray, pixelDemScale)
    # NEW:
    gradYArray, gradXArray = np.gradient(demArray, pixelDemScale)

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
    finiteCurvatureDemList = curvatureDemArray[
        np.isfinite(curvatureDemArray[:])
    ]
    print(" non-nan finite curvature cell number:", end=" ")
    finiteCurvatureDemList.shape[0]
    curvatureDemMean = np.nanmean(finiteCurvatureDemList)
    curvatureDemStdDevn = np.nanstd(finiteCurvatureDemList)
    print(" mean: ", curvatureDemMean)
    print(" standard deviation: ", curvatureDemStdDevn)
    return curvatureDemArray
