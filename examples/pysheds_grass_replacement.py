import numpy as np
from pathlib import Path
from pysheds.grid import Grid
from pysheds.view import Raster

filt_dem_path = Path("data", "OC1mTest_PM_filtered.tif")

grid = Grid.from_raster(filt_dem_path)
filt_dem = grid.read_raster(filt_dem_path)
flooded_dem = grid.fill_depressions(filt_dem)
inflated_dem = grid.resolve_flats(flooded_dem)

# MFD flow direction
grass_dirmap = (2, 1, 8, 7, 6, 5, 4, 3)
# shape (8, m, n)
fdir_mfd = grid.flowdir(
    inflated_dem,
    dirmap=grass_dirmap,
    routing="mfd",
)
# write max MFD flow direction to raster
fdir_mfd_max = np.argmax(fdir_mfd, axis=0)  # shape (m, n)
fdir_mfd_max_raster = Raster(
    np.array(fdir_mfd_max).astype("float64"),
    grid.viewfinder,
)
grid.to_raster(
    fdir_mfd_max_raster,
    Path("data", "pysheds_max_mfd_fdr.tif"),
    blockxsize=16,
    blockysize=16,
)

# MFD flow accumulation
acc_mfd = grid.accumulation(
    fdir_mfd,
    dirmap=grass_dirmap,
    routing="mfd",
)

# dinf flow direction and accumulation
fdir_inf = grid.flowdir(
    inflated_dem,
    dirmap=grass_dirmap,
    routing="dinf",
)
acc_inf = grid.accumulation(
    fdir_inf,
    dirmap=grass_dirmap,
    routing="dinf",
)
