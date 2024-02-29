import numpy as np
from pysheds.grid import Grid
from pysheds.view import Raster

grid = Grid.from_raster("sablake_10m.tif")
dem = grid.read_raster("sablake_10m.tif")
flooded_dem = grid.fill_depressions(dem)
inflated_dem = grid.resolve_flats(flooded_dem)

fdir_mfd = grid.flowdir(inflated_dem, routing="mfd")  # shape (8, m, n)
acc = grid.accumulation(fdir_mfd, routing="mfd")

fdir_mfd_max = np.argmax(fdir_mfd, axis=0)  # shape (m, n)

# need float64 dtype for nodata value
fdir_mfd_max_raster = Raster(
    np.array(fdir_mfd_max).astype("float64"),
    grid.viewfinder,
)

grid.to_raster(
    fdir_mfd_max_raster,
    "fdir_mfd_max_raster_test.tif",
)
