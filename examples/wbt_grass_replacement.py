import geopandas as gpd
import rasterio as rio

from pathlib import Path
from rasterio.features import rasterize
from whitebox.whitebox_tools import WhiteboxTools

wbt = WhiteboxTools()
wbt.set_verbose_mode(True)  # False suppresses all output
wbt.set_compress_rasters(True)
workdir = Path(Path(__file__).parent, "data")
wbt.set_working_dir(workdir)

# input
filtered_dem = "OC1mTest_filtered.tif"

# outputs
filtered_filled_dem = "OC1mTest_filtered_filled.tif"
quinn_mfd_fac = "OC1mTest_filtered_filled_mfd_fac.tif"
dinf_fac = "OC1mTest_filtered_filled_dinf_fac.tif"
dinf_fdr = "OC1mTest_filtered_filled_dinf_fdr.tif"
d8_fdr = "OC1mTest_filtered_filled_d8_fdr.tif"
basins = "OC1mTest_filtered_filled_d8_fdr_basins.tif"

# fill DEM depressions
wbt.fill_depressions(
    dem=filtered_dem,
    output=filtered_filled_dem,
    fix_flats=True,
)

# calculate MFD flow accumulation
wbt.quinn_flow_accumulation(
    dem=filtered_filled_dem,
    output=quinn_mfd_fac,
    out_type="cells",
)

# calculate dinf flow direction
wbt.d_inf_pointer(
    dem=filtered_filled_dem,
    output=dinf_fdr,
)

# calculate dinf flow accumulation
wbt.d_inf_flow_accumulation(
    i=dinf_fdr,
    output=dinf_fac,
    out_type="cells",
    pntr=True,
)

# calculate flow direction
wbt.d8_pointer(
    dem=filtered_filled_dem,
    output=d8_fdr,
)

# delineate basins
wbt.basins(
    d8_pntr=d8_fdr,
    output=basins,
)

# delineate segment catchments
# first rasterize channelSegment.shp for wbt.watershed pour_pts
with rio.open(Path(workdir, "OC1mTest.tif")) as ds:
    # don't read band into memory, just get profile
    ds_profile = ds.profile
gdf = gpd.read_file(Path(workdir, "OC1mTest_channelSegment.shp"))
streams_raster = rasterize(
    zip(gdf.geometry, gdf["HYDROID"]),
    out_shape=(ds_profile["height"], ds_profile["width"]),
    dtype="int",
    transform=ds_profile["transform"],
    fill=0,
)
with rio.open(Path(workdir, "streams_raster.tif"), "w", **ds_profile) as ds:
    ds.write(streams_raster, 1)
wbt.watershed(
    d8_pntr=d8_fdr,
    pour_pts="streams_raster.tif",
    output="streams_wbt_watersheds.tif",
)
