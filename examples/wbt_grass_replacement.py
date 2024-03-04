from whitebox.whitebox_tools import WhiteboxTools

wbt = WhiteboxTools()
wbt.set_verbose_mode(False)
wbt.set_working_dir("/home/mark/repos/wip_geoflood/examples/data/")

# input
filtered_dem = "OC1mTest_PM_filtered.tif"

# outputs
filtered_filled_dem = "OC1mTest_PM_filtered_filled.tif"
quinn_mfd_fac = "OC1mTest_PM_filtered_filled_mfd_fac.tif"
d8_fdr = "OC1mTest_PM_filtered_filled_d8_fdr.tif"
basins = "OC1mTest_PM_filtered_filled_d8_fdr_basins.tif"

# fill DEM depressions
wbt.fill_depressions(
    dem=filtered_dem,
    output=filtered_filled_dem,
    # fix_flats=True,
)

# calculate flow accumulation
wbt.quinn_flow_accumulation(
    dem=filtered_filled_dem,
    output=quinn_mfd_fac,
    out_type="cells",
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
