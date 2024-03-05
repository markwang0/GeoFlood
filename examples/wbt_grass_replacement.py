from pathlib import Path
from whitebox.whitebox_tools import WhiteboxTools

wbt = WhiteboxTools()
wbt.set_verbose_mode(True)  # False suppresses all output
wbt.set_working_dir(Path(Path(__file__).parent, "data"))

# input
filtered_dem = "OC1mTest_PM_filtered.tif"

# outputs
filtered_filled_dem = "OC1mTest_PM_filtered_filled.tif"
quinn_mfd_fac = "OC1mTest_PM_filtered_filled_mfd_fac.tif"
dinf_fac = "OC1mTest_PM_filtered_filled_dinf_fac.tif"
d8_fdr = "OC1mTest_PM_filtered_filled_d8_fdr.tif"
basins = "OC1mTest_PM_filtered_filled_d8_fdr_basins.tif"

# fill DEM depressions
wbt.fill_depressions(
    dem=filtered_dem,
    output=filtered_filled_dem,
    # fix_flats=True,
)

# calculate MFD flow accumulation
wbt.quinn_flow_accumulation(
    dem=filtered_filled_dem,
    output=quinn_mfd_fac,
    out_type="cells",
)

# calculate dinf flow accumulation
wbt.d_inf_flow_accumulation(
    i=filtered_filled_dem,
    output=dinf_fac,
    out_type="cells",
    pntr=False,
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
