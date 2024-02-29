`pygeonet_grass_py3.py` general workflow

1. read in `"PM_filtered_grassgis.tif"`
2. rename to `"OC1mTest.tif"` [demFileName]
3. Find flow accumulation and flow directions [`r.watershed`](https://grass.osgeo.org/grass83/manuals/r.watershed.html)
    - if shape x > 4000 | y > 4000 --> use swap
    - `-a` Use positive flow accumulation even for likely underestimates
    - elevation=filtered DEM
    - threshold=1500 (thresholdAreaSubBasinIndexing default) (Minimum size of exterior watershed basin)
    - Outputs
        - accumulation raster `acc1v23`
        - drainage dir raster `dra1v23`
4. Find outlets: cells with negative flow dir [`r.mapcalc`](https://grass.osgeo.org/grass82/manuals/r.mapcalc.html)
    - `outletmap = if(dra1v23 >= 0,null(),1)`
    - output = `outletmap`
5. Convert outlets to raster [`r.to.vect`](https://grass.osgeo.org/grass82/manuals/r.to.vect.html)
    - output = `outletsmapvec`
6. Delineate basins using outlets [`r.stream.basins`](https://grass.osgeo.org/grass82/manuals/addons/r.stream.basins.html)
    - input = `dra1v23`, `outletsmapvec`
    - output = `outletsbasins`
7. Save output rasters to `.tif`
    - `outletmap`
        - `"[dem_name]_outlets.tif"`
        - do arithmetic on drainage raster
        - `Float32`
    - `acc1v23`
        - `"[dem_name]_fac.tif"`
        - flow accumulation raster
        - `Float64`
    - `dra1v23`
        - `"[dem_name]_fdr.tif"`
        - flow drainage direction raster
        - `Int32`
    - `outletbasins`
        - `"[dem_name]_fdr.tif"`
        - basins raster from drainage raster
        - `Int16`
8. Stop timer