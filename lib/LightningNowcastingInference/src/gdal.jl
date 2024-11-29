using GDAL
using ClimateBase

"""
Reescales a ClimArray from [0, 1] to [0, 255] and stores it as a geotiff raster format.
"""
function save_tiff_uint8(dstpath::AbstractString, src::ClimArray)
    mktemp() do path, _
        ncwrite(path, src)
        ds = GDAL.gdalopen(path, GDAL.GA_ReadOnly)
        options = GDAL.gdaltranslateoptionsnew(["-ot", "Byte", "-scale", "0", "1", "0", "255"], C_NULL)
        GDAL.gdaltranslate(dstpath, ds, options, C_NULL)
    end
end

