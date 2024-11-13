def add_crs():
    import rasterio
    from rasterio.crs import CRS

    # Open the raster file in 'read' mode
    with rasterio.open('/home/nicku/PycharmProjects/test/test_folder/mask.tif', 'r') as src:
        # Copy the metadata and data
        meta = src.meta
        data = src.read()

    # Update the CRS in the metadata
    meta['crs'] = CRS.from_epsg(4326)  # Replace 4326 with the EPSG code of the CRS you want to use

    # Open the raster file in 'write' mode and update the metadata
    with rasterio.open('/home/nicku/PycharmProjects/test/test_folder/mask.tif', 'w', **meta) as dst:
        dst.write(data)

def copy_tif_metadata_to_tif():
    from osgeo import gdal, osr

    # Load the source GeoTIFF
    src = gdal.Open(r'C:\Users\DELL\PycharmProjects\building_footprints\new_dataset - Copy\taizhou_dataset\mask_new.tif')
    src_proj = src.GetProjection()
    src_geotrans = src.GetGeoTransform()

    # Load the destination GeoTIFF
    dst = gdal.Open(r'C:\Users\DELL\PycharmProjects\building_footprints\new_dataset - Copy\taizhou_dataset\rgb.tif', gdal.GA_Update)

    # Set the same coordinate system on the destination GeoTIFF
    dst_proj = osr.SpatialReference()
    dst_proj.ImportFromWkt(src_proj)
    dst.SetProjection(dst_proj.ExportToWkt())

    # Set the same geotransform on the destination GeoTIFF
    dst.SetGeoTransform(src_geotrans)

    src = None
    dst = None

copy_tif_metadata_to_tif()