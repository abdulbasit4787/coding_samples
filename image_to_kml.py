def small_image_kml():
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    from osgeo import gdal
    import cv2
    import numpy as np
    import simplekml
    import math

    tif_file = r'C:\Users\DELL\PycharmProjects\building_project\hangzhou_mask.tif'
    ds = gdal.Open(tif_file)
    band = ds.GetRasterBand(1)
    data = band.ReadAsArray()
    binary_data = np.where(data == 255, 1, 0).astype(np.uint8)
    binary_data = binary_data.astype(np.uint8)
    contours, hierarchy = cv2.findContours(binary_data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    kml = simplekml.Kml()
    for contour in contours:
        polygon = kml.newpolygon(name='Building Footprint')
        # points = [(point[0][0], point[0][1]) for point in contour]
        points = [(point[0][1], point[0][0]) for point in contour[::-1]]
        # points = [(ds.RasterXSize - point[0], ds.RasterYSize - point[1]) for point in points]

        angle = math.pi / 2
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        rotated_points = [(cos_angle * x + sin_angle * y, -sin_angle * x + cos_angle * y) for (x, y) in points]

        polygon.outerboundaryis = rotated_points

    kml_file = r'C:\Users\DELL\PycharmProjects\building_project\hangzhou.kml'
    kml.save(kml_file)

def large_image_kml():
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    import numpy as np
    import cv2
    from osgeo import gdal
    import simplekml

    image_path = r'C:\Users\DELL\PycharmProjects\building_project\hangzhou_mask.tif'
    # Load the image
    img = Image.open(image_path)
    # Convert to numpy array
    binary_data = np.array(img)

    # Find contours
    contours, hierarchy = cv2.findContours(binary_data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Load the geotransform from the TIFF
    ds = gdal.Open(image_path)
    geotransform = ds.GetGeoTransform()

    """def pixel_to_geo(coord, geotransform):
        x, y = coord
        geo_x = geotransform[0] + x * geotransform[1] + y * geotransform[2]
        geo_y = geotransform[3] + x * geotransform[4] + y * geotransform[5]
        return geo_x, geo_y"""

    def pixel_to_geo(coord, geotransform, max_y):
        """Convert pixel coordinates to geographic coordinates."""
        x, y = coord
        y = max_y - y  # Flip the y-coordinate
        geo_x = geotransform[0] + x * geotransform[1] + y * geotransform[2]
        geo_y = geotransform[3] + x * geotransform[4] + y * geotransform[5]
        return geo_x, geo_y

    # Create a KML object
    kml = simplekml.Kml()
    max_y = binary_data.shape[0]  # Get the height of the image
    for contour in contours:
        # Convert contour points to geographic coordinates
        geo_coords = [pixel_to_geo(tuple(pt[0]), geotransform, max_y) for pt in contour]

        # Create a polygon and add it to the KML
        pol = kml.newpolygon(outerboundaryis=geo_coords)

    # Save the KML file
    kml.save(r'C:\Users\DELL\PycharmProjects\building_project\hangzhou.kml')
large_image_kml()