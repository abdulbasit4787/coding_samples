IMAGE_DIR = r"/home/nick/PycharmProjects/project1/taizhou_dataset/玉环市_卫图3/17.tif"
SAVE_DIR = r"/home/nick/PycharmProjects/project1/taizhou_dataset/tiles_18"



import os
from itertools import product
import rasterio as rio
from rasterio import windows

in_path = '/home/nick/PycharmProjects/project1/taizhou_dataset/玉环市_卫图3/'
input_filename = '17.tif'

out_path = '/home/nick/PycharmProjects/project1/taizhou_dataset/tiles_18'
output_filename = 'tile_{}-{}.png'

import math
import os
import gdal
from datetime import datetime

#Save subset ROI to given path
def subsetGeoTiff(ds, outFileName, arr_out, start, size, bands ):

    driver = gdal.GetDriverByName("GTiff")
    #set compression
    outdata = driver.Create(outFileName, size[0], size[1], bands, gdal.GDT_UInt16)
    newGeoTransform = list( ds.GetGeoTransform() )
    newGeoTransform[0] = newGeoTransform[0] + start[0]*newGeoTransform[1] + start[1]*newGeoTransform[2]
    newGeoTransform[3] = newGeoTransform[3] + start[0]*newGeoTransform[4] + start[1]*newGeoTransform[5]

    outdata.SetGeoTransform( newGeoTransform )
    outdata.SetProjection(ds.GetProjection())

    for i in range(0,bands) :
        outdata.GetRasterBand(i+1).WriteArray(arr_out[i, :, :])
        outdata.GetRasterBand(i+1).SetNoDataValue(0)

    outdata.FlushCache()
    outdata = None

startTime = datetime.now()
print("Start" , datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))

Path = "D:/Git/Scripts/T37SEB_20200825.tif"
outDir, file_extension = os.path.splitext(Path)
filename = os.path.split(outDir)[1]
#Create a folder in the same name as the image
if not os.path.exists(filename):
    os.makedirs(filename)

#Open dataset and get contents as a numpy array
ds = gdal.Open(Path)
image = ds.ReadAsArray()

imageWidth = ds.RasterXSize
imageHeight = ds.RasterYSize

tileSizeX = 256
tileSizeY = 256

offsetX = int(tileSizeX/2)
offsetY = int(tileSizeY/2)

tileSize = (tileSizeY, tileSizeX)


for startX in range(0, imageWidth, offsetX):
    for startY in range(0, imageHeight, offsetY):
        endX = startX + tileSizeX
        endY = startY + tileSizeY

        currentTile = image[:, startX:endX,startY:endY]
        #if you want to save save directly with opencv
        # However reverse order of data
        #cv2.imwrite(filename + '_%d_%d' % (nTileY, nTileX)  + file_extension, currentTile)
        start = (startY,startX)
        outFullFileName = os.path.join( outDir, filename + '_%d_%d' % (startY, startX)  + file_extension )
        subsetGeoTiff(ds, outFullFileName, currentTile, start, tileSize,  currentTile.shape[0] )


endTime = datetime.now()
ds=None

print("Finished " , datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), " in  " , endTime-startTime )