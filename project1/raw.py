#resize all images to 512x512
import PIL
import os
import os.path
from PIL import Image

"""f = r'C:/Users/DELL G3/PycharmProjects/project1/dataset/png/val_labels'
for file in os.listdir(f):
    f_img = f+"/"+file
    img = Image.open(f_img)
    img = img.resize((512,512))
    img.save(f_img, subsampling=0,  quality = 100)
"""

"""import glob
import os

from image_fragment.fragment import ImageFragment

# FOR .jpg, .png, .jpeg
from imageio import imread, imsave

# FOR .tiff
from tifffile import imread, imsave

ORIGINAL_DIM_OF_IMAGE = (1500, 1500, 3)
CROP_TO_DIM = (384, 384, 3)

image_fragment = ImageFragment.image_fragment_3d(
    fragment_size=(384, 384, 3), org_size=ORIGINAL_DIM_OF_IMAGE
)

IMAGE_DIR = r"C:/Users/DELL G3/PycharmProjects/project1/test_dataset/test"
SAVE_DIR = r"C:/Users/DELL G3/PycharmProjects/project1/test_dataset/test_modified"

for file in glob.glob(
    os.path.join(IMAGE_DIR, "*")
):
    image = imread(file)
    for i, fragment in enumerate(image_fragment):
        # GET DATA THAT BELONGS TO THE FRAGMENT
        fragmented_image = fragment.get_fragment_data(image)

        imsave(
            os.path.join(
                SAVE_DIR,
                f"{i}_{os.path.basename(file)}",
            ),
            fragmented_image,
        )"""
import tensorflow as tf

print(f"Tensorflow version: {tf.__version__}")
print(f"Keras Version: {tf.keras.__version__}")
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")


#im = Image.open('/home/nick/PycharmProjects/project1/taizhou_dataset/玉环市_卫图3/玉环市_卫图3_Level_18.tif')


"""from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np

dataset = gdal.Open(r'/home/nick/PycharmProjects/project1/taizhou_dataset/玉环市_卫图3/玉环市_卫图3_Level_18.tif')

band1 = dataset.GetRasterBand(1) # Red channel
band2 = dataset.GetRasterBand(2) # Green channel
band3 = dataset.GetRasterBand(3) # Blue channel
b1 = band1.ReadAsArray()
b2 = band2.ReadAsArray()
b3 = band3.ReadAsArray()
img = np.dstack((b1, b2, b3))
f = plt.figure()
plt.imshow(img)
plt.savefig('Tiff.png')
plt.show()
"""