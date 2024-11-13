import os
import glob
import re
from osgeo import gdal, osr


def parse_folder_name(folder_name: str) -> int:
    match = re.search(r"\d+", folder_name)
    if match:
        return int(match.group())
    else:
        raise ValueError("Folder name does not contain a numeric value.")


def parse_image_name(image_path: str) -> int:
    image_name = os.path.basename(image_path)
    match = re.search(r"\d+", image_name)
    if match:
        return int(match.group())
    else:
        raise ValueError("Image name does not contain a numeric value.")


def merge_images2(input_dir: str, output_file: str):
    folders = sorted(os.listdir(input_dir), key=parse_folder_name)

    # Initialize min_y and max_y
    min_y, max_y = float('inf'), float('-inf')

    for folder in folders:
        image_files = sorted(glob.glob(os.path.join(input_dir, folder, "*")), key=parse_image_name)
        min_y = min(min_y, parse_image_name(image_files[0]))
        max_y = max(max_y, parse_image_name(image_files[-1]) + 1)

    first_folder = os.path.join(input_dir, folders[0])
    first_image = gdal.Open(glob.glob(os.path.join(first_folder, "*"))[0])
    width, height = first_image.RasterXSize, first_image.RasterYSize

    min_x = parse_folder_name(folders[0])
    max_x = parse_folder_name(folders[-1]) + 1

    merged_image_width = (max_x - min_x) * width
    merged_image_height = (max_y - min_y) * height

    driver = gdal.GetDriverByName('GTiff')
    merged_image = driver.Create(output_file, merged_image_width, merged_image_height, 3, gdal.GDT_Byte)

    background_color = [255, 255, 255]  # RGB values for white

    for band in range(1, 4):
        merged_image.GetRasterBand(band).Fill(background_color[band - 1])

    merged_tile_count = 0

    for x, folder in enumerate(folders):
        image_files = sorted(glob.glob(os.path.join(input_dir, folder, "*")), key=parse_image_name)
        for y, image_file in enumerate(image_files):
            image_name = os.path.join(folder, os.path.basename(image_file))

            print(f"Merging image: {image_file}")
            image = gdal.Open(image_file)
            folder_num = parse_folder_name(folder)
            image_num = parse_image_name(image_name)
            xpos = (folder_num - min_x) * width
            ypos = (image_num - min_y) * height

            print(f"Image '{image_name}' is pasted at position ({xpos}, {ypos})")

            data = image.ReadAsArray()
            for band in range(data.shape[0]):
                merged_image.GetRasterBand(band + 1).WriteArray(data[band], xpos, ypos)

            merged_tile_count += 1

    merged_image.FlushCache()
    print(f"Number of tiles merged: {merged_tile_count}")


def main():
    input_dir2 = r"\\192.168.1.115\智元数据交换\个人文件夹\hsh\亚洲\17"
    output_file2 = "rgb.tif"
    if os.path.exists(output_file2):
        os.remove(output_file2)

    merge_images2(input_dir2, output_file2)


if __name__ == "__main__":
    main()
