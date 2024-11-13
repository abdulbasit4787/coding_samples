def merge():
    import os
    import glob
    import re
    from osgeo import gdal, osr
    import datetime

    start_time = datetime.datetime.now()

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

    def merge_images(input_dir: str, output_file: str):
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
        merged_image = driver.Create(output_file, merged_image_width, merged_image_height, 1,
                                     gdal.GDT_Byte)  # Update to 1 band

        background_color = 0  # 0 for black background, 255 for white background

        merged_image.GetRasterBand(1).Fill(background_color)

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
                if len(data.shape) == 3:
                    data = data[0]  # Convert 3D array to 2D by selecting the first band
                merged_image.GetRasterBand(1).WriteArray(data, xpos, ypos)  # Write data to the single band

                merged_tile_count += 1

        merged_image.FlushCache()
        print(f"Number of tiles merged: {merged_tile_count}")


    def main():
        input_dir = r"C:\Users\DELL\PycharmProjects\building_project\mask_arranged"
        output_file = r"C:\Users\DELL\PycharmProjects\building_project\hangzhou_mask.tif"
        merge_images(input_dir, output_file)
        end_time = datetime.datetime.now()
        execution_time = end_time-start_time
        print(f"The code executed in {execution_time} minutes")

    if __name__ == "__main__":
        main()
merge()