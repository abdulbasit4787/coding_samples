import numpy as np
import os
import glob
import csv
import pandas as pd
import ast
import cv2 as cv
import matplotlib.pyplot as plt
import cv2
import time
import matplotlib.cm as cm


left_images_dir = "C:/xampp/htdocs/myserver/rectification/left/"
right_images_dir = "C:/xampp/htdocs/myserver/rectification/right/"
SIFT_path = "C:/xampp/htdocs/myserver/SIFT Keypoints/"
keypoint_path = "C:/xampp/htdocs/myserver/Keypoint matches/"
epilines_path = "C:/xampp/htdocs/myserver/epilines/"
left_rectified_path = "C:/xampp/htdocs/myserver/rectified_images/left/"
right_rectified_path = "C:/xampp/htdocs/myserver/rectified_images/right/"
rectified_images_path = "C:/xampp/htdocs/myserver/rectified_images/rectified_plots/"
disparity_path = "C:/xampp/htdocs/myserver/disparity_images/"
depth_path = "C:/xampp/htdocs/myserver/depth_images/"
camera_image_path = "C:/xampp/htdocs/myserver/camera_image/"
two_point_distance_path = "C:/xampp/htdocs/myserver/two_point_distance/"


def draw_lines(img1src, img2src, lines, pts1src, pts2src):
    """
    Draws lines and circles on two images based on given line parameters and corresponding points.

    Args:
        img1src (numpy.ndarray): Source image 1 (grayscale).
        img2src (numpy.ndarray): Source image 2 (grayscale).
        lines (numpy.ndarray): Array of line parameters (rho, theta).
        pts1src (numpy.ndarray): Array of points in source image 1.
        pts2src (numpy.ndarray): Array of points in source image 2.

    Returns:
        img1color (numpy.ndarray): Image 1 with lines and circles (BGR color).
        img2color (numpy.ndarray): Image 2 with circles (BGR color).
        :rtype: object
        :param img1src:
        :param img2src:
        :param lines:
        :param pts1src:
        :param pts2src:
        :return:
    """
    # Get the shape of image 1
    r, c = img1src.shape

    # Convert grayscale images to BGR color images
    img1color = cv.cvtColor(img1src, cv.COLOR_GRAY2BGR)
    img2color = cv.cvtColor(img2src, cv.COLOR_GRAY2BGR)

    # Edit: use the same random seed so that two images are comparable!
    np.random.seed(0)

    # Draw lines and circles on the images
    for r, pt1, pt2 in zip(lines, pts1src, pts2src):
        # Generate a random color
        color = tuple(np.random.randint(0, 255, 3).tolist())

        # Calculate the line endpoints
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])

        # Draw the line and circles on image 1
        img1color = cv.line(img1color, (x0, y0), (x1, y1), color, 1)
        img1color = cv.circle(img1color, tuple(pt1), 5, color, -1)

        # Draw circles on image 2
        img2color = cv.circle(img2color, tuple(pt2), 5, color, -1)

    return img1color, img2color


def load_camera_parameters(filename):
    """
    Load camera parameters from a CSV file.

    Args:
        filename (str): The name of the CSV file.

    Returns:
        parameters (dict): A dictionary containing camera parameters.
        :rtype: object
        :param filename:
        :return:
    """
    with open(filename, 'r') as file:
        # Read the CSV file
        reader = csv.reader(file)

        # Get the headers and the corresponding values from the first row
        headers = next(reader)
        values = next(reader)

        # Create a dictionary to store the camera parameters
        parameters = {}

        # Extract and store each parameter with its corresponding header
        for header, value in zip(headers, values):
            # Convert the parameter value from string to a numpy array
            parameters[header] = np.array(eval(value))

    return parameters


def main(fx, fy, baseline_m, pixel_size_m):
    """
        Perform calibration, rectification, depth estimation, and calculate distances.

        Args:
            fx (float): Focal length along the x-axis.
            fy (float): Focal length along the y-axis.
            baseline_m (float): Baseline distance between the cameras in meters.
            pixel_size_m (float): Size of a pixel in meters.

        Distances Calculated:
            - Camera-Object Distance: Estimate the distance from the camera to objects in the scene using the depth image.
            - 2-Point Distance: Measure the distance between two points of interest on the rectified left image in the real world.
        :param fx:
        :param fy:
        :param baseline_m:
        :param pixel_size_m:
        Returns:
            None

    """
    # calibration and rectification

    # Get the lists of left and right image files
    left_image_files = sorted(os.listdir(left_images_dir))
    right_image_files = sorted(os.listdir(right_images_dir))

    # Check if the number of left and right images is equal
    if len(left_image_files) != len(right_image_files):
        raise Exception("The number of left and right images should be equal.")

    for i in range(len(left_image_files)):
        # Load the left and right images
        left_image_path = os.path.join(left_images_dir, left_image_files[i])
        right_image_path = os.path.join(right_images_dir, right_image_files[i])

        left_image = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
        right_image = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)

        # Perform SIFT feature detection and extraction
        sift = cv.SIFT_create()

        kp1, des1 = sift.detectAndCompute(left_image, None)
        kp2, des2 = sift.detectAndCompute(right_image, None)

        # Visualize SIFT keypoints
        imgSift = cv.drawKeypoints(left_image, kp1, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv.imwrite(SIFT_path + f"SIFT Keypoints{i}.png", imgSift)

        # Perform feature matching using FLANN
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=80)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # Filter the matches based on the Lowe's ratio test
        matchesMask = [[0, 0] for i in range(len(matches))]
        good = []
        pts1 = []
        pts2 = []

        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                matchesMask[i] = [1, 0]
                good.append(m)
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)

        # Visualize keypoint matches
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask[200:500],
                           flags=cv.DrawMatchesFlags_DEFAULT)

        keypoint_matches = cv.drawMatchesKnn(
            left_image, kp1, right_image, kp2, matches[200:500], None, **draw_params)
        cv2.imwrite(keypoint_path + f"Keypoint matches{i}.png", keypoint_matches)

        # Perform fundamental matrix estimation using RANSAC
        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        fundamental_matrix, inliers = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC)

        # Filter the inlier points
        pts1 = pts1[inliers.ravel() == 1]
        pts2 = pts2[inliers.ravel() == 1]

        # Compute the epilines and draw them on the images
        lines1 = cv.computeCorrespondEpilines(
            pts2.reshape(-1, 1, 2), 2, fundamental_matrix)
        lines1 = lines1.reshape(-1, 3)
        img5, img6 = draw_lines(left_image, right_image, lines1, pts1, pts2)

        lines2 = cv.computeCorrespondEpilines(
            pts1.reshape(-1, 1, 2), 1, fundamental_matrix)
        lines2 = lines2.reshape(-1, 3)
        img3, img4 = draw_lines(right_image, left_image, lines2, pts2, pts1)

        # Visualize the epilines
        plt.subplot(121), plt.imshow(img5)
        plt.subplot(122), plt.imshow(img3)
        plt.suptitle("Epilines in both images")
        plt.savefig(epilines_path + f"epilines{i}.png", dpi=800)
        # plt.show()
        plt.close()
        plt.clf()

        # Rectify the images
        h1, w1 = left_image.shape
        h2, w2 = right_image.shape
        _, H1, H2 = cv.stereoRectifyUncalibrated(np.float32(pts1), np.float32(pts2), fundamental_matrix,
                                                 imgSize=(w1, h1))

        # Save the rectified images
        img1_rectified = cv.warpPerspective(left_image, H1, (w1, h1))
        img2_rectified = cv.warpPerspective(right_image, H2, (w2, h2))

        left_rect_img = left_rectified_path + f"rectified_{i}.png"
        right_rect_img = right_rectified_path + f"rectified_{i}.png"

        cv2.imwrite(left_rect_img, img1_rectified)
        cv2.imwrite(right_rect_img, img2_rectified)

        # Visualize the rectified images
        fig, axes = plt.subplots(1, 2, figsize=(15, 10))
        axes[0].imshow(img1_rectified, cmap="gray")
        axes[1].imshow(img2_rectified, cmap="gray")
        axes[0].axhline(250)
        axes[1].axhline(250)
        axes[0].axhline(450)
        axes[1].axhline(450)
        plt.suptitle(f"Rectified images {i}")
        plt.savefig(rectified_images_path + f"rectified_images_{i}.png", dpi=800)
        plt.close()
        plt.clf()

        # Disparity map estimation
        rectified_left = img1_rectified  # cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
        rectified_right = img2_rectified  # cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)

        stereo = cv.StereoSGBM_create(
            minDisparity=-128,
            numDisparities=128 + 128,
            blockSize=8,
            # Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
            uniquenessRatio=5,
            speckleWindowSize=200,
            # Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
            speckleRange=6,  # To get rid of these artifacts and tiny dortion in real images, adjust
            disp12MaxDiff=0,
            P1=8 * 1 * 11 * 11,
            P2=32 * 1 * 11 * 11,
        )
        disparity_SGBM = stereo.compute(rectified_left, rectified_right)
        disparity_SGBM = cv.normalize(disparity_SGBM, disparity_SGBM, alpha=255, beta=0, norm_type=cv.NORM_MINMAX)
        disparity_SGBM = np.uint8(disparity_SGBM)

        plt.imshow(disparity_SGBM)
        plt.colorbar()
        disparity_image_path = disparity_path + f"disparity_{i}.png"
        plt.imsave(disparity_image_path, disparity_SGBM, dpi=800)
        plt.close()

        # Depth map estimation
        focal_length_m = ((fx + fy) / 2) * pixel_size_m
        disparity_map_saved = cv2.imread(disparity_image_path, cv2.IMREAD_GRAYSCALE)

        depth_map = np.zeros_like(disparity_map_saved, dtype=np.float32)
        valid_pixels = disparity_map_saved > 0
        depth_map[valid_pixels] = (baseline_m * focal_length_m) / (disparity_map_saved[valid_pixels] * pixel_size_m)
        depth_map = cv2.normalize(depth_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_image_path = depth_path + f"depth_{i}.png"
        cv2.imwrite(depth_image_path, depth_map)

        # Camera-object distance estimation
        depth_map2 = depth_map  # cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)

        # Convert the depth map to real-world distances
        min_depth_m = -0.5
        max_depth_m = 10
        real_depth_map = min_depth_m + (depth_map2 / 255.0) * (max_depth_m - min_depth_m)

        # Image center
        height, width = real_depth_map.shape
        y, x = height // 2, width // 2

        # Get the real-world distance of the point at the center of the image
        real_distance = real_depth_map[y, x]

        real_distance = round(real_distance, 2)

        # color_depth_map = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2RGB)

        cv2.circle(depth_map2, (x, y), radius=10, color=(255, 0, 0), thickness=-1)
        plt.imshow(depth_map2)
        plt.text(x, y - 10, f"{real_distance} m", color='red', fontsize=12, ha='center')
        plt.savefig(camera_image_path + f"depth/image_{i}.png", dpi=800)

        #plt.show()
        #time.sleep(5)
        #plt.close()
        plt.clf()

        # rectified combined
        left_rectified_image = cv2.imread(left_rect_img, cv2.IMREAD_GRAYSCALE)
        right_rectified_image = cv2.imread(right_rect_img, cv2.IMREAD_GRAYSCALE)

        if left_rectified_image.shape != right_rectified_image.shape:
            print(f"Images not the same size for iteration {i}.")
            continue
        overlap_image = cv2.addWeighted(left_rectified_image, 0.5, right_rectified_image, 0.5, 0)
        overlap_rect_rgb = cv2.cvtColor(overlap_image, cv2.COLOR_GRAY2RGB)
        height, width = overlap_image.shape
        y, x = height // 2, width // 2
        real_distance_overlap = real_depth_map[y, x]
        real_distance_overlap = round(real_distance_overlap, 2)
        cv2.putText(overlap_rect_rgb, f"{real_distance_overlap} m", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 255), 2)
        cv2.imwrite(camera_image_path + f"rectified_combined/rectified_combined_{i}.png", overlap_rect_rgb)
        #cv2.imshow("Image", overlap_rect_rgb)
        #cv2.waitKey(5000)
        cv2.destroyAllWindows()

        #rgb combined

        left_image_rgb = cv2.imread(left_image_path)
        right_image_rgb = cv2.imread(right_image_path)

        if left_image_rgb is None:
            print(f"Failed to load {left_image_path}")
            continue
        if right_image_rgb is None:
            print(f"Failed to load {right_image_path}")
            continue

        h, w = left_image_rgb.shape[:2]
        y, x = h // 2, w // 2

        alpha = 0.5
        blend_image_rgb = cv2.addWeighted(left_image_rgb, alpha, right_image_rgb, 1 - alpha, 0)
        real_distance = real_depth_map[y, x]

        real_distance = round(real_distance, 2)

        cv2.circle(blend_image_rgb, (x, y), radius=10, color=(255, 0, 0), thickness=-1)

        #blend_image_rgb = cv2.cvtColor(blend_image_rgb, cv2.COLOR_BGR2RGB)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        distance_text = f"{real_distance} m"
        text_size, _ = cv2.getTextSize(distance_text, font, font_scale, font_thickness)
        text_position = (x - text_size[0] // 2, y - 10)

        # Draw the text on the image
        cv2.putText(blend_image_rgb, distance_text, text_position, font, font_scale, (0, 0, 255), font_thickness,
                    cv2.LINE_AA)

        # Save the annotated image
        cv2.imwrite(camera_image_path + f"rgb_combined/rgb_combined_{i}.png", blend_image_rgb)

        # Display the image
        cv2.imshow("RGB Image", blend_image_rgb)
        cv2.waitKey(5000)  # Display for 5 seconds
        cv2.destroyAllWindows()

        # 2 point distance

        depth_map_img = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)
        rgb_left_image = cv2.imread(left_image_path)

        image_height, image_width, _ = rgb_left_image.shape

        cx1 = cx2 = image_width / 2
        cy1 = cy2 = image_height / 2
        fx1 = fx2 = fx
        fy1 = fy2 = fy

        points = []

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                if len(points) == 2:
                    points.append((x, y))
                    Z1 = depth_map_img[points[0][1], points[0][0]]
                    Z2 = depth_map_img[points[1][1], points[1][0]]
                    X1 = (points[0][0] - cx1) * Z1 / fx1
                    Y1 = (points[0][1] - cy1) * Z1 / fy1
                    X2 = (points[1][0] - cx2) * Z2 / fx2
                    Y2 = (points[1][1] - cy2) * Z2 / fy2
                    distance = (np.sqrt((X2 - X1) ** 2 + (Y2 - Y1) ** 2 + (Z2 - Z1) ** 2)) / 100
                    print(f"Real world distance between the points is: {distance} meters")

                    cv2.circle(rgb_left_image, points[0], 5, (0, 0, 255), -1)
                    cv2.circle(rgb_left_image, points[1], 5, (0, 0, 255), -1)

                    cv2.line(rgb_left_image, points[0], points[1], (0, 0, 255), 2)

                    text_position = ((points[0][0] + points[1][0]) // 2, (points[0][1] + points[1][1]) // 2)
                    cv2.putText(rgb_left_image, f"{distance:.2f} meters", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 255), 2)

                    cv2.imwrite(two_point_distance_path + f"final_image{i}.png", rgb_left_image)

                    cv2.imshow("Image", rgb_left_image)
                    cv2.waitKey(5000)
                    cv2.destroyAllWindows()

        cv2.imshow('image', rgb_left_image)
        cv2.setMouseCallback('image', click_event)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


while True:
    # Check if there are image files in the directory
    image_files = os.listdir(left_images_dir)
    if len(image_files) == 0:
        print("No images found in the directory. Waiting...")
        time.sleep(5)
        continue

    # Execute the main code if there are image files
    main(fx=565.5734695282406, fy=566.6751428944614, baseline_m=0.238, pixel_size_m=(1.12 / 1000000))
    time.sleep(5)