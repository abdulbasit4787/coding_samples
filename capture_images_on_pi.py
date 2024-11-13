import numpy as np
import os
import cv2
import time


def capture_images(camera1, camera2):
    """
    Capture images from two cameras and save them to designated directories.

    Args:
        camera1 (int): The camera ID for the first camera.
        camera2 (int): The camera ID for the second camera.
        :param camera1: camera1 ID
        :param camera2: camera2 ID
    Returns:
            None

    """
    # Define the server path and create directories for left and right images
    server_path = r"C:\xampp\htdocs\myserver\rectification"
    os.makedirs(os.path.join(server_path, "left"), exist_ok=True)
    os.makedirs(os.path.join(server_path, "right"), exist_ok=True)
    path_left = os.path.join(server_path, "left")
    path_right = os.path.join(server_path, "right")

    # Open video captures for the two cameras
    cap1 = cv2.VideoCapture(camera1)
    cap2 = cv2.VideoCapture(camera2)

    # Check if the cameras were successfully opened
    if not (cap1.isOpened() and cap2.isOpened()):
        print("Failed to open cameras.")
        return

    # Disable autofocus for both cameras
    cap1.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap2.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    img_counter = 0
    while img_counter < 50:
        # Read frames from both cameras
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not (ret1 and ret2):
            print("Failed to receive frames from the cameras. Exiting ...")
            break

        # Increase brightness of frames if needed
        brightness_increase1 = 0
        brightness_increase2 = 0
        frame1 = cv2.add(frame1, np.ones(frame1.shape, dtype=np.uint8) * brightness_increase1)
        frame2 = cv2.add(frame2, np.ones(frame2.shape, dtype=np.uint8) * brightness_increase2)

        # Generate a unique timestamp for the current image
        current_time = time.strftime("%Y%m%d-%H%M%S")

        # Save frames as images in the respective left and right directories
        cv2.imwrite(os.path.join(path_left, f"left_{current_time}_{img_counter}.png"), frame1)
        cv2.imwrite(os.path.join(path_right, f"right_{current_time}_{img_counter}.png"), frame2)
        img_counter += 1

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) == ord('q'):
            break

    print("Image capture completed successfully.")

    # Release the video captures and close any remaining windows
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()


# Continuously capture images from the cameras every 5 seconds
while True:
    capture_images(1, 2)
    time.sleep(5)
