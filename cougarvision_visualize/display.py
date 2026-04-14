"""CougarVision Visualize Output
This script is intended to be run alongside fetch_and_alert.py in a second
terminal or tmux terminal. It displays the most recent classified detections
on a 3x3 grid on one screen and most recent images on a second 9x9 grid.
    - cougarvision conda environment must be activated
    - fetch_and_alert.py must be run with visualize_output: param set to 'True'
    - path_to_unlabeled_output: and path_to_labeled_output: parameters filled
      out in the config file.
    - the command line argument to run is python3 display.py
      </full/path/to/yaml/>
    *fetch_and_alert.py will create the folders for you
    if you only include the paths but the folders are not yet created.
This script assumes 2 monitors and will display a blank screen on either
display if the minimum number of images is not met, 9 for screen 1 and 81
for screen 2. Later versions will account for this and still display images,
but for now if that is an issue you can fill the folder with black images with
the correct nomenclature: image_1.jpg, image_2.jpg... and it will replace the
black images as they come in.

"""


import os
import time
import argparse
import numpy as np
import yaml
import cv2
from dataclasses import fields
from cougarvision_utils.get_info import display_info
from fetch_and_alert import get_config_info
from screeninfo import get_monitors


def get_screen_resolutions():
    """Function to get the screen resolutions for both monitors.

    Returns:
        list: list of tuples that represent the width and height of
            each monitor.
    """
    monitors = get_monitors()
    resolutions = [(monitor.width, monitor.height) for monitor in monitors]
    return resolutions


def get_recent_images(f_p, num_images):
    """Function to return the newest x num of images from folder.

    Args:
        f_P (string): Path to folder of images.
        num_images (int): desired amount of images from folder.

    Returns:
        list: list of valid images from the image folder path.
    """
    fil = [f for f in os.listdir(f_p) if os.path.isfile(os.path.join(f_p, f))]

    if not fil:
        return []

    # logic only needed locally
    def sort_key_func(file_name):
        try:
            return int(os.path.splitext(file_name.split('_')[1])[0])
        except ValueError:
            return float('-inf')

    fil.sort(key=sort_key_func, reverse=True)
    newest_files = fil[:num_images]
    images = [cv2.imread(os.path.join(f_p, file)) for file in newest_files]
    images = [img for img in images if img is not None]
    return images


def display_images(window, images, size, window_name='CougarVision'):
    """Function to display x amount of recent images in size x size grid.

    Args:
        window (tuple): tuple of monitor height and width.
        images (list): List of images from labeled images folder.
        size (int): Desired size of grid.
        window_name ('obj':'str', optional): Title of the window

    """
    screen_width, screen_height = window
    num_images_row = size
    num_images_col = size

    max_w_image = screen_width // num_images_row
    max_h_image = screen_height // num_images_col

    display_img = np.zeros((screen_height, screen_width, 3), np.uint8)

    for i, img in enumerate(images):
        if img is not None:
            x_offset = (i % num_images_row) * max_w_image
            y_offset = (i // num_images_row) * max_h_image

            resized_image = cv2.resize(img, (max_w_image, max_h_image))

            y_slice = slice(y_offset, y_offset + max_h_image)
            x_slice = slice(x_offset, x_offset + max_w_image)
            display_img[y_slice, x_slice] = resized_image

    cv2.imshow(window_name, display_img)


def setup_windows(resolutions):
    """Defines windows and places them on correct monitors.

    Args:
        resolutions (list): List of tuples that has width
            and height of monoitors.
    Returns:
        tuple: tuple of (str, str, bool) where the str values
            represent window titles and bool represents if 
            there is a second monitor,
    """
    window_1 = 'CougarVision'
    window_2 = "Newest Image"

    second_monitor = len(resolutions) > 1

    # define first window on first monitor
    cv2.namedWindow(window_1, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_1, 0, 0)

    # if second monitor exists
    if second_monitor:
        cv2.namedWindow(window_2, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_2, resolutions[0][0], 0)

    # full screen
    cv2.setWindowProperty(window_1, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    if second_monitor:
        cv2.setWindowProperty(window_2, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)

    return window_1, window_2, second_monitor


def main_display():
    """Runs main program."""

    resolutions = get_screen_resolutions()
    window_1, window_2, second_monitor = setup_windows(resolutions)

    config = get_config_info(display_info)

    while True:
        new_img = get_recent_images(config.path_to_labeled_output, 9)
        if len(new_img) >= 9:
            display_images(resolutions[0], new_img, 3, window_1)
        if second_monitor:
            newer_img = get_recent_images(config.save_dir, 81)
            if len(newer_img) >= 81:
                display_images(resolutions[1], newer_img, 9, window_2)

        time.sleep(1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main_display()
