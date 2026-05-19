"""CougarVision Visualize Output

Usage:
    python3 -m cougarvision_visualize.display config/<config_file_name>

This script is intended to be run alongside fetch_and_alert.py in a second
terminal or tmux terminal. It displays the most recent classified detections
on a 3x3 grid on one screen and most recent images on a second 9x9 grid

    - must run script from /cougarvision not /cougarvision_visuzalize
    - cougarvision conda environment must be activated
    - fetch_and_alert.py must be run with visualize_output: param set to 'True'
    - path_to_unlabeled_output: and path_to_labeled_output: parameters filled
      out in the config file.
    - the command line argument to run is python3 display.py
      </full/path/to/yaml/>
    - Set the correct amount of monitors in the config file under display_num
    *fetch_and_alert.py will create the folders for you
    if you only include the paths but the folders are not yet created.

This script will assume the amount monitors based on the display_num value
found in the config file. If two monitors are defined but one found the
system will exit. If one monitor is defined but two monitors are found it
will use the second monitor to display the 9 most recent images otherwise
it will use the one monitor available. If there are two monitors it will use
the first to display the 9 most recent images and the second to display the 81
most recent images. If the folders are empty the system will display blank
screens until images are added to the folder.
"""


import os
import time
import sys
import re
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2
from screeninfo import get_monitors
from cougarvision_utils.get_info import DisplayInfo
from fetch_and_alert import get_config_info


def get_screen_resolutions():
    """Function to get the screen resolutions for both monitors.

    Returns:
        list: list of tuples that represent the width and height of
            each monitor.
    """
    monitors = get_monitors()
    resolutions = [(monitor.width,
                    monitor.height,
                    monitor.x,
                    monitor.y) for monitor in monitors]
    return resolutions


def get_recent_images(f_p, num_images):
    """Function to return the newest x num of images from folder.

    Args:
        f_P (string): Path to folder of images.
        num_images (int): desired amount of images from folder.

    Returns:
        list: list of valid images from the image folder path.
    """
    # files in directory f_p and checks file paths
    fil = [f.name for f in Path(f_p).iterdir() if f.is_file()]

    if not fil:
        return []

    # logic only needed locally sorts images but timestamp
    def sort_key_func(file_name):
        try:
            match = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}",
                              file_name)
            if match:
                return datetime.strptime(match.group(), "%Y-%m-%d %H:%M:%S")

        except ValueError:
            return float('-inf')

        return None

    fil.sort(key=sort_key_func, reverse=True)
    newest_files = fil[:num_images]
    images = [cv2.imread(os.path.join(f_p, file)) for file in newest_files]
    images = [img for img in images if img is not None]
    return images


def display_images(window, images, size, window_name='CougarVision'):
    # pylint: disable=too-many-locals
    """Function to display x amount of recent images in size x size grid.

    Args:
        window (tuple): tuple of monitor height and width.
        images (list): List of images from labeled images folder.
        size (int): Desired size of grid.
        window_name ('obj':'str', optional): Title of the window

    """
    screen_width, screen_height, _, _ = window
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


def setup_windows(resolutions, num_screen):
    """Defines windows and places them on correct monitors.

    Args:
        resolutions (list): List of tuples that has width
            and height of monoitors.
        num_screen (int): Number of expected screens given by
            the yaml file.
    Returns:
        tuple: tuple of (str, str, bool) where the str values
            represent window titles and bool represents if
            there is a second monitor,
    """
    window_1 = 'CougarVision'
    window_2 = 'Newest Image'

    second_monitor = len(resolutions) > 1
    one_monitor = False

    if num_screen == 2 and second_monitor is False:
        return None  # Error case

    if num_screen == 1:
        cv2.namedWindow(window_1, cv2.WINDOW_NORMAL)
        if second_monitor:
            cv2.moveWindow(window_1, resolutions[1][2], resolutions[1][3])
            second_monitor = False
            one_monitor = True
            # represents content being shown on second monitor
        else:
            cv2.moveWindow(window_1, resolutions[0][2], resolutions[0][3])

        cv2.setWindowProperty(window_1, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
    else:
        # define first window on first monitor
        cv2.namedWindow(window_1, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_1, resolutions[0][2], resolutions[0][3])

        cv2.namedWindow(window_2, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_2, resolutions[1][2], resolutions[1][3])

        # full screen
        cv2.setWindowProperty(window_1, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
        cv2.setWindowProperty(window_2, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)

    return window_1, window_2, second_monitor, one_monitor


def main_display():
    """Runs main program."""

    config = get_config_info(DisplayInfo)
    resolutions = get_screen_resolutions()

    try:
        window_1, window_2, second_monitor, one_monitor = setup_windows(
                                                           resolutions,
                                                           config.display_num,
                                                           )
    except TypeError:
        print("Defined 2 screens in configuration file but found only\n"
              "1 actual screen. Change value in yaml or "
              "connect another screen.")
        sys.exit()

    while True:
        labeled_img = get_recent_images(config.path_to_labeled_output, 9)
        unlabeled_img = get_recent_images(config.save_dir, 81)

        if config.default_screen:
            if not one_monitor:
                display_images(resolutions[0], labeled_img, 3, window_1)
            else:
                display_images(resolutions[1], labeled_img, 3, window_1)

            if second_monitor:
                display_images(resolutions[1], unlabeled_img, 9, window_2)
        else:
            if not one_monitor:
                display_images(resolutions[0], unlabeled_img, 9, window_1)
            else:
                display_images(resolutions[1], unlabeled_img, 9, window_1)

            if second_monitor:
                display_images(resolutions[1], labeled_img, 3, window_2)

        time.sleep(1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main_display()
