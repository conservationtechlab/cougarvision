'''CougarVision Visualize Output
This script is intended to be run alongside fetch_and_alert.py in a second
terminal or tmux terminal. It displays the most recent classified detections
on a 3x3 grid on one screen and most recent images on a second 9x9 grid.
    - cougarvision conda environment must be activated
    - fetch_and_alert.py must be run with visualize_output: param set to 'True'
    - path_to_unlabeled_output: and path_to_labeled_output: parameters filled
      out in the config file.
    *fetch_and_alert.py will create the folders for you
    if you only include the paths but the folders are not yet created.
This script assumes 2 monitors and will display a blank screen on either
display if the minimum number of images is not met, 9 for screen 1 and 81
for screen 2. Later versions will account for this and still display images,
but for now if that is an issue you can fill the folder with black images with
the correct nomenclature: image_1.jpg, image_2.jpg... and it will replace the
black images as they come in.
'''


import os
import time
import argparse
import numpy as np
import yaml
import cv2
from screeninfo import get_monitors


def get_screen_resolutions():
    '''Function to get the screen resolutions for both monitors'''
    monitors = get_monitors()
    resolutions = [(monitor.width, monitor.height) for monitor in monitors]
    return resolutions


def get_newest_images(folder_path, num_images):
    '''Function to return the newest x num of images from folder'''
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    if not files:
        return []

    def sort_key_func(file_name):
        try:
            return int(os.path.splitext(file_name.split('_')[1])[0])
        except ValueError:
            return float('-inf')

    files.sort(key=sort_key_func, reverse=True)
    newest_files = files[:num_images]
    images = [cv2.imread(os.path.join(folder_path, file)) for file in newest_files]
    images = [img for img in images if img is not None]
    return images


def display_images(images, window_name='CougarVision'):
    '''Function to display labeled 9 recent images in 3x3 grid'''
    resolutions = get_screen_resolutions()
    screen_height = resolutions[0][1]
    screen_width = resolutions[0][0]

    num_images_per_row = 3
    num_images_per_col = 3

    max_width_per_image = screen_width // num_images_per_row
    max_height_per_image = screen_height // num_images_per_col

    display_img = np.zeros((screen_height, screen_width, 3), np.uint8)

    for i, img in enumerate(images):
        if img is not None:
            x_offset = (i % num_images_per_row) * max_width_per_image
            y_offset = (i // num_images_per_row) * max_height_per_image

            resized_image = cv2.resize(img, (max_width_per_image, max_height_per_image))

            display_img[y_offset:y_offset + max_height_per_image, x_offset:x_offset + max_width_per_image] = resized_image

    cv2.imshow(window_name, display_img)


def display_more_images(images, window_2='Newest Image'):
    '''Function to display the 81 unlabeled images on 2nd monitor 9x9'''
    resolutions = get_screen_resolutions()
    screen_height = resolutions[1][1]
    screen_width = resolutions[1][0]

    num_images_per_row = 9
    num_images_per_col = 9

    max_width_per_image = screen_width // num_images_per_row
    max_height_per_image = screen_height // num_images_per_col

    display_img = np.zeros((screen_height, screen_width, 3), np.uint8)

    for i, img in enumerate(images):
        if img is not None:
            x_offset = (i % num_images_per_row) * max_width_per_image
            y_offset = (i // num_images_per_row) * max_height_per_image

            resized_image = cv2.resize(img, (max_width_per_image, max_height_per_image))

            display_img[y_offset:y_offset + max_height_per_image, x_offset:x_offset + max_width_per_image] = resized_image

    cv2.imshow(window_2, display_img)


if __name__ == "__main__":
    WINDOW_NAME = 'CougarVision'
    WINDOW_2 = "Newest Image"

    RESOLUTIONS = get_screen_resolutions()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_2, cv2.WINDOW_NORMAL)

    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.moveWindow(WINDOW_2, RESOLUTIONS[0][0], 0)

    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)
    cv2.setWindowProperty(WINDOW_2, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    PARSER = argparse.ArgumentParser(description='Retrieves images from \
                                     email & web scraper & runs detection')
    PARSER.add_argument('config', type=str, help='Path to config file')
    ARGS = PARSER.parse_args()
    CONFIG_FILE = ARGS.config

    with open(CONFIG_FILE, 'r', encoding='utf-8') as stream:
        CONFIG = yaml.safe_load(stream)
    LABELED = CONFIG['path_to_labeled_output']
    UNLABELED = CONFIG['path_to_unlabeled_output']

    while True:
        NEW_IMG = get_newest_images(LABELED, 9)
        if len(NEW_IMG) >= 9:
            display_images(NEW_IMG, WINDOW_NAME)

        NEWER_IMG = get_newest_images(UNLABELED, 81)
        if len(NEWER_IMG) >= 81:
            display_more_images(NEWER_IMG, WINDOW_2)

        time.sleep(1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
