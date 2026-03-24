'''Fetch and Alert

This script allows users to retrieve thumbnail images uploaded
from cellular camera traps, classify them by species, and send
the photo as an alert to a specified end point.

This script and its modules depend on some but not all scripts
in this module including, cougarvision_utils/alert.py, cropping.py,
detect_img.py, get_images.py, ImageCropGenerator.py, and
earthranger_utils/attach_image_er.py, and post_event_er.py, and the
config/fetch_and_alert.yml and last_id.txt.

One must configure their individual fetch_and_alert.yml file to fit
their needs by ensuring the file paths, usernames and passwords, camera
dictionary, and image classifiers are correct. The .yml file is also
where one can choose whether they would like email alerts or to send
the classified images to Earthranger.
'''

# Import local utilities
import argparse
import time
import warnings
from datetime import datetime as dt
import logging
import schedule

from sageranger.post_monthly import post_monthly_obs
from cougarvision_utils.detect_img import detect
from cougarvision_utils.alert import checkin
from cougarvision_utils.get_images import fetch_image_api
from cougarvision_utils.get_info import ConfigInfo


def logger():
    '''Function for creating log file'''
    logging.basicConfig(filename='cougarvision.log', level=logging.INFO)


def fetch_detect_alert(config):
    '''Functions for fetching images, detection, and sending alerts'''
    # Run the scheduler
    print("Running fetch_and_alert")
    print("Fetching images")
    images = fetch_image_api(config)
    print('Finished fetching images')
    print('Starting Detection')
    # actually dont need to pass these arguements just need config?
    detect(images, config, config.CLASSIFIER_MODEL,
           config.DETECTOR_MODEL, config.CLASS_LIST)
    print('Finished Detection')
    print("Sleeping since: " + str(dt.now()))


def parse_args():
    ''' This function creates an arguement parser that creates an
    args container with the arguement 'CONFIG'. It returns the
    container to access config file path use args.CONFIG after
    calling parse_args()
    '''

    parser = argparse.ArgumentParser(description='Retrieves images from \
                                    email & web scraper & runs detection')
    parser.add_argument('CONFIG', type=str, help='Path to config file')

    # return args container
    return parser.parse_args()


def main():
    ''''Runs main program and schedules future runs'''

    # Numpy FutureWarnings from tensorflow import
    warnings.filterwarnings('ignore', category=FutureWarning)

    logger()
    args = parse_args()
    config_path = args.CONFIG
    config = ConfigInfo(config_path)

    # pass arguement config from args container
    fetch_detect_alert(config)

    # lambda keeps fetch and detect callable
    schedule.every(config.INTERVAL).minutes.do(lambda:
                                               fetch_detect_alert(config))
    schedule.every(config.CHECKIN_INTERVAL).hours.do(
                                                     checkin,
                                                     config.DEV_EMAILS,
                                                     config.USERNAME,
                                                     config.PASSWORD,
                                                     config.HOST
                                                     )
    schedule.every(30).days.do(post_monthly_obs,
                               config.TOKEN, config.AUTH)

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()
