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
import yaml
import schedule

from cougarvision_utils.detect_img import detect
from cougarvision_utils.alert import checkin
from cougarvision_utils.get_images import fetch_image_api
from sageranger.post_monthly import post_monthly_obs

import animl


# Numpy FutureWarnings from tensorflow import
warnings.filterwarnings('ignore', category=FutureWarning)
# Parse arguments


def logger():
    '''Function for creating log file'''
    logging.basicConfig(filename='cougarvision.log', level=logging.INFO)


def fetch_detect_alert(CONFIG, CLASSIFIER_MODEL, DETECTOR_MODEL):
    '''Functions for fetching images, detection, and sending alerts'''
    # Run the scheduler
    print("Running fetch_and_alert")
    print("Fetching images")
    images = fetch_image_api(CONFIG)
    print('Finished fetching images')
    print('Starting Detection')
    detect(images, CONFIG, CLASSIFIER_MODEL, DETECTOR_MODEL)
    print('Finished Detection')
    print("Sleeping since: " + str(dt.now()))


def main(CONFIG):
    ''''Runs main program and schedules future runs'''
    # Set Email Variables for fetching
    logger()
    
    # load models once
    CLASSIFIER_MODEL = animl.load_classifier(CONFIG['classifier_model'])
    DETECTOR_MODEL = animl.load_detector( CONFIG['detector_model'], model_type='mdv5')

    # run fetch at start
    fetch_detect_alert(CONFIG, CLASSIFIER_MODEL, DETECTOR_MODEL)

    # Schedule future fetches
    schedule.every(10).minutes.do(fetch_detect_alert, CONFIG, CLASSIFIER_MODEL, DETECTOR_MODEL)
    schedule.every(CONFIG['checkin_interval']).hours.do(checkin,
                                              CONFIG['dev_emails'],
                                              CONFIG['username'],
                                              CONFIG['password'],
                                              'imap.gmail.com')
    schedule.every(30).days.do(post_monthly_obs, CONFIG['token'], CONFIG['authorization'])

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Retrieves images from \
                                 email & web scraper & runs detection')
    parser.add_argument('config', type=str, help='Path to config file')
    args = parser.parse_args()
    config_file = args.config
    # Load Configuration Settings from YML file
    with open(config_file, 'r', encoding='utf-8') as stream:
        CONFIG = yaml.safe_load(stream)

    main(CONFIG)
