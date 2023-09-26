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
import yaml
import schedule
from cougarvision_utils.detect_img import detect
from cougarvision_utils.alert import checkin
from cougarvision_utils.get_images import fetch_image_api
from animl.predictSpecies import load_classifier
from animl.detectMD import load_MD_model


# Numpy FutureWarnings from tensorflow import
warnings.filterwarnings('ignore', category=FutureWarning)
# Parse arguments
PARSER = argparse.ArgumentParser(description='Retrieves images from \
                                 email & web scraper & runs detection')
PARSER.add_argument('config', type=str, help='Path to config file')
ARGS = PARSER.parse_args()
CONFIG_FILE = ARGS.config
# Load Configuration Settings from YML file
with open(CONFIG_FILE, 'r', encoding='utf-8') as stream:
    CONFIG = yaml.safe_load(stream)
# Set Email Variables for fetching
USERNAME = CONFIG['username']
PASSWORD = CONFIG['password']
TO_EMAILS = CONFIG['to_emails']
CLASSIFIER = CONFIG['classifier_model']
DETECTOR = CONFIG['detector_model']
DEV_EMAILS = CONFIG['dev_emails']
HOST = 'imap.gmail.com'


# Set interval for checking in
CHECKIN_INTERVAL = CONFIG['checkin_interval']

# load models once
CLASSIFIER_MODEL = load_classifier(CLASSIFIER)
DETECTOR_MODEL = load_MD_model(DETECTOR)


def fetch_detect_alert():
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


def main():
    ''''Runs main program and schedules future runs'''
    fetch_detect_alert()
    schedule.every(10).minutes.do(fetch_detect_alert)
    schedule.every(CHECKIN_INTERVAL).hours.do(checkin, DEV_EMAILS,
                                              USERNAME, PASSWORD, HOST)

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()
