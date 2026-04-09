"""Fetch and Alert

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
"""

# Import local utilities
import argparse
import time
import warnings
from datetime import datetime as dt
import logging
from dataclasses import fields
import schedule
import sys
import yaml

from sageranger.post_monthly import post_monthly_obs
from cougarvision_utils.detect_img import detect
from cougarvision_utils.alert import checkin
from cougarvision_utils.get_images import fetch_image_api
from cougarvision_utils.get_info import ConfigInfo


def logger():
    """Function for creating log file"""
    logging.basicConfig(filename='cougarvision.log', level=logging.INFO)


def fetch_detect_alert(config):
    """Function for fetching images, detection, and sending alerts"""
    # Run the scheduler
    print("Running fetch_and_alert")
    print("Fetching images")
    images = fetch_image_api(config)
    print('Finished fetching images')
    print('Starting Detection')
    detect(images, config)
    print('Finished Detection')
    print("Sleeping since: " + str(dt.now()))


def parse_args():
    """Creates parser for config yaml.

    This function creates an arguement parser that creates an
    args container with the arguement 'CONFIG'.

    Returns:
        argsparse.Namespace: An object containing all parsed arguement
            values as attributes (e.g., args.CONFIG).
    """
    parser = argparse.ArgumentParser(description='Retrieves images from \
                                    email & web scraper & runs detection')
    parser.add_argument('CONFIG', type=str, help='Path to config file')

    return parser.parse_args()


def main():
    '''Runs main program and schedules future runs'''
    
    # Numpy FutureWarnings from tensorflow import
    warnings.filterwarnings('ignore', category=FutureWarning) 
    
    logger()
    args = parse_args()
    config_path = args.CONFIG
    
    with open(config_path, 'r', encoding='utf-8') as file:
        config_dict = yaml.safe_load(file)

    # for direct mapping only use fields in ConfigInfo
    valid_keys = {f.name for f in fields(ConfigInfo)}
    filtered_keys = {k: v for k, v in config_dict.items() if k in valid_keys}

    config = ConfigInfo(**filtered_keys)

    # pass ConfigInfo dataclass object
    fetch_detect_alert(config)

    # lambda keeps fetch and detect callable
    if config.visualize_output is True:
        schedule.every(config.run_scheduler).seconds.do(lambda:
                                                        fetch_detect_alert(config))
    else:
        schedule.every(config.run_scheduler).minutes.do(lambda:
                                                    fetch_detect_alert(config))
        
    schedule.every(config.checkin_interval).hours.do(
                                                     checkin,
                                                     config.dev_emails,
                                                     config.username,
                                                     config.password,
                                                     config.host
                                                     )
    schedule.every(30).days.do(post_monthly_obs,
                               config.token, config.authorization)

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()
