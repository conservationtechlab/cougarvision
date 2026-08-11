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
import time
import warnings
from datetime import datetime as dt
import logging
import schedule

from sageranger.post_monthly import post_monthly_obs
from sageranger.unpack_info import get_config_info
from cougarvision_utils.detect_img import detect
from cougarvision_utils.alert import checkin
from cougarvision_utils.get_images import fetch_image_api
from cougarvision_utils.get_info import ConfigInfo


def logger():
    """Function for creating log file"""
    logging.basicConfig(filename='cougarvision.log', level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')


def fetch_detect_alert(config):
    """Function for fetching images, detection, and sending alerts"""
    # Run the scheduler
    logging.info("Starting cougarvision.")
    print("Running fetch_and_alert")
    print("Fetching images")
    images = fetch_image_api(config)
    print('Finished fetching images')
    print('Starting Detection')
    detect(images, config)
    print('Finished Detection')
    print("Sleeping since: " + str(dt.now()))


def main():
    """Runs main program and schedules future runs"""

    # Numpy FutureWarnings from tensorflow import
    warnings.filterwarnings('ignore', category=FutureWarning)

    logger()
    config = get_config_info(ConfigInfo)

    # pass ConfigInfo dataclass object
    fetch_detect_alert(config)

    # lambda keeps fetch and detect callable
    if config.visualize_output is True:
        schedule.every(config.run_scheduler
                       ).seconds.do(lambda:
                                    fetch_detect_alert(config))
    else:
        schedule.every(3
                       ).minutes.do(lambda:
                                    fetch_detect_alert(config))

    schedule.every(config.checkin_interval
                   ).minutes.do(lambda:
                              checkin(config))
    
    if config.post_monthly:
        schedule.every(30).days.do(lambda: post_monthly_obs(
                                   config.authorization,
                                   config.camera_names))
                           

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()
