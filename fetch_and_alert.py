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

from sageranger.post_monthly import post_monthly_obs
from animl.classification import load_classifier
from animl.detection import load_detector 
from cougarvision_utils.detect_img import detect
from cougarvision_utils.alert import checkin
from cougarvision_utils.get_images import fetch_image_api

class ConfigInfo:
    """ This class is used to define elements from the config file"""
    #class atributes
    # def _init_(self):
    # Numpy FutureWarnings from tensorflow import
    warnings.filterwarnings('ignore', category=FutureWarning)
    # Parse arguments
    PARSER = argparse.ArgumentParser(description='Retrieves images from \
                                 email & web scraper & runs detection')
    PARSER.add_argument('config', type=str, help='Path to config file')
    ARGS = PARSER.parse_args()
    CONFIG_FILE = ARGS.config
    with open(CONFIG_FILE, 'r', encoding='utf-8') as stream:
        CONFIG = yaml.safe_load(stream)
        
    USERNAME = CONFIG['username']
    PASSWORD = CONFIG['password']
    TOKEN = CONFIG['token']
    AUTH = CONFIG['authorization']
    CLASSIFIER = CONFIG['classifier_model']
    DETECTOR = CONFIG['detector_model']
    DEV_EMAILS = CONFIG['dev_emails']
    HOST = 'imap.gmail.com'
    CLASSES = CONFIG['classes']
    MODEL_TYPE = CONFIG['detector_model_type']
    CHECKIN_INTERVAL = CONFIG['checkin_interval']
    INTERVAL = CONFIG['run_scheduler']
    CLASSIFIER_MODEL, CLASS_LIST = load_classifier(CLASSIFIER, CLASSES)
    DETECTOR_MODEL = load_detector(DETECTOR, MODEL_TYPE)
    #for detect
    email_alerts = bool(CONFIG['email_alerts'])
    er_alerts = bool(CONFIG['er_alerts'])
    log_dir = CONFIG['log_dir']
    checkpoint_f = CONFIG['checkpoint_frequency']
    confidence = CONFIG['confidence']
    targets = CONFIG['alert_targets']
    consumer_emails = CONFIG['consumer_emails']
    token = CONFIG['token']
    #for fetch image api
    SAVE_DIR = CONFIG['save_dir'] 
    camera_names = dict(CONFIG['camera_names'])
    base = CONFIG['strikeforce_api']
    accounts = CONFIG['username_scraper']
    auth_token = CONFIG['auth_token']
    path = "./last_id.txt"
    password_scraper = CONFIG['password_scraper']

    def _init_(self):
        #do nothing 
        print("")


def logger():
    '''Function for creating log file'''
    logging.basicConfig(filename='cougarvision.log', level=logging.INFO)

#added config agruement
def fetch_detect_alert(config):
    '''Functions for fetching images, detection, and sending alerts'''
    # Run the scheduler
    print("Running fetch_and_alert")
    print("Fetching images")
    #changed CONFIG to config
    images = fetch_image_api(config)
    print('Finished fetching images')
    print('Starting Detection')
    #changed CONFIG to config and added class
    #actually dont need to pass these arguements just need config?
    detect(images, config, config.CLASSIFIER_MODEL, config.DETECTOR_MODEL, config.CLASS_LIST)
    print('Finished Detection')
    print("Sleeping since: " + str(dt.now()))


def main():
    ''''Runs main program and schedules future runs'''
    logger()
    config = ConfigInfo()
    #pass config object
    fetch_detect_alert(config)
    schedule.every(config.INTERVAL).minutes.do(lambda: fetch_detect_alert(config))
    schedule.every(config.CHECKIN_INTERVAL).hours.do(checkin, config.DEV_EMAILS,
                                              config.USERNAME, config.PASSWORD, config.HOST)
    schedule.every(30).days.do(post_monthly_obs, config.TOKEN, config.AUTH)

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()
