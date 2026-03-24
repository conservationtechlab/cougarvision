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
from cougarvision_utils.get_info import ConfigInfo

#class ConfigInfo:
 #   """ This class is used to define elements from the config file"""
    #class atributes
    # Numpy FutureWarnings from tensorflow import
#    warnings.filterwarnings('ignore', category=FutureWarning)
    # Parse arguments
#    PARSER = argparse.ArgumentParser(description='Retrieves images from \
#                                 email & web scraper & runs detection')
#    PARSER.add_argument('config', type=str, help='Path to config file')
#    ARGS = PARSER.parse_args()
#    CONFIG_FILE = ARGS.config
#    with open(CONFIG_FILE, 'r', encoding='utf-8') as stream:
#        CONFIG = yaml.safe_load(stream)
        
#    USERNAME = CONFIG['username']
#    PASSWORD = CONFIG['password']
#    TOKEN = CONFIG['token']
#    AUTH = CONFIG['authorization']
#    CLASSIFIER = CONFIG['classifier_model']
#   DETECTOR = CONFIG['detector_model']
#    DEV_EMAILS = CONFIG['dev_emails']
#    HOST = 'imap.gmail.com'
#    CLASSES = CONFIG['classes']
#    MODEL_TYPE = CONFIG['detector_model_type']
#    CHECKIN_INTERVAL = CONFIG['checkin_interval']
#    INTERVAL = CONFIG['run_scheduler']
#    CLASSIFIER_MODEL, CLASS_LIST = load_classifier(CLASSIFIER, CLASSES)
#    DETECTOR_MODEL = load_detector(DETECTOR, MODEL_TYPE)
    #for detect
#    EMAIL_ALERTS = bool(CONFIG['email_alerts'])
#    ER_ALERTS = bool(CONFIG['er_alerts'])
#    LOG_DIR = CONFIG['log_dir']
#    CHECKPOINT_F= CONFIG['checkpoint_frequency']
#    CONFIDENCE = CONFIG['confidence']
#    TARGETS = CONFIG['alert_targets']
#    CONSUMER_EMAILS= CONFIG['consumer_emails']
#    TOKEN = CONFIG['token']
    #for fetch image api
#    SAVE_DIR = CONFIG['save_dir'] 
#    CAMERA_NAMES = dict(CONFIG['camera_names'])
#    BASE = CONFIG['strikeforce_api']
#    ACCOUNTS = CONFIG['username_scraper']
#    AUTH_TOKEN = CONFIG['auth_token']
#    PATH = "./last_id.txt"
#    PASSWORD_SCRAPER = CONFIG['password_scraper']

#    def _init_(self):
        #do nothing 
#        print("")


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

#adding args to main instead of class
def parse_args():
    # Numpy FutureWarnings from tensorflow import
    warnings.filterwarnings('ignore', category=FutureWarning) 
    PARSER = argparse.ArgumentParser(description='Retrieves images from \
                                 email & web scraper & runs detection')
    PARSER.add_argument('config', type=str, help='Path to config file')
    
    #return args container
    return PARSER.parse_args()


   
def main():
    ''''Runs main program and schedules future runs'''
    logger()
    args = parse_args()
    config = ConfigInfo(args)
    #pass arguement config from args container
    fetch_detect_alert(args.config)
    
    #lambda keeps fetch and detect callable
    schedule.every(config.INTERVAL).minutes.do(lambda: fetch_detect_alert(config))
    schedule.every(config.CHECKIN_INTERVAL).hours.do(checkin, config.DEV_EMAILS,
                                              config.USERNAME, config.PASSWORD, config.HOST)
    schedule.every(30).days.do(post_monthly_obs, config.TOKEN, config.AUTH)

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()
