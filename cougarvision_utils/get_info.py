
import argparse
import warnings
import yaml

from animl.classification import load_classifier
from animl.detection import load_detector 

class ConfigInfo:
    """ This class is used to define elements from the config file"""
    #class atributes
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
    EMAIL_ALERTS = bool(CONFIG['email_alerts'])
    ER_ALERTS = bool(CONFIG['er_alerts'])
    LOG_DIR = CONFIG['log_dir']
    CHECKPOINT_F= CONFIG['checkpoint_frequency']
    CONFIDENCE = CONFIG['confidence']
    TARGETS = CONFIG['alert_targets']
    CONSUMER_EMAILS= CONFIG['consumer_emails']
    TOKEN = CONFIG['token']
    #for fetch image api
    SAVE_DIR = CONFIG['save_dir'] 
    CAMERA_NAMES = dict(CONFIG['camera_names'])
    BASE = CONFIG['strikeforce_api']
    ACCOUNTS = CONFIG['username_scraper']
    AUTH_TOKEN = CONFIG['auth_token']
    PATH = "./last_id.txt"
    PASSWORD_SCRAPER = CONFIG['password_scraper']

    def _init_(self):
        #do nothing 
        print("")
