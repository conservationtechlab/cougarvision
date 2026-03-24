
import argparse
import yaml

from animl.classification import load_classifier
from animl.detection import load_detector 

class ConfigInfo:
    """ This class is used to define elements from the config file"""
    #class atributes

    def __init__(self, config_path: str):

        with open(config_path, 'r', encoding='utf-8') as stream:
            self.CONFIG = yaml.safe_load(stream)
        
        self.USERNAME = self.CONFIG['username']
        self.PASSWORD = self.CONFIG['password']
        self.TOKEN = self.CONFIG['token']
        self.AUTH = self.CONFIG['authorization']
        self.CLASSIFIER = self.CONFIG['classifier_model']
        self.DETECTOR = self.CONFIG['detector_model']
        self.DEV_EMAILS = self.CONFIG['dev_emails']
        self.HOST = self.CONFIG['host']
        self.CLASSES = self.CONFIG['classes']
        self.MODEL_TYPE = self.CONFIG['detector_model_type']
        self.CHECKIN_INTERVAL = self.CONFIG['checkin_interval']
        self.INTERVAL = self.CONFIG['run_scheduler']
        self.CLASSIFIER_MODEL, self.CLASS_LIST = load_classifier(self.CLASSIFIER, self.CLASSES)
        self.DETECTOR_MODEL = load_detector(self.DETECTOR, self.MODEL_TYPE)
        #for detect
        self.EMAIL_ALERTS = bool(self.CONFIG['email_alerts'])
        self.ER_ALERTS = bool(self.CONFIG['er_alerts'])
        self.LOG_DIR = self.CONFIG['log_dir']
        self.CHECKPOINT_F= self.CONFIG['checkpoint_frequency']
        self.CONFIDENCE = self.CONFIG['confidence']
        self.TARGETS = self.CONFIG['alert_targets']
        self.CONSUMER_EMAILS= self.CONFIG['consumer_emails']
        self.TOKEN = self.CONFIG['token']
        #for fetch image api
        self.SAVE_DIR = self.CONFIG['save_dir'] 
        self.CAMERA_NAMES = dict(self.CONFIG['camera_names'])
        self.BASE = self.CONFIG['strikeforce_api']
        self.ACCOUNTS = self.CONFIG['username_scraper']
        self.AUTH_TOKEN = self.CONFIG['auth_token']
        self.PATH = self.CONFIG['path']
        self.PASSWORD_SCRAPER = self.CONFIG['password_scraper']





