'''
ConfigInfo is a class that holds attributes from the configuration
file (fetch_and_cougar.yml). It expects a configuration file path
to be provided when initalizing a configInfo object.
'''
import yaml

from animl.classification import load_classifier
from animl.detection import load_detector


class ConfigInfo:
    """ This class is used to define elements from the config file"""
    def __init__(self, config_path: str):

        with open(config_path, 'r', encoding='utf-8') as stream:
            self.config = yaml.safe_load(stream)

        self.username = self.config['username']
        self.password = self.config['password']
        self.token = self.config['token']
        self.auth = self.config['authorization']
        self.classifier = self.config['classifier_model']
        self.detector = self.config['detector_model']
        self.dev_emails = self.config['dev_emails']
        self.host = self.config['host']
        self.classes = self.config['classes']
        self.model_type = self.config['detector_model_type']
        self.checkin_interval = self.config['checkin_interval']
        self.interval = self.config['run_scheduler']
        self.classifier_model, self.class_list = load_classifier(
                                                self.classifier,
                                                self.classes)
        self.detector_model = load_detector(self.detector, self.model_type)
        # for detect
        self.email_alerts = bool(self.config['email_alerts'])
        self.er_alerts = bool(self.config['er_alerts'])
        self.log_dir = self.config['log_dir']
        self.checkpoint_f = self.config['checkpoint_frequency']
        self.confidence = self.config['confidence']
        self.targets = self.config['alert_targets']
        self.consumer_emails = self.config['consumer_emails']
        # for fetch image api
        self.save_dir = self.config['save_dir']
        self.camera_names = dict(self.config['camera_names'])
        self.base = self.config['strikeforce_api']
        self.accounts = self.config['username_scraper']
        self.auth_token = self.config['auth_token']
        self.id_path = self.config['id_path']
        self.password_scraper = self.config['password_scraper']
        self.traps_path = self.config['camera_traps_path']
