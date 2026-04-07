""" Get Info

Get_info holds the ConfigInfo data class that holds attribute values
from the configuration file. It expects a configuration file path to be
provided when using the class method. Fetch_and_alert, get_images,
and detect_img rely on these attribute values.
"""

from dataclasses import dataclass, field
from typing import Any
import yaml

from animl.classification import load_classifier
from animl.detection import load_detector


@dataclass
class ConfigInfo:
    """Defines values from the config yaml.

    Dataclasses automatically create an _init_ function but
    here we use a class method and post_init function to separate
    parsing the yaml and loading the models.

    Example:
        To create a ConfigInfo dataclass object use the from_yaml
        function.

             config = ConfigInfo.from_yaml(config_path)

    """
    # assign fields
    username: str
    password: str
    token: str
    auth: str
    classifier: str
    detector: str
    dev_emails: list
    host: str
    classes: list
    model_type: str
    checkin_interval: int
    interval: int
    email_alerts: bool
    er_alerts: bool
    log_dir: str
    checkpoint_f: int
    confidence: float
    targets: list
    consumer_emails: list
    save_dir: str
    camera_names: dict
    base: str
    accounts: str
    auth_token: str
    id_path: str
    password_scraper: str

    # runtime fields not apart of inital constructor
    classifier_model: Any = field(init=False)
    class_list: list = field(init=False)
    detector_model: object = field(init=False)

    def __post_init__(self):
        """Loads classifer and detector models.

        This method handles the extra logic of the derived field
        values after the default init function.
        """

        self.classifier_model, self.class_list = load_classifier(
            self.classifier, self.classes
        )
        self.detector_model = load_detector(
            self.detector, self.model_type
        )

    @classmethod
    def from_yaml(cls, config_path: str) -> "ConfigInfo":
        """Loads and parses config file.

        Args:
            config_path (str): The path to the configuration
                file.
        Returns:
             ConfigInfo: updated instance of itself.
        """

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return cls(
            username=config['username'],
            password=config['password'],
            token=config['token'],
            auth=config['authorization'],
            classifier=config['classifier_model'],
            detector=config['detector_model'],
            dev_emails=config['dev_emails'],
            host=config['host'],
            classes=config['classes'],
            model_type=config['detector_model_type'],
            checkin_interval=config['checkin_interval'],
            interval=config['run_scheduler'],
            # used in detect
            email_alerts=bool(config['email_alerts']),
            er_alerts=bool(config['er_alerts']),
            log_dir=config['log_dir'],
            checkpoint_f=config['checkpoint_frequency'],
            confidence=config['confidence'],
            targets=config['alert_targets'],
            consumer_emails=config['consumer_emails'],
            # used in fetch image api
            save_dir=config['save_dir'],
            camera_names=dict(config['camera_names']),
            base=config['strikeforce_api'],
            accounts=config['username_scraper'],
            auth_token=config['auth_token'],
            id_path=config['id_path'],
            password_scraper=config['password_scraper'],

        )
