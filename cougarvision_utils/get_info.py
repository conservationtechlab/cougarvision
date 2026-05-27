""" Get Info

Get_info holds the ConfigInfo data class that holds attribute
values from the configuration file. It also hold the
display_info class that has values from the config directly
related to the display file.Direct mapping is handled in fetch
and alert in get_config_info function.Fields defined in this class
will be the values mapped.The field values must match exactly to
the config file. Fetch_and_alert, get_images, display and detect_img
rely on these attribute values.
"""

from dataclasses import dataclass, field
from typing import Any

from animl.classification import load_classifier
from animl.detection import load_detector


@dataclass
class ConfigInfo:
    """Defines values from the config yaml.

    Dataclasses automatically create an _init_ function but
    here we also use a post_init function to separate
    parsing the yaml and loading the models.

    """
    # assign fields
    # pylint: disable=too-many-instance-attributes
    username: str
    password: str
    authorization: str
    classifier_model: str
    detector_model: str
    dev_emails: list
    host: str
    classes: str
    detector_model_type: str
    checkin_interval: int
    run_scheduler: int
    email_alerts: bool
    er_alerts: bool
    log_dir: str
    checkpoint_frequency: int
    confidence: float
    alert_targets: list
    consumer_emails: list
    save_dir: str
    camera_names: dict
    strikeforce_api: str
    username_scraper: str
    auth_token: str
    id_path: str
    password_scraper: str
    post_monthly: bool
    visualize_output: str
    path_to_unlabeled_output: str
    path_to_labeled_output: str

    # runtime fields not apart of inital constructor
    classifier_model_load: Any = field(init=False)
    class_list: list = field(init=False)
    detector_model_load: object = field(init=False)

    def __post_init__(self):
        """Loads classifer and detector models.

        This method handles the extra logic of the derived field
        values after the default init function.
        """

        self.classifier_model_load, self.class_list = load_classifier(
            self.classifier_model, self.classes
        )
        self.detector_model_load = load_detector(
            self.detector_model, self.detector_model_type
        )


@dataclass
class DisplayInfo:
    """Define values from the config for display."""
    path_to_unlabeled_output: str
    path_to_labeled_output: str
    save_dir: str
    display_num: int
    default_screen: bool
