'''Detect Img

This script defines the function responsible for classifying
images based on a trained classifier and sending alerts to either
email or Earthranger as specified by the fetch_and_alert.yml config file.

The defined function depends on local modules cropping.py, alert.py,
post_event_er.py, and attach_image_er.py as well as some functions
that must be imported from animl.
'''

from io import BytesIO
from datetime import datetime as dt
import re
import sys
import yaml
from PIL import Image
from animl import classification, split
from sageranger import is_target, attach_image, post_event
from animl import detection

from cougarvision_utils.cropping import draw_bounding_box_on_image
from cougarvision_utils.alert import smtp_setup, send_alert


with open("config/cameratraps.yml", 'r') as stream:
    CAM_CONFIG = yaml.safe_load(stream)
    sys.path.append(CAM_CONFIG['camera_traps_path'])


def detect(images, config, c_model, d_model, class_list):
    '''
    This function takes in a dataframe of images and runs a detector model,
    classifies the species of interest, and sends alerts either to email or an
    interface called Earthranger

    Args:
    images: a nested array of information regarding each photo that is to be
        run through the detector and is formatted
        ['strikeforce id']['thumbnail url']['local file path']
    config: the unpacked config values from fetch_and_alert.yml that contains
        necessary parameters the function needs
    '''
    email_alerts = bool(config['email_alerts'])
    er_alerts = bool(config['er_alerts'])
    log_dir = config['log_dir']
    checkpoint_f = config['checkpoint_frequency']
    confidence = config['confidence']
    targets = config['alert_targets']
    username = config['username']
    password = config['password']
    consumer_emails = config['consumer_emails']
    dev_emails = config['dev_emails']
    host = 'imap.gmail.com'
    token = config['token']
    authorization = config['authorization']

    if len(images) > 0:
        # extract paths from dataframe
        image_paths = images[:, 2]
        # detection.detect expects the image paths in a list
        image_path_list = image_paths.tolist()
        # Run Detection
        results = detection.detect(d_model,
                                   image_path_list,
                                   resize_width=1280,
                                   resize_height=1280,
                                   confidence_threshold=confidence,
                                   checkpoint_frequency=checkpoint_f,
                                   batch_size=4
                                   )
        # Parse results
        data_frame = detection.parse_detections(results)
        # single classification function checks for the file extension so we add it
        data_frame["extension"] = data_frame["filepath"].str.extract(r'(\.[^.]+)$', expand=False).str.lower()
        # filter out all non animal detections
        if not data_frame.empty:
            animal_df = split.get_animals(data_frame)
            other_df = split.get_empty(data_frame)
            # run classifier on animal detections if there are any
            if not animal_df.empty:
                predictions_raw = classification.classify(c_model,
                                                          animal_df,
                                                          batch_size=4)
                # single classification expects a list
                class_list_for_series = class_list["species"].tolist()
                preds = classification.single_classification(animal_df,
                                                             None,
                                                             predictions_raw,
                                                             class_list_for_series
                                                             )
                cougars = preds[preds['prediction'].isin(targets)]
                # drops all detections with confidence less than threshold
                cougars = cougars[cougars['confidence'] >= confidence]
                # reset dataframe index
                cougars = cougars.reset_index(drop=True)
                # create a row in the dataframe containing only the camera name
                # flake8: disable-next
                cougars['cam_name'] = cougars['filepath'].apply(lambda x: re.findall(r'[A-Z]\d+', x)[0])  # noqa: E501  # pylint: disable-msg=line-too-long
                # Sends alert for each cougar detection
                for idx in range(len(cougars.index)):
                    label = cougars.at[idx, 'prediction']
                    # uncomment this line to use conf value for dev email alert
                    prob = str(cougars.at[idx, 'confidence'])
                    img = Image.open(cougars.at[idx, 'filepath'])
                    draw_bounding_box_on_image(img,
                                               cougars.at[idx, 'bbox_y'],
                                               cougars.at[idx, 'bbox_x'],
                                               cougars.at[idx,
                                                          'bbox_y'] +
                                               cougars.at[idx,
                                                          'bbox_h'],
                                               cougars.at[idx,
                                                          'bbox_x'] +
                                               cougars.at[idx,
                                                          'bbox_w'],
                                               expansion=0,
                                               use_normalized_coordinates=True)
                    image_bytes = BytesIO()
                    img.save(image_bytes, format="JPEG")
                    img_byte = image_bytes.getvalue()
                    cam_name = cougars.at[idx, 'cam_name']
                    if label in targets and er_alerts is True:
                        is_target(cam_name, token, authorization, label)
                    # Email or Earthranger alerts as dictated in the config yml
                    if er_alerts is True:
                        event_id = post_event(label,
                                              cam_name,
                                              token,
                                              authorization)
                        response = attach_image(event_id,
                                                img_byte,
                                                token,
                                                authorization,
                                                label)
                        print(response)
                    if email_alerts is True:
                        smtp_server = smtp_setup(username, password, host)
                        dev = 0
                        send_alert(label, image_bytes, smtp_server,
                                   username, consumer_emails, dev, prob)
                        dev = 1
                        send_alert(label, image_bytes, smtp_server,
                                   username, dev_emails, dev, prob)
                # Write Dataframe to csv
                date = "%m-%d-%Y_%H:%M:%S"
                cougars.to_csv(f'{log_dir}dataframe_{dt.now().strftime(date)}')
