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
from PIL import Image
from animl import classification, split
from animl import detection
from sageranger import is_target, attach_image, post_event

from cougarvision_utils.cropping import draw_bounding_box_on_image
from cougarvision_utils.alert import smtp_setup, send_alert


def detect(images, config):  # pylint: disable=too-many-locals

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

    # add path to camera traps repository instead using
    # cougar traps yaml
    sys.path.append(config.traps_path)

    if len(images) > 0:
        # extract paths from dataframe
        image_paths = images[:, 2]
        # detection.detect expects the image paths in a list
        image_path_list = image_paths.tolist()
        # Run Detection
        results = detection.detect(config.detector_model,
                                   image_path_list,
                                   resize_width=1280,
                                   resize_height=1280,
                                   confidence_threshold=config.confidence,
                                   checkpoint_frequency=config.checkpoint_f,
                                   batch_size=4
                                   )
        # Parse results
        data_frame = detection.parse_detections(results)
        # single classification function checks for the file
        # extension so we add it
        data_frame["extension"] = data_frame["filepath"].str.extract(
                                            r'(\.[^.]+)$',
                                            expand=False).str.lower()
        # filter out all non animal detections
        if not data_frame.empty:
            animal_df = split.get_animals(data_frame)
            # other_df = split.get_empty(data_frame)
            # run classifier on animal detections if there are any
            if not animal_df.empty:
                predictions_raw = classification.classify(config.
                                                          classifier_model,
                                                          animal_df,
                                                          batch_size=4
                                                          )
                # single classification expects a list
                class_list_series = config.class_list["species"].tolist()
                preds = classification.single_classification(animal_df,
                                                             None,
                                                             predictions_raw,
                                                             class_list_series
                                                             )
                cougars = preds[preds['prediction'].isin(config.targets)]
                # drops all detections with confidence less than threshold
                cougars = cougars[cougars['confidence'] >= config.confidence]
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
                    if label in config.targets and config.er_alerts is True:
                        is_target(cam_name, config.token, config.auth, label)
                    # Email or Earthranger alerts as dictated in the config yml
                    if config.er_alerts is True:
                        event_id = post_event(label,
                                              cam_name,
                                              config.token,
                                              config.auth)
                        response = attach_image(event_id,
                                                img_byte,
                                                config.token,
                                                config.auth,
                                                label)
                        print(response)
                    if config.email_alerts is True:
                        smtp_server = smtp_setup(config.username,
                                                 config.password,
                                                 config.host
                                                 )
                        dev = 0
                        send_alert(label, image_bytes, smtp_server,
                                   config.username, config.consumer_emails,
                                   dev, prob
                                   )
                        dev = 1
                        send_alert(label, image_bytes, smtp_server,
                                   config.username, config.dev_emails,
                                   dev, prob)

                # Write Dataframe to csv
                current_date = dt.now()
                formatted_dt = current_date.strftime("%m-%d-%Y_%H:%M:%S")
                cougars.to_csv(f'{config.log_dir}dataframe_{formatted_dt}')
