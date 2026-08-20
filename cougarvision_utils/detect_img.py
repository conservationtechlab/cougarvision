"""Detect Img

This script defines the function responsible for classifying
images based on a trained classifier and sending alerts to either
email or Earthranger as specified by the fetch_and_alert.yml config file.

The defined function depends on local modules cropping.py, alert.py,
post_event_er.py, and attach_image_er.py as well as some functions
that must be imported from animl.
"""

from io import BytesIO
from datetime import datetime as dt
import os
import logging
from PIL import Image
import animl
from sageranger import is_target, attach_image, post_event

from cougarvision_utils.cropping import draw_bounding_box_on_image
from cougarvision_utils.alert import send_alert


def detect(images, config):  # pylint: disable=too-many-locals
    """The function detects alert_targets.

    This function takes in a dataframe of images and runs a detector model,
    classifies the species of interest defined in the config yaml, and sends
    alerts either to email or an interface called Earthranger.

    Args:
        images(array): a nested array of information regarding each photo that
          is to be run through the detector and is formatted
          ['strikeforce id']['thumbnail url']['local file path']
        config (ConfigInfo): the unpacked config values from
            fetch_and_alert.yml that contains necessary parameters
            the function needs.
    """
    # pylint: disable=too-many-nested-blocks
    # pylint: disable=too-many-statements

    if len(images) > 0:
        # extract paths from dataframe
        image_paths = images[:, 2]
        # detection.detect expects the image paths in a list
        image_path_list = image_paths.tolist()
        # Run Detection
        # confidendce and checkpoint frequency
        conf = config.confidence
        ch_f = config.checkpoint_frequency
        results = animl.detect(config.detector_model_load,
                               image_path_list,
                               resize_width=1280,
                               resize_height=1280,
                               confidence_threshold=conf,
                               checkpoint_frequency=ch_f,
                               batch_size=4)
        # Parse results
        data_frame = animl.parse_detections(results)
        # single classification function checks for the file
        # extension so we add it
        data_frame["extension"] = data_frame["filepath"].str.extract(
                                            r'(\.[^.]+)$',
                                            expand=False).str.lower()
        # filter out all non animal detections
        if not data_frame.empty:
            animal_df = animl.get_animals(data_frame)

            # run classifier on animal detections if there are any
            if not animal_df.empty:
                classifer_model = config.classifier_model_load
                predictions_raw = animl.classify(classifer_model,
                                                 animal_df,
                                                 batch_size=4)
                # single classification expects a list
                class_list_series = config.class_list["species"].tolist()
                preds = animl.single_classification(animal_df,
                                                    None,
                                                    predictions_raw,
                                                    class_list_series)

                cougars = preds[preds['prediction'].isin(config.alert_targets)]
                # drops all detections with confidence less than threshold
                cougars = cougars[cougars['confidence'] >= config.confidence]
                # reset dataframe index
                cougars = cougars.reset_index(drop=True)
                # create a row in datafra,e containing only the camera name
                # regex expression works for most names except names starting
                # with a number
                cougars["cam_name"
                        ] = cougars["filepath"
                                    ].str.extract(r'([A-Za-z][A-Za-z\d\s\W]*)')
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

                    labeled_img = config.path_to_labeled_output
                    if config.visualize_output is True:
                        os.makedirs(labeled_img, exist_ok=True)
                        # get image names from orginal image list
                        # created and returned in get_images
                        name = images[idx][2]
                        name = name.split('/')[-1]
                        new_file_name = labeled_img + "/" + name

                        with open(new_file_name, "wb") as folder:
                            folder.write(img_byte)

                    cam_name = cougars.at[idx, 'cam_name']
                    er_alerts = config.er_alerts
                    if er_alerts is True:
                        # post an observation
                        try:
                            is_target(cam_name,
                                      config.authorization,
                                      label)
                            logging.info("Posted observation to"
                                         "earthranger.")
                        except IndexError as e:
                            logging.warning("IndexError: %s", str(e))
                            print(f"Index error: {e}")

                        # post an event
                        try:
                            event_id = post_event(label,
                                                  cam_name,
                                                  config.authorization)
                            response = attach_image(event_id,
                                                    img_byte,
                                                    config.authorization,
                                                    label)
                            logging.info("Posted event on earthranger with "
                                         "associated img.")
                            print(response)
                        except IndexError as e:
                            logging.warning("Index Error: %s", str(e))
                            print(f"Index error: {e}")

                    if config.email_alerts is True:
                        dev = 0
                        send_alert(config, label, image_bytes,
                                   dev, prob)
                        dev = 1
                        send_alert(config, label, image_bytes,
                                   dev, prob)

                # Write Dataframe to csv
                current_date = dt.now()
                formatted_dt = current_date.strftime("%m-%d-%Y_%H:%M:%S")
                logs = config.log_dir
                os.makedirs(logs, exist_ok=True)
                cougars.to_csv(f'{logs}dataframe_{formatted_dt}')
