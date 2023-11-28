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
import time
import re
import sys
import logging
import yaml
from PIL import Image
from animl import parse_results, classify, split
from sageranger import is_target, attach_image, post_event
from animl.detectMD import detect_MD_batch

from cougarvision_utils.cropping import draw_bounding_box_on_image
from cougarvision_utils.alert import smtp_setup, send_alert
from cougarvision_visualize.visualize_helper import get_last_file_number
from cougarvision_visualize.visualize_helper import create_folder


with open("config/cameratraps.yml", 'r') as stream:
    camera_traps_config = yaml.safe_load(stream)
    sys.path.append(camera_traps_config['camera_traps_path'])


def detect(images, config, c_model, d_model):
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
    # use_variation = int(config['use_variation'])
    email_alerts = bool(config['email_alerts'])
    er_alerts = bool(config['er_alerts'])
    log_dir = config['log_dir']
    checkpoint_f = config['checkpoint_frequency']
    confidence = config['confidence']
    classes = config['classes']
    targets = config['alert_targets']
    username = config['username']
    password = config['password']
    consumer_emails = config['consumer_emails']
    dev_emails = config['dev_emails']
    host = 'imap.gmail.com'
    token = config['token']
    authorization = config['authorization']
    color = config['color']
    visualize_output = config['visualize_output']
    labeled_img = config['path_to_labeled_output']
    if len(images) > 0:
        # extract paths from dataframe
        image_paths = images[2]
        # Run Detection
        start = time.time()
        results = detect_MD_batch(d_model,
                                  image_paths,
                                  checkpoint_path=None,
                                  confidence_threshold=confidence,
                                  checkpoint_frequency=checkpoint_f,
                                  results=None,
                                  quiet=False,
                                  image_size=None)
        end = time.time()
        md_time = end - start
        logging.debug('Time to detect: ' + str(md_time))
        # Parse results
        data_frame = parse_results.from_MD(results, None, None)
        # filter out all non animal detections
        if not data_frame.empty:
            animal_df = split.getAnimals(data_frame)
            otherdf = split.getEmpty(data_frame)
            # run classifier on animal detections if there are any
            if not animal_df.empty:
                # create generator for images
                start = time.time()
                predictions = classify.predict_species(animal_df, c_model,
                                                       batch=4)
                end = time.time()
                cls_time = end - start
                logging.debug('Time to classify: ' + str(cls_time))
                # Parse results
                max_df = parse_results.from_classifier(animal_df,
                                                       predictions,
                                                       classes,
                                                       None,
                                                       False)
                # Creates a data frame with all relevant data
                cougars = max_df[max_df['prediction'].isin(targets)]
                # drops all detections with confidence less than threshold
                cougars = cougars[cougars['conf'] >= confidence]
                # reset dataframe index
                cougars = cougars.reset_index(drop=True)
                # create a row in the dataframe containing only the camera name
                # flake8: disable-next
                cougars['cam_name'] = cougars['file'].apply(lambda x: re.findall(r'[A-Z]\d+', x)[0])  # noqa: E501  # pylint: disable-msg=line-too-long
                # Sends alert for each cougar detection
                for idx in range(len(cougars.index)):
                    label = cougars.at[idx, 'prediction']
                    # uncomment this line to use conf value for dev email alert
                    prob = str(cougars.at[idx, 'conf'])
                    label = cougars.at[idx, 'class']
                    img = Image.open(cougars.at[idx, 'file'])
                    draw_bounding_box_on_image(img,
                                               cougars.at[idx, 'bbox2'],
                                               cougars.at[idx, 'bbox1'],
                                               cougars.at[idx,
                                                          'bbox2'] +
                                               cougars.at[idx,
                                                          'bbox4'],
                                               cougars.at[idx,
                                                          'bbox1'] +
                                               cougars.at[idx,
                                                          'bbox3'],
                                               color,
                                               expansion=0,
                                               use_normalized_coordinates=True)
                    image_bytes = BytesIO()
                    img.save(image_bytes, format="JPEG")
                    img_byte = image_bytes.getvalue()
                    if visualize_output is True:
                        folder_path = create_folder(labeled_img)
                        last_file_number = get_last_file_number(folder_path)
                        new_file_number = last_file_number + 1
                        new_file_name = f"{folder_path}/image_{new_file_number}.jpg"

                        with open(new_file_name, "wb") as folder:
                            folder.write(img_byte)

                    cam_name = cougars.at[idx, 'cam_name']
                    if label in targets and er_alerts is True:
                        is_target(cam_name, token, authorization, label)
                    # Email or Earthranger alerts as dictated in the config yml
                    logging.info('Sending detection to Earthranger')
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
                        logging.info(response)

                    logging.info('Sending detection email')
                    if email_alerts is True:
                        smtp_server = smtp_setup(username, password, host)
                        dev = 0
                        send_alert(label, image_bytes, smtp_server,
                                   username, consumer_emails, dev, prob)
                        dev = 1
                        send_alert(label, image_bytes, smtp_server,
                                   username, dev_emails, dev, prob)
                    logging.info('Finished sending detection email')
                # Write Dataframe to csv
                date = "%m-%d-%Y_%H:%M:%S"
                cougars.to_csv(f'{log_dir}dataframe_{dt.now().strftime(date)}')
