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
from PIL import Image
from tensorflow import keras
from animl import FileManagement, ImageCropGenerator, DetectMD
from sageranger import is_target, attach_image, post_event

from cougarvision_utils.cropping import draw_bounding_box_on_image
from cougarvision_utils.alert import smtp_setup, send_alert


def detect(images, config):  # pylint: disable-msg=too-many-locals
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
    detector_model = config['detector_model']
    email_alerts = bool(config['email_alerts'])
    er_alerts = bool(config['er_alerts'])
    classifier_model = config['classifier_model']
    model = keras.models.load_model(classifier_model)
    log_dir = config['log_dir']
    checkpoint_frequency = config['checkpoint_frequency']
    confidence_threshold = config['confidence']
    classes = config['classes']
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
        # Run Detection
        results = DetectMD.load_and_run_detector_batch(image_paths,
                                                       detector_model,
                                                       log_dir,
                                                       confidence_threshold,
                                                       checkpoint_frequency,
                                                       [])
        # Parse results
        data_frame = FileManagement.parseMD(results)
        # filter out all non animal detections
        if not data_frame.empty:
            animal_df, _ = FileManagement.filterImages(data_frame)
            # run classifier on animal detections if there are any
            if not animal_df.empty:
                # create generator for images

                generator = ImageCropGenerator.\
                    GenerateCropsFromFile(animal_df)
                # Run Classifier
                predictions = model.predict_generator(generator,
                                                      steps=len(generator),
                                                      verbose=1)
                # Parse results
                max_df = FileManagement.parseCM(animal_df, None,
                                                predictions, classes)
                # Creates a data frame with all relevant data
                cougars = max_df[max_df['class'].isin(targets)]
                # drops all detections with confidence less than threshold
                cougars = cougars[cougars['conf'] >= confidence_threshold]
                # reset dataframe index
                cougars = cougars.reset_index(drop=True)
                # create a row in the dataframe containing only the camera name
                # flake8: disable-next
                cougars['cam_name'] = cougars['file'].apply(lambda x: re.findall(r'[A-Z]\d+', x)[0])  # noqa: E501  # pylint: disable-msg=line-too-long
                # Sends alert for each cougar detection
                for idx in range(len(cougars.index)):
                    label = cougars.at[idx, 'class']
                    prob = str(cougars.at[idx, 'conf'])
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
