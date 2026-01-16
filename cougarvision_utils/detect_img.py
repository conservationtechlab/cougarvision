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

import animl
from sageranger import is_target, attach_image, post_event

#from cougarvision_utils.cropping import draw_bounding_box_on_image
from cougarvision_utils.alert import smtp_setup, send_alert


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
    log_dir = config['log_dir']

    classes = animl.load_class_list(config['class_list'])['class']
    targets = config['alert_targets']
    confidence_threshold = config['confidence_threshold']

    if len(images) > 0:
        # Run Detection
        results = animl.detect(d_model,
                               images,
                               1280, 1280,
                               checkpoint_path=None,
                               confidence_threshold=confidence_threshold,
                               checkpoint_frequency=config['checkpoint_frequency'],
                               batch_size=4)
        # Parse results
        data_frame = animl.parse_detections(results)
        # filter out all non animal detections
        if not data_frame.empty:
            animal_df = animl.get_animals(data_frame)
            # run classifier on animal detections if there are any
            if not animal_df.empty:
                # create generator for images
                predictions = animl.classify(c_model, animal_df,
                                             resize_height=299, resize_width=299,
                                             batch_size=4)
                # Parse results
                max_df = animl.single_classification(animal_df, None, predictions, classes,)
                # Creates a data frame with all relevant data
                cougars = max_df[max_df['prediction'].isin(targets)]
                # drops all detections with confidence less than threshold
                cougars = cougars[cougars['conf'] >= confidence_threshold]
                # reset dataframe index
                cougars = cougars.reset_index(drop=True)
                # create a row in the dataframe containing only the camera name
                # flake8: disable-next
                cougars['cam_name'] = cougars['file'].apply(lambda x: re.findall(r'[A-Z]\d+', x)[0])  # noqa: E501  # pylint: disable-msg=line-too-long
                # Sends alert for each cougar detection
                for i, row in cougars.iterrows():
                    # draw bounding box on image
                    #img = draw_bounding_box_on_image(row, expansion=0, use_normalized_coordinates=True)
                    img = animl.plot_box(row, file_col='file', prediction=True)
                    # save image
                    image_bytes = BytesIO()
                    img.save(image_bytes, format="JPEG")
                    img_byte = image_bytes.getvalue()

                    label = row['prediction']
                    prob = str(row['conf'])
                    cam_name = row['cam_name']
                    # connect to earthranger
                    if bool(config['er_alerts']):
                        token = config['token']
                        authorization = config['authorization']

                        is_target(cam_name, token, authorization, label)

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
                    # email alerts
                    if bool(config['email_alerts']):
                        username = config['username']
                        password = config['password']
                        host = 'imap.gmail.com'
                        consumer_emails = config['consumer_emails']
                        dev_emails = config['dev_emails']

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
