# Import local utilities
from cougarvision_utils.alert import smtp_setup, sendAlert
from cougarvision_utils.cropping import load_to_crop, crop
from cougarvision_utils.fetch_emails import imap_setup, fetch_emails, extractAttachments
from cougarvision_utils.log_utils import log_classification,save_image
from cougarvision_utils.web_scraping import fetch_images

import json
import schedule
import time
import json
import sys
import warnings
import yaml
from tqdm import tqdm
from PIL import Image
from datetime import datetime
from concurrent import futures
from io import BytesIO
import uuid
import torch
from torchvision import transforms
import numpy as np
import humanfriendly


# Adds CameraTraps to Sys path, import specific utilities
with open("config/cameratraps.yml", 'r') as stream:
    camera_traps_config = yaml.safe_load(stream)
    sys.path.append(camera_traps_config['camera_traps_path'])
# Camera Traps utils
from detection.tf_detector import TFDetector
import visualization.visualization_utils as viz_utils


# Numpy FutureWarnings from tensorflow import
warnings.filterwarnings('ignore', category=FutureWarning)

# Load Configuration Settings from YML file
with open("config/fetch_and_alert.yml", 'r') as stream:
    config = yaml.safe_load(stream)

import tensorflow as tf


# Set Email Variables for fetching
username = config['username']
password = config['password']
from_emails = config['from_emails']
to_emails = config['to_emails']
host = 'imap.gmail.com'


# Model Variables
DETECTOR_MODEL = config['detector_model']
CLASSIFIER_MODEL = config['classifier_model']

# Model Setup
# Detector Model
tf_detector = TFDetector(DETECTOR_MODEL)

# Classifier Model
model = tf.keras.models.load_model(CLASSIFIER_MODEL)

# Set Confidence 
confidence_threshold = config['confidence']

# Load Classifier Labels
with open('labels/southwest_labels.txt', 'r') as f:
    data = f.read().splitlines()
southwest_labels = np.asarray(data)

timestamp = datetime.now()


def process_image(image_data,smtp_server):
    # Extract image data and convert to PIL
    frame = image_data[0]
    camera_name = image_data[1]
    picture_timestamp = image_data[2]
    picture_id = image_data[3]
    image = Image.open(frame)

    # Run Detector on Image
    result = tf_detector.generate_detections_one_image(
        image,
        '0',
        confidence_threshold
    )

    foundDetection = False
    # If there are detections run classifier & alerts
    for detection in result['detections']:
        if int(detection["category"]) == 3:
            continue
        foundDetection = True
        # Crops each bbox from detector
        img = Image.open(frame)
        bbox = detection['bbox']
        crop_result = crop(img, bbox)
        

        # Preprocessing image for classifier
        crop_result = tf.image.resize(crop_result,[456,456])
        
        crop_array = tf.keras.preprocessing.image.img_to_array(crop_result)
        crop_array = tf.expand_dims(crop_array, 0)

        # Perform Inference
        preds = model(np.asarray(crop_array))

        
        # Extract Top Prediction
        idx = preds.numpy().argmax(axis=-1)[0]
        prob = preds.numpy()[0][idx]
        label = southwest_labels[idx]

        # Draw Bounding Box
        viz_utils.draw_bounding_box_on_image(img,
                bbox[1], bbox[0], bbox[1] + bbox[3], bbox[0] + bbox[2],
                clss=idx,
                thickness=4,
                expansion=0,
                display_str_list=f'{label} {prob*100}',
                use_normalized_coordinates=True,
                label_font_size=25)

        img_file = f"{camera_name}_{timestamp.strftime('%m-%d-%Y_%H%M%S')}.jpg"
        

        log_classification(picture_timestamp,camera_name,picture_id,label,prob)
        

        # Send alert based on labels
        imageBytes = BytesIO()
        img.save(imageBytes,format=img.format)
        if label == "cougar" and prob >= .50:
            sendAlert(label, prob,imageBytes,smtp_server, username, to_emails)
    if foundDetection:
        save_image(image,img_file)
    

def main():
    # Reset Timestamp
    global timestamp
    timestamp = datetime.now()

    #Initialize smpt connection to send emails
    smtp_server = smtp_setup(username,password,host)

    # Gets a list of attachments from unread emails from bigfoot camera
    # mail = imap_setup(host, username, password)
    # images = extractAttachments(fetch_emails(mail,from_emails,timestamp),mail)

    # Calls web scraping fetch 
    web_images = fetch_images() 


    # Process images if they exist
    if len(web_images) > 0:
        for image in web_images:    
            process_image(image,smtp_server)
        

# Run code immediately  
main()

# Create schedule
schedule.every(10).minutes.do(main)

# Run Schedule
while True:
    schedule.run_pending()
    time.sleep(1)
