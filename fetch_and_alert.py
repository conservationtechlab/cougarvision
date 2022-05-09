# Import local utilities
from cougarvision_utils.fetch_emails import imap_setup, fetch_emails, extractAttachments
import schedule
import time
import json
import sys
import warnings
import yaml
from PIL import Image
from datetime import datetime
import numpy as np
import humanfriendly
from cougarvision_utils.ImageCropGenerator import GenerateCropsFromFile


# Adds CameraTraps to Sys path, import specific utilities
with open("config/cameratraps.yml", 'r') as stream:
    camera_traps_config = yaml.safe_load(stream)
    sys.path.append(camera_traps_config['camera_traps_path'])

# noinspection PyUnresolvedReferences
from detection.tf_detector import TFDetector
from detection.run_detector_batch import process_images

# Numpy FutureWarnings from tensorflow import
warnings.filterwarnings('ignore', category=FutureWarning)

# Load Configuration Settings from YML file
with open("config/fetch_and_alert.yml", 'r') as stream:
    config = yaml.safe_load(stream)

# Comment out if On GPU
# force_cpu = True
# if force_cpu:
#     import os
#     os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# threads = 1


import tensorflow as tf
from tensorflow import keras
# Set Email Variables for fetching
username = config['username']
password = config['password']
from_emails = config['from_emails']
to_emails = config['to_emails']
host = 'imap.gmail.com'

# Model Variables
detector_model = config['detector_model']
classifier_model = config['classifier_model']

# Model Setup
# Detector Model
start_time = time.time()
tf_detector = TFDetector(config['detector_model'])
print(f'Loaded detector model in {humanfriendly.format_timespan(time.time() - start_time)}')

# Classifier Model
model = keras.models.load_model(classifier_model)

# Set Confidence
confidence_threshold = config['confidence']

# Set threads for load_and_crop
threads = config['threads']

# Load Labels for classifier
labels_map = json.load(open('labels/labels_map.txt'))
labels_map = [labels_map[str(i)] for i in range(1000)]

with open('/home/kyra/mnt/machinelearning/Models/Southwest/classes.txt', 'r') as f:
    data = f.read().splitlines()
southwest_labels = np.asarray(data)

timestamp = datetime.now()


def main():
    # Gets a list of attachments from unread emails from bigfoot camera
    mail = imap_setup(host, username, password)
    global timestamp
    images = extractAttachments(fetch_emails(mail, from_emails, timestamp), mail)
    print(images)

    # Reset Timestamp
    timestamp = datetime.now()
    print(timestamp)

    if len(images) > 0:

        results = process_images(images, tf_detector, confidence_threshold)
        print(results)

        # Make a list of File names from the results
        # Make a list of bounding boxes
        bboxes = []
        filenames = []
        for dictionary in results:
            detections = dictionary['detections']
            if len(detections) > 0:
                detection_dictionary = detections[0]
                bboxes.append(detection_dictionary['bbox'])

                time.sleep(1)
                image = Image.open(dictionary['file'])
                image_path = f"image_{datetime.now().strftime('%m-%d-%Y_%H%M%S')}.jpg"
                image.save(image_path)
                filenames.append(image_path)

        # Make bboxes into array
        bboxes = np.array(bboxes)

        # Create Generator
        if len(filenames) > 0:
            generator = GenerateCropsFromFile(filenames, bboxes)
            predictions = model.predict(generator)
            print(predictions)


main()

schedule.every(10).minutes.do(main)

while True:
    schedule.run_pending()
    time.sleep(1)
