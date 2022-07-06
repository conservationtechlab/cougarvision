# Import local utilities
import time
import warnings
from datetime import datetime
from io import BytesIO

import pandas as pd
import schedule
import yaml
from PIL import Image
from animl import FileManagement, ImageCropGenerator, DetectMD
from tensorflow import keras

from cougarvision_utils.alert import sendAlert, smtp_setup
from cougarvision_utils.fetch_emails import imap_setup, fetch_emails, extractAttachments
from cougarvision_utils.web_scraping import fetch_images

# Numpy FutureWarnings from tensorflow import
warnings.filterwarnings('ignore', category=FutureWarning)

# Load Configuration Settings from YML file
with open("web_scraping.yml", 'r') as stream:
    config = yaml.safe_load(stream)
    # Set Email Variables for fetching
    username = config['username']
    password = config['password']
    from_emails = config['from_emails']
    to_emails = config['to_emails']
    csv_path = config['csv_path']
host = 'imap.gmail.com'

# Model Variables
detector_model = config['detector_model']
classifier_model = config['classifier_model']

# Model Setup
checkpoint_path = config['checkpoint_path']
checkpoint_frequency = config['checkpoint_frequency']

# Classifier Model
model = keras.models.load_model(classifier_model)

# Set Confidence
confidence_threshold = config['confidence']

# Set threads for load_and_crop
threads = config['threads']
classes = config['classes']
timestamp = datetime.now()


def detect(images):
    if len(images) > 0:
        # extract paths from dataframe
        image_paths = images.file.values.tolist()

        # Run Detection
        results = DetectMD.load_and_run_detector_batch(image_paths, detector_model, checkpoint_path,
                                                       confidence_threshold, checkpoint_frequency, [])
        # Parse results
        df = FileManagement.parseMD(results)

        # filter out all non animal detections
        animalDataframe, otherDataframe = FileManagement.filterImages(df)

        # run classifier on animal detections if there are any
        if not animalDataframe.empty:
            # create generator for images
            generator = ImageCropGenerator.GenerateCropsFromFile(animalDataframe)
            # Run Classifier
            predictions = model.predict(generator)
            # Parse results
            maxDataframe = FileManagement.parseCM(animalDataframe, otherDataframe, predictions, classes)

            # Creates a large data frame with all relevant data
            cougars = animalDataframe
            cougars['prediction'] = maxDataframe['class']
            cougars['prediction_conf'] = pd.DataFrame(predictions).max(axis=1)
            # Add relevant data to cougars dataframe from original images dataframe
            cougars = cougars.merge(images)
            # drops all non cougar detections
            cougars = cougars[cougars['prediction'].astype(str) == 'cougar']
            # reset dataframe index
            cougars = cougars.reset_index(drop=True)
            # Sends alert for each cougar detection
            for idx in range(len(cougars.index)):
                label = cougars.at[idx, 'prediction']
                prob = cougars.at[idx, 'prediction_conf']
                img = Image.open(cougars.at[idx, 'file'])
                imageBytes = BytesIO()
                img.save(imageBytes, format=img.format)
                smtp_server = smtp_setup(username, password, host)
                sendAlert(label, prob, imageBytes, smtp_server, username, to_emails)
            # Write Dataframe to csv
            cougars.to_csv(f'{csv_path}dataframe_{datetime.now().strftime("%m-%d-%Y_%H:%M:%S")}')


def run_emails():
    # Gets a list of attachments from unread emails from bigfoot camera
    mail = imap_setup(host, username, password)
    global timestamp
    images = extractAttachments(fetch_emails(mail, from_emails, timestamp), mail)
    detect(images)


def run_scraper():
    images = fetch_images()

    detect(images)


def main():
    run_emails()
    run_scraper()


main()

schedule.every(10).minutes.do(main)

while True:
    schedule.run_pending()
    time.sleep(1)
