# Import local utilities
import argparse
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
from cougarvision_utils.cropping import draw_bounding_box_on_image
from cougarvision_utils.fetch_emails import imap_setup, fetch_emails, extractAttachments
from cougarvision_utils.web_scraping import fetch_images

# Numpy FutureWarnings from tensorflow import
warnings.filterwarnings('ignore', category=FutureWarning)
# Parse arguments
parser = argparse.ArgumentParser(description='Retrieves images from email and web scraper and runs detection')
parser.add_argument('config', type=str, help='Path to config file')
args = parser.parse_args()
config_file = args.config
# Load Configuration Settings from YML file
with open(config_file, 'r') as stream:
    config = yaml.safe_load(stream)
# Set Email Variables for fetching
username = config['username']
password = config['password']
from_emails = config['from_emails']
to_emails = config['to_emails']
home_dir = config['home_dir']
csv_path = home_dir + config['csv_path']
host = 'imap.gmail.com'

# Model Variables
detector_model = home_dir + config['detector_model']
classifier_model = home_dir + config['classifier_model']

# Model Setup
checkpoint_path = home_dir + config['checkpoint_path']
checkpoint_frequency = config['checkpoint_frequency']

# Classifier Model
model = keras.models.load_model(classifier_model)

# Set Confidence and target
confidence_threshold = config['confidence']
targets = config['alert_targets']

# Set threads for load_and_crop
threads = config['threads']
classes = home_dir + config['classes']
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
        if not df.empty:
            animalDataframe, otherDataframe = FileManagement.filterImages(df)
            # run classifier on animal detections if there are any
            if not animalDataframe.empty:
                # create generator for images
                generator = ImageCropGenerator.GenerateCropsFromFile(animalDataframe)
                # Run Classifier
                predictions = model.predict(generator, steps=len(generator))
                # Parse results
                parsed_df = FileManagement.parseCM(animalDataframe, otherDataframe, predictions, classes)
                # Creates a large data frame with all relevant data
                full_df = animalDataframe
                full_df['prediction'] = parsed_df['class']
                full_df['prediction_conf'] = pd.DataFrame(predictions).max(axis=1)
                # Add relevant data to full_df dataframe from original images dataframe
                full_df = full_df.merge(images)
                # Write Dataframe to csv
                full_df.to_csv(f'{csv_path}dataframe_{datetime.now().strftime("%m-%d-%Y_%H:%M:%S")}')
                # Sends alert for each 'target' detection
                for idx in range(len(full_df.index)):
                    label = full_df.at[idx, 'prediction']
                    prob = full_df.at[idx, 'prediction_conf']
                    if label in targets and prob >= confidence_threshold:
                        img = Image.open(full_df.at[idx, 'file'])
                        draw_bounding_box_on_image(img,
                                                   full_df.at[idx, 'bbox2'], full_df.at[idx, 'bbox1'],
                                                   full_df.at[idx, 'bbox2'] + full_df.at[idx, 'bbox4'],
                                                   full_df.at[idx, 'bbox1'] + full_df.at[idx, 'bbox3'],
                                                   clss=idx,
                                                   thickness=4,
                                                   expansion=0,
                                                   display_str_list=f'{label} {prob * 100}',
                                                   use_normalized_coordinates=True,
                                                   label_font_size=25)
                        imageBytes = BytesIO()
                        img.save(imageBytes, format=img.format)
                        smtp_server = smtp_setup(username, password, host)
                        sendAlert(label, prob, imageBytes, smtp_server, username, to_emails)


def run_emails():
    # Gets a list of attachments from unread emails from bigfoot camera
    mail = imap_setup(host, username, password)
    global timestamp
    print('Starting Email Fetcher')
    images = extractAttachments(fetch_emails(mail, from_emails, timestamp), mail, config_file)
    print('Finished Email Fetcher')
    print('Starting Detection')
    detect(images)
    print('Finished Detection')


def run_scraper():
    print('Starting Web Scraper')
    images = fetch_images(config_file)
    print('Finished Web Scraper')
    print('Starting Detection')
    detect(images)
    print('Finished Detection')


def main():
    # Run the scheduler
    print("Running fetch_and_alert")
    # run_emails()
    run_scraper()
    print("Sleeping since: " + str(datetime.now()))


main()

schedule.every(10).minutes.do(main)

while True:
    schedule.run_pending()
    time.sleep(1)
