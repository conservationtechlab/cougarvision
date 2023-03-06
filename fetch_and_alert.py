# Import local utilities
import argparse
import time
import warnings
from datetime import datetime
from email.message import EmailMessage
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
from cougarvision_utils.get_images import fetch_image_api

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
log_dir = config['log_dir']
save_dir = config['save_dir']
host = 'imap.gmail.com'

# Model Variables
detector_model = config['detector_model']
classifier_model = config['classifier_model']
checkpoint_frequency = config['checkpoint_frequency']

# Classifier Model
model = keras.models.load_model(classifier_model)

# Set Confidence and target
confidence_threshold = config['confidence']
targets = config['alert_targets']
# Set interval for checking in
checkin_interval = config['checkin_interval']
# Set threads for load_and_crop
threads = config['threads']
classes = config['classes']
timestamp = datetime.now()



def detect(images):
  if len(images) > 0:
      # extract paths from dataframe
      image_paths = images[:,2]
      
      # Run Detection
      results = DetectMD.load_and_run_detector_batch(image_paths, detector_model, log_dir,
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
              predictions = model.predict_generator(generator, steps=len(generator), verbose=1)
              # Parse results
              maxDataframe = FileManagement.parseCM(animalDataframe, None, predictions, classes)
              # Creates a data frame with all relevant data
              cougars = maxDataframe[maxDataframe['class'].isin(targets)]
              # drops all detections with confidence less than threshold
              cougars = cougars[cougars['conf'] >= confidence_threshold]
              # reset dataframe index
              cougars = cougars.reset_index(drop=True)
              # Sends alert for each cougar detection
              for idx in range(len(cougars.index)):
                  label = cougars.at[idx, 'class']
                  prob = cougars.at[idx, 'conf']
                  img = Image.open(cougars.at[idx, 'file'])
                  draw_bounding_box_on_image(img,
                                             cougars.at[idx, 'bbox2'], cougars.at[idx, 'bbox1'],
                                             cougars.at[idx, 'bbox2'] + cougars.at[idx, 'bbox4'],
                                             cougars.at[idx, 'bbox1'] + cougars.at[idx, 'bbox3'],
                                             expansion=0,
                                             use_normalized_coordinates=True,)
                  imageBytes = BytesIO()
                  img.save(imageBytes, format=img.format)
                  smtp_server = smtp_setup(username, password, host)
                  sendAlert(label, prob, imageBytes, smtp_server, username, to_emails)
              # Write Dataframe to csv
              cougars.to_csv(f'{log_dir}dataframe_{datetime.now().strftime("%m-%d-%Y_%H:%M:%S")}')


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
    images = fetch_image_api(config_file)
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


def checkin():
    print("Checking in at: " + str(datetime.now()))
    # Construct Email Content
    email_message = EmailMessage()
    email_message.add_header('To', ', '.join(to_emails))
    email_message.add_header('From', username)
    email_message.add_header('Subject', 'Checkin')
    email_message.add_header('X-Priority', '1')  # Urgency, 1 highest, 5 lowest
    email_message.set_content('Still Alive :)')
    # Server sends email message
    smtp_server = smtp_setup(username, password, host)
    server = smtp_server
    server.send_message(email_message)


main()

schedule.every(10).minutes.do(main)
schedule.every(checkin_interval).hours.do(checkin)

while True:
    schedule.run_pending()
    time.sleep(1)
