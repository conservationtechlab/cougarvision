'''Fetch and Alert

This script allows users to retrieve thumbnail images uploaded
from cellular camera traps, classify them by species, and send
the photo as an alert to a specified end point.

This script and its modules depend on some but not all scripts
in this module including, cougarvision_utils/alert.py, cropping.py,
detect_img.py, get_images.py, ImageCropGenerator.py, and
earthranger_utils/attach_image_er.py, and post_event_er.py, and the
config/fetch_and_alert.yml and last_id.txt.

One must configure their individual fetch_and_alert.yml file to fit
their needs by ensuring the file paths, usernames and passwords, camera
dictionary, and image classifiers are correct. The .yml file is also
where one can choose whether they would like email alerts or to send
the classified images to Earthranger.
'''

# Import local utilities
import argparse
import time
import warnings
from datetime import datetime as dt
from email.message import EmailMessage
import yaml
import schedule
from cougarvision_utils.detect_img import detect
from cougarvision_utils.alert import smtp_setup
from cougarvision_utils.get_images import fetch_image_api


# Numpy FutureWarnings from tensorflow import
warnings.filterwarnings('ignore', category=FutureWarning)
# Parse arguments
PARSER = argparse.ArgumentParser(description='Retrieves images from \
                                 email & web scraper & runs detection')
PARSER.add_argument('config', type=str, help='Path to config file')
ARGS = PARSER.parse_args()
CONFIG_FILE = ARGS.config
# Load Configuration Settings from YML file
with open(CONFIG_FILE, 'r', encoding='utf-8') as stream:
    CONFIG = yaml.safe_load(stream)
# Set Email Variables for fetching
USERNAME = CONFIG['username']
PASSWORD = CONFIG['password']
TO_EMAILS = CONFIG['to_emails']
HOST = 'imap.gmail.com'


# Set interval for checking in
CHECKIN_INTERVAL = CONFIG['checkin_interval']


def run_scraper():
    ''''Gets the new images from strikeforce and runs a pre-trained
    detector model on them and sends images with animals of interest to
    specified emails'''
    print('Starting Web Scraper')
    images = fetch_image_api(CONFIG)
    print('Finished Web Scraper')
    print('Starting Detection')
    detect(images, CONFIG)
    print('Finished Detection')


def main():
    '''Runs the fetching and alerting functions and begins scheduler
    to loop the fetching and alerting periodically'''
    # Run the scheduler
    print("Running fetch_and_alert")
    # run_emails()
    run_scraper()
    print("Sleeping since: " + str(dt.now()))


def checkin():
    '''Sends server status to specified email at specified time interval'''
    print("Checking in at: " + str(dt.now()))
    # Construct Email Content
    email_message = EmailMessage()
    email_message.add_header('To', ', '.join(TO_EMAILS))
    email_message.add_header('From', USERNAME)
    email_message.add_header('Subject', 'Checkin')
    email_message.add_header('X-Priority', '1')  # Urgency, 1 highest, 5 lowest
    email_message.set_content('Still Alive :)')
    # Server sends email message
    smtp_server = smtp_setup(USERNAME, PASSWORD, HOST)
    server = smtp_server
    server.send_message(email_message)


main()

schedule.every(10).minutes.do(main)
schedule.every(CHECKIN_INTERVAL).hours.do(checkin)

while True:
    schedule.run_pending()
    time.sleep(1)
