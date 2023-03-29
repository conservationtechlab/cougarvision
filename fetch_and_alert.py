# Import local utilities
import argparse
import time
import warnings
from datetime import datetime as dt
from email.message import EmailMessage

import schedule
import yaml

from cougarvision_utils.detect_img import detect
from cougarvision_utils.alert import smtp_setup
from cougarvision_utils.get_images import fetch_image_api


# Numpy FutureWarnings from tensorflow import
warnings.filterwarnings('ignore', category=FutureWarning)
# Parse arguments
parser = argparse.ArgumentParser(description='Retrieves images from \
                                 email & web scraper & runs detection')
parser.add_argument('config', type=str, help='Path to config file')
args = parser.parse_args()
config_file = args.config
# Load Configuration Settings from YML file
with open(config_file, 'r') as stream:
    config = yaml.safe_load(stream)
# Set Email Variables for fetching
username = config['username']
password = config['password']
to_emails = config['to_emails']
host = 'imap.gmail.com'


# Set interval for checking in
checkin_interval = config['checkin_interval']


def run_scraper():
    print('Starting Web Scraper')
    images = fetch_image_api(config)
    print('Finished Web Scraper')
    print('Starting Detection')
    detect(images, config)
    print('Finished Detection')


def main():
    # Run the scheduler
    print("Running fetch_and_alert")
    # run_emails()
    run_scraper()
    print("Sleeping since: " + str(dt.now()))


def checkin():
    print("Checking in at: " + str(dt.now()))
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
