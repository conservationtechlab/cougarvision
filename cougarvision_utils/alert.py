'''Alert

This script defines two functions, smtp_setup and send_alert. The first
function creates a server through which one can send an email from a
specified account. The second function creates the message containing a
classified animal of interest along with it's label to send to specified
emails via the smtp_server created.
'''

import mimetypes
from email.message import EmailMessage
from smtplib import SMTP_SSL, SMTP_SSL_PORT
from datetime import datetime as dt


def smtp_setup(username, password, host):
    """SMTP Setup

    This function creates a simple mail transfer protocol by taking in a
    host email, a username and password for an email.

    Args:
    username (str): username for email to send message from
        string from config
    password (str): password for email message will be sent from,
        string from config
    host (str): IMAP protocol to download gmail messages, initialized in
        detect_img.py

    Returns:
    SMTP_SSL object logged into the mailing account specified in
        config yml
    """
    # Init sending mail
    smtp_server = SMTP_SSL(host, port=SMTP_SSL_PORT)
    smtp_server.set_debuglevel(1)  # Show SMTP server interactions
    smtp_server.login(username, password)
    return smtp_server


def send_alert(config, alert, img, dev, conf):
    """Send Alert

    This function takes in the animal label, the image of the animal of
    interest, the SMTP server created, and the to and from emails to send
    the alert containing that specific image along with the confidence value.

    Args:
    alert (str): label of animal that the alert is being created for
    conf (float): confidence value of the classifier that the animal it says it
        is is the animal it is
    img(bytes): the PIL.Image of the image that is to be sent, to be converted
        to binary
    config (dict): holds the values of username, password, host,
        dev/consumeremails for email setup and info for recipients.
    """
    # Construct Email Content
    email_message = EmailMessage()
    email_message['To'] = ', '.join(config.consumer_emails)
    email_message['from'] = config.username
    email_message['Subject'] = 'Alert!'
    email_message['X-Priority'] = '1'  # Urgency, 1 highest, 5 lowest
    message = ""
    if dev == 0:
        message = "Potential " + alert + " detected by CougarVision "\
                  + "system.\n\nPlease review attached image to verify"\
                  + " detection. Cougarvision is set to be sensitive to"\
                  + " avoid missing species of interest so other animals "\
                  + "and artifacts have been known to trigger the system."
    elif dev != 0:
        message = "Potential " + alert + " detected with confidence value: "\
                  + conf

    # Prepare Image format
    binary_data = img.getvalue()

    # Attach image to email
    filename = 'detection.jpg'
    maintype, _, subtype = (mimetypes.guess_type(filename)[0] or
                            'application/octet-stream').partition("/")
    email_message.add_attachment(binary_data, maintype=maintype,
                                 subtype=subtype, filename=filename)

    # Server sends email message
    email_message.set_content(message)
    server = smtp_setup(config.username, config.password, config.host)
    server.send_message(email_message)
    server.quit()


def checkin(config):
    """Sends server status to specified email at specified time interval

    Args:
    config (dict): holds the values of username, password, host,
        dev/consumeremails for email setup and info for recipients.
    """
    print("Checking in at: " + str(dt.now()))

    # Construct Email Content
    email_message = EmailMessage()
    email_message['To'] = ', '.join(config.dev_emails)
    email_message['from'] = config.username
    email_message['Subject'] = 'Checkin'
    email_message.add_header('X-Priority', '1')  # Urgency, 1 highest, 5 lowest
    message = "still Alive :) "
    email_message.set_content(message)

    # Server sends email message
    server = smtp_setup(config.username, config.password, config.host)
    server.send_message(email_message)
    server.quit()
