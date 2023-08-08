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
    '''SMTP Setup

    This function creates a simple mail transfer protocol by taking in a
    host email, a username and password for an email.

    Args:
    username: username for email to send message from, string from config
    password: password for email message will be sent from, string from config
    host: IMAP protocol to download gmail messages, initialized in
    detect_img.py

    Returns: SMTP_SSL object logged into the mailing account specified in
    config yml
    '''
    # Init sending mail
    smtp_server = SMTP_SSL(host, port=SMTP_SSL_PORT)
    smtp_server.set_debuglevel(1)  # Show SMTP server interactions
    smtp_server.login(username, password)
    return smtp_server


def send_alert(alert, img, smtp_server, from_email, to_emails, dev, conf):
    '''Send Alert

    This function takes in the animal label, the image of the animal of
    interest, the SMTP server created, and the to and from emails to send
    the alert containing that specific image along with the confidence value.

    Args:
    alert: label of animal that the alert is being created for
    conf: confidence value of the classifier that the animal it says it
    is is the animal it is
    img: the PIL.Image of the image that is to be sent, to be converted
    to binary
    smtp_server: where the email is to be sent from, SMTP_SSL object
    from_email: the outgoing address for the alert
    to_emails: the emails the alert will be sent to, defined in config yml file
    '''
    # Construct Email Content
    email_message = EmailMessage()
    email_message.add_header('To', ', '.join(to_emails))
    email_message.add_header('From', from_email)
    email_message.add_header('Subject', 'Alert!')
    email_message.add_header('X-Priority', '1')  # Urgency, 1 highest, 5 lowest
    if dev == 0:
        message = "Potential " + alert + " detected by CougarVision "\
                  + "system.\n\nPlease review attached image to verify"\
                  + " detection. Cougarvision is set to be sensitive to"\
                  + " avoid missing species of interest so other animals "\
                  + "and artifacts have been known to trigger the system."
    elif dev != 0:
        message = "Potential " + alert + " detected with confidence value: "\
                  + conf

    email_message.set_content(message)

    # Prepare Image format
    binary_data = img.getvalue()

    # Attach image to email
    filename = 'detection.jpg'
    maintype, _, subtype = (mimetypes.guess_type(filename)[0] or
                            'application/octet-stream').partition("/")
    email_message.add_attachment(binary_data, maintype=maintype,
                                 subtype=subtype, filename=filename)

    # Server sends email message
    server = smtp_server
    server.send_message(email_message)


def checkin(to_emails, username, password, host):
    '''Sends server status to specified email at specified time interval'''
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
