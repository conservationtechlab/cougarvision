import mimetypes
from email.message import EmailMessage
from smtplib import SMTP_SSL, SMTP_SSL_PORT


def smtp_setup(username, password, host):
    # Init sending mail 
    smtp_server = SMTP_SSL(host, port=SMTP_SSL_PORT)
    smtp_server.set_debuglevel(1)  # Show SMTP server interactions
    smtp_server.login(username, password)
    return smtp_server


def sendAlert(alert, conf, img, smtp_server, from_email, to_emails):
    # Construct Email Content
    email_message = EmailMessage()
    email_message.add_header('To', ', '.join(to_emails))
    email_message.add_header('From', from_email)
    email_message.add_header('Subject', 'Alert!')
    email_message.add_header('X-Priority', '1')  # Urgency, 1 highest, 5 lowest
    email_message.set_content('I found a ' + alert + ' with confidence ' + str(conf))

    # Prepare Image format
    binary_data = img.getvalue()

    # Attach image to email
    filename = 'detection.jpg'
    maintype, _, subtype = (mimetypes.guess_type(filename)[0] or 'application/octet-stream').partition("/")
    email_message.add_attachment(binary_data, maintype=maintype, subtype=subtype, filename=filename)

    # Server sends email message
    server = smtp_server
    server.send_message(email_message)
