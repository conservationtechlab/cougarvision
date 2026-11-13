import datetime
import logging
from email.utils import parsedate_to_datetime

import pandas as pd
import yaml
from PIL import Image
from imbox import Imbox


def imap_setup(host, username, password):
    return Imbox(host, username=username, password=password, ssl=True, ssl_context=None, starttls=False)


def fetch_emails(imap_server, emails, timestamp):
    print("Fetching emails")
    # Filter only messages sent from Camera that have not been read yet
    messages_group = []
    for email in emails:
        messages_group.append(imap_server.messages(sent_from=email, unread=True))
    return messages_group


def extractAttachments(messages_group, mail, image_path):
    rows = []
    # Loop through each message and append attachment to images array
    for messages in messages_group:
        for (uid, message) in messages:
            # grab timestamp
            # grab camera_name
            # grab picture id
            print(message.sent_from[0]['email'])
            if message.sent_from[0]['email'] == "support@bigfootsmtp.com":
                picture_id = message.subject[:-4]
                camera_name = "b019"
                timestamp = datetime.datetime.strptime(message.body['plain'][0].split("Time:(")[1][:-2],
                                                       "%m/%d/%Y  %H:%M:%S")
            elif message.sent_from[0]['email'] == "noreply@wirelesstrophycam.com":
                camera_name = message.subject.split(' ')[1][1:-1]
                message_body = message.body['plain'][0]
                split1 = message_body.split(":")
                print(split1)
                picture_id = split1[1][:-11]
                if len(split1) > 2:
                    time_string = split1[2][1:] + ":" + split1[3] + ":" + split1[4][:2]
                    timestamp = datetime.datetime.strptime(time_string, "%m/%d/%Y  %H:%M:%S")
                else:
                    timestamp = datetime.datetime.strptime("01/01/2000 01:01:01", "%m/%d/%Y  %H:%M:%S")
            else:
                # GardePro: one email can mix cameras, so name/time are read per photo
                camera_name = picture_id = timestamp = None

            for idx, attachment in enumerate(message.attachments):
                filename = attachment.get('filename') or ''
                if not (attachment.get('content-type', '').startswith('image/')
                        or filename.lower().endswith(('.jpg', '.jpeg', '.png'))):
                    continue
                img = Image.open(attachment.get('content'))
                if camera_name is None:
                    # GardePro writes the camera name set in the app to EXIF Artist
                    exif = img.getexif()
                    cam = str(exif.get(315) or '').strip()
                    if not cam:
                        logging.warning('skipped GardePro img %s: no camera name '
                                        'set in GardePro app', filename)
                        continue
                    pid = filename.rsplit('.', 1)[0]
                    ts = exif.get_ifd(0x8769).get(36867)  # DateTimeOriginal, local
                    ts = (ts.replace(':', '-', 2) if ts else
                          parsedate_to_datetime(message.date).strftime('%Y-%m-%d %H:%M:%S'))
                else:
                    cam, pid, ts = camera_name, picture_id, timestamp
                # Create path to save image
                path = f'{image_path}{cam}_{pid}_{ts}_{idx}.jpg'
                # Save image
                img.convert('RGB').save(path)
                # Append to dataframe
                rows.append([pid, '', path])
            mail.mark_seen(uid)
    return rows
