import datetime

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


def extractAttachments(messages_group, mail, config_path):
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
        image_path = config['image_path']
    df = pd.DataFrame(columns=['file', 'camera', 'time', 'image_id'])
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

            for idx, attachment in enumerate(message.attachments):
                # Create path to save image
                path = f'{image_path}{camera_name}_{timestamp}_{picture_id}.png'
                # Save image
                img = Image.open(attachment.get('content'))
                img.save(path)
                # Append to dataframe
                df = df.append({'file': path, 'camera_name': camera_name, 'time': timestamp,
                                'image_id': picture_id}, ignore_index=True)
                mail.mark_seen(uid)
    return df
