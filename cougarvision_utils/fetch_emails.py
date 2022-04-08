from imbox import Imbox

import datetime
import re 
def imap_setup(host, username, password):
    return Imbox(host, username=username, password=password, ssl=True, ssl_context=None, starttls=False)


def fetch_emails(imap_server,emails,timestamp):
    print("Fetching emails")
    # Filter only messages sent from Camera that have not been read yet
    messages_group = []
    for email in emails:
        messages_group.append(imap_server.messages(sent_from=email, unread=True) )
        # messages_group.append(imap_server.messages(sent_from=email, date__gt=timestamp) )
    return messages_group

def extractAttachments(messages_group,mail):
    images = []
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
                timestamp = datetime.datetime.strptime(message.body['plain'][0].split("Time:(")[1][:-2], "%m/%d/%Y  %H:%M:%S")
            elif message.sent_from[0]['email'] == "noreply@wirelesstrophycam.com":
                camera_name = message.subject.split(' ')[1][1:-1]
                message_body = message.body['plain'][0]
                split1 = message_body.split(":")
                picture_id = split1[1][:-11]
                if (len(split1) > 2):
                    time_string = split1[2][1:] + ":" + split1[3] + ":" + split1[4][:2] 
                    timestamp = datetime.datetime.strptime(time_string, "%m/%d/%Y  %H:%M:%S")
                else:
                    timestamp = datetime.datetime.strptime("01/01/2000 01:01:01", "%m/%d/%Y  %H:%M:%S")


            for idx, attachment in enumerate(message.attachments):
                images.append((attachment.get('content'),camera_name,timestamp,picture_id))
                # mail.mark_seen(uid)
    return images


