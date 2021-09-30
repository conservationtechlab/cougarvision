from imbox import Imbox



def imap_setup(host, username, password):
    return Imbox(host, username=username, password=password, ssl=True, ssl_context=None, starttls=False)


def fetch_emails(imap_server):
    print("Fetching emails")
    # Filter only messages sent from Camera that have not been read yet
    messages = imap_server.messages(sent_from='support@bigfootsmtp.com', unread=True) 
    return messages

def extractAttachments(messages,mail):
    images = []
    # Loop through each message and append attachment to images array
    for (uid, message) in messages:
        for idx, attachment in enumerate(message.attachments):
            images.append(attachment.get('content'))
            mail.mark_seen(uid)
    return images
