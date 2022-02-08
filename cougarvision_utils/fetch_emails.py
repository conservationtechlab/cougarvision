from imbox import Imbox



def imap_setup(host, username, password):
    return Imbox(host, username=username, password=password, ssl=True, ssl_context=None, starttls=False)


def fetch_emails(imap_server,emails):
    print("Fetching emails")
    # Filter only messages sent from Camera that have not been read yet
    messages_group = []
    for email in emails:
        messages_group.append(imap_server.messages(sent_from=email, unread=True) )
    return messages_group

def extractAttachments(messages_group,mail):
    images = []
    # Loop through each message and append attachment to images array
    for messages in messages_group:
        for (uid, message) in messages:
            for idx, attachment in enumerate(message.attachments):
                images.append(attachment.get('content'))
                mail.mark_seen(uid)
    return images
