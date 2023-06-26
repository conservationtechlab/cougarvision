'''Attach Image ER

This module defines a function called attach_image that attaches the classified
image with a bounding box, named according to the classified animal to an event
in earthranger using the api.
'''
import requests

def attach_image(event_id, img_byte, token, authorization, label):
    '''Attach Image

    This function attaches a classified and labeled image to an event in Earthranger
    as an alert system.

    Args:
    event_id: the unique id provided by Earthranger to be able to associate
        an image here with an event alert previously created for for each
        image to be added
    img_byte: the image to be posted in binary because that is how ER accepts it
    token: unique token for ER to authenticate http request, defined in config yml
    authorization: the other auth token for ER as defined in config yml, this was
        retrieved from the interactive api on ER
        https://<YOUR INSTANCE>.pamdas.org/api/v1.0/docs/interactive/
    label: the name of the animal identified to be sent, used as title of jpeg

    Returns: the http request response code to tell us if the call worked or not
    '''

    url = 'https://sagebrush.pamdas.org/api/v1.0/activity/event/' + event_id + '/files'

    unique_image_name = label + ".jpeg"

    content_type = 'application/jpeg'
    headers = {
        'X-CSRFToken': token,
        'Authorization': authorization
    }

    files = {'filecontent.file': (unique_image_name, img_byte, content_type)}
    response = requests.post(url, headers=headers, files=files)

    return response
