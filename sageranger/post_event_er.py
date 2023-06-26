'''Post Event ER

This module defines a function called post_event which creates a report
in Earthranger that shows up on the map and returns the id that event.
'''
import json
from datetime import datetime, timedelta
from sageranger.get_cam_location import cam_location

import requests

def post_event(label, cam_name, token, authorization):
    '''Post Event

    This function takes in the camera name and label of an image and returns
    the unique id of an event created in Earthranger for the specific image to
    be added. It calls cam_location() in order to make the event appear in the 
    correc goegraphical location on the Earthranger instance.

    Args:
    label: a string value of the animal that the image was classified as
    cam_name: a string of the specific name of the camera that the image came from
        as it also is in Earthranger
    token: unique token for ER to authenticate http request, defined in config yml
    authorization: the other auth token for ER as defined in config yml, this was
        retrieved from the interactive api on ER
        https://<YOUR INSTANCE>.pamdas.org/api/v1.0/docs/interactive/
    label: the name of the animal identified to be sent, used as title of jpeg

    Returns: the unique id of the event that was posted to be used to attach
        an image to, event must be created before image is added
    '''
    URL = 'https://sagebrush.pamdas.org/api/v1.0/activity/events/'

    Headers = {
      'X-CSRFToken': token,
      'Authorization': authorization
    }
   
    

    cam, subject_id = cam_location(cam_name, token, authorization)
    lat = cam[1]
    longi = cam[0]

    event_data = {
        "event_type": "cougarvision_detection",
        "priority": 100,
        "state": "active",
        "location": {
            "latitude": lat,
            "longitude": longi
        },
        "event_details": {
            "cam_name": cam_name,
            "animal_label": label
        },
        "event_category": "monitoring",
        "related_subjects": [
                {
                    "content_type": "observations.subject",
                    "id": subject_id
                }
        ]
}
    new_event = requests.post(URL, headers=Headers, json=event_data)
    response_json = new_event.json()

    return response_json['data']['id']
