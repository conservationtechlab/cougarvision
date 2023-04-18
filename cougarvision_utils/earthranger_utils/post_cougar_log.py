'''Post Cougar Log

This module defines a function called is_cougar which adds an observation
to a specific camera in Earthranger with the time and the fact that a cougar
was detected.
'''
from datetime import datetime
import requests
from cougarvision_utils.earthranger_utils.get_cam_location import cam_location


def is_cougar(cam_name, token, authorization):
    '''Is Cougar

    This function takes in the camera name and http api tokens only if
    a cougar was detected, and it then creates an observation for the specific
    camera a cougar was detected at and logs the time so that there is a historical
    backlog for each camera of all its cougar detections.

    Args:
    cam_name: a string of the specific name of the camera that the image came from
        as it also is in Earthranger
    token: unique token for ER to authenticate http request, defined in config yml
    authorization: the other auth token for ER as defined in config yml, this was
        retrieved from the interactive api on ER
        https://<YOUR INSTANCE>.pamdas.org/api/v1.0/docs/interactive/
    '''
    
    headers = {
        'X-CSRFToken': token,
        'Authorization': authorization
        }

    current_time = datetime.utcnow()
    formatted_time = current_time.strftime('%Y-%m-%dT%H:%M:%S.%f') + 'Z'

    cam, subject_id = cam_location(cam_name, token, authorization)
    lat = cam[1]
    longi = cam[0]
    url = 'https://sagebrush.pamdas.org/api/v1.0/subject/' + subject_id + '/sources/'
    response = requests.get(url, headers=headers)
    response_json = response.json()

    source_id = response_json['data'][0]['id']

    url2 = 'https://sagebrush.pamdas.org/api/v1.0/observations/'

    payload = {"location": {"longitude": longi, "latitude": lat}, "recorded_at": formatted_time, "source": source_id, "device_status_properties": [{"value": "cougar", "label": "animal", "units": ""}], "additional": {"animal": "cougar"}}

    requests.post(url2, headers=headers, json=payload)
