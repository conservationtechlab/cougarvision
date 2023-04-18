'''Get Cam Location

This script defines a function called cam_location where an http request is performed
to retrieve the lat, long, and unique id for the camera trap that the current classified
image from strikeforce has in earthranger so that the event may be linked to the specific
camera.
'''

import requests

def cam_location(cam_name, token, authorization):
    '''Cam Location

    This function takes in the name of the camera in earthranger and returns
    the lat and longs of that camera as well as earthrangers unique identifyer
    for it.

    Args:
    cam_name: the name (B001, B002...) of the camera as it is in earthranger 'str'
    token: the token for api calls in earthranger 'str'
    authorization: another token for ER api calls as specified in config yml 'str'

    Return: lat and longs of the camera location as a list and the unique id of the camera
        in earthranger as a 'str'
    '''
    
    url = 'https://sagebrush.pamdas.org/api/v1.0/subjects/?name=' + cam_name
    headers = {
        'X-CSRFToken': token,
        'Authorization': authorization,
        'Accept': 'application/json'
    }
    response = requests.get(url, headers=headers)
    response_json = response.json()

    cam_locations = response_json['data'][0]['last_position']['geometry']['coordinates']
    subject_id = response_json['data'][0]['id']
    return(cam_locations, subject_id)
