"""Script for obtaining strikeforce camera IDs"""
import json
import requests


def get_data(base, request, parameters, username, authentication_token):
    '''Function for retrieving camera info from strikeforce
    Args:
        base: strikeforce api link
        request: cameras
        parameters: none
        username: strikeforce username email
        authentication_token: auth token obtained from strikeforceget.py

    Returns:
        json with all camera info'''
    call = base + request + "?" + parameters
    headers = {"X-User-Email": username, "X-User-Token": authentication_token}
    response = requests.get(call, headers=headers, timeout=20)
    data_response = response.text
    return json.loads(data_response)


BASE = "https://api.strikeforcewireless.com/api/v2/"
REQUEST = "cameras"
PARAMETERS = ""
USERNAME = "<insert username>"
AUTH_TOKEN = "<insert auth_token>"

data = get_data(BASE, REQUEST, PARAMETERS, USERNAME, AUTH_TOKEN)
pretty_json = json.dumps(data, indent=4)
cameras = []
list_of_cam_info = data['data']
for _, i in enumerate(list_of_cam_info):
    cameras = list_of_cam_info[i]['id']
    print(cameras)
