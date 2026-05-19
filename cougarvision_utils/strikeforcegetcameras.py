'''Script for obtaining strikeforce camera IDs'''
import requests
import json


def getData(base, request, parameters, username, authentication_token):
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
    response = requests.get(call, headers=headers)
    data = response.text
    return json.loads(data)


base = "https://api.strikeforcewireless.com/api/v2/"
request = "cameras"
parameters = ""
username = "<insert username>"
authentication_token = "<insert auth_token>"

data = getData(base, request, parameters, username, authentication_token)
pretty_json = json.dumps(data, indent=4)
cameras = []
list_of_cam_info = data['data']
for i in range(len(list_of_cam_info)):
    cameras = list_of_cam_info[i]['id']
    print(cameras)
