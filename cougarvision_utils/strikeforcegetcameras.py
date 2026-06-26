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
USERNAME = "username"
AUTH_TOKEN = "token"


data = get_data(BASE, REQUEST, PARAMETERS, USERNAME, AUTH_TOKEN)
pretty_json = json.dumps(data, indent=4)
cameras = []
# print(list(data.keys()))

#list_of_cam_data = data["data"] 
#for idx, d in enumerate(list_of_cam_data): 
# optional print for logging purposes
# last_synced = list_of_cam_data[idx]['attributes']['last_sync_time']
# print("Name: " + name, "ID: " + id , "date: " + last_synced)

list_of_cam_info = data["included"]

for idx, (name) in enumerate(list_of_cam_info): 
    if list_of_cam_info[idx]['attributes'].get('camera_id') is not None:
        name = list_of_cam_info[idx]['attributes']['name']
        id = list_of_cam_info[idx]['attributes']['camera_id']
        print(idx," ",id, " ", name)

        temp_tuple = id, name 
        cameras.append(temp_tuple)

camera_dict = dict(cameras)
print(camera_dict)
