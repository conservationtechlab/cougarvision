"""Script for obtaining strikeforce camera IDs"""
import json
import requests

BASE = "https://api.strikeforcewireless.com/api/v2/"
REQUEST = "cameras"
PARAMETERS = ""
USERNAME = "<username>"
AUTH_TOKEN = "<token>"


def get_data(base, request, parameters, username, authentication_token):
    """Function for retrieving camera info from strikeforce
    Args:
        base: strikeforce api link
        request: cameras
        parameters: none
        username: strikeforce username email
        authentication_token: auth token obtained from strikeforceget.py

    Returns:
        json with all camera info
    """
    call = base + request + "?" + parameters
    headers = {"X-User-Email": username, "X-User-Token": authentication_token}
    response = requests.get(call, headers=headers, timeout=20)
    data_response = response.text
    return json.loads(data_response)


def main():
    """Main

    This function parses the json response to retireve
    the camera id and name and prints a dict of the values
    mapped to the corresponding camera for the config value
    'camera_names'
    """

    data = get_data(BASE, REQUEST, PARAMETERS, USERNAME, AUTH_TOKEN)
    cameras = []

    # print(list(data.keys()))
    # optional -- missing cam id for now
    # list_of_cam_data = data["data"]
    # for idx, d in enumerate(list_of_cam_data):
    # last_synced = list_of_cam_data[idx]['attributes']['last_sync_time']

    list_of_cam_info = data["included"]

    for idx, name in enumerate(list_of_cam_info):
        if list_of_cam_info[idx]['attributes'].get('camera_id') is not None:
            name = list_of_cam_info[idx]['attributes']['name']
            cam_id = list_of_cam_info[idx]['attributes']['camera_id']
            temp_tuple = cam_id, name
            cameras.append(temp_tuple)

    camera_dict = dict(cameras)
    print(camera_dict)


if __name__ == "__main__":
    main()
