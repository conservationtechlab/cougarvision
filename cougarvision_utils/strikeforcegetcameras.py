import requests
import json

def getData(base, request, parameters, username, authentication_token):
    call = base + request + "?" + parameters
    headers = {"X-User-Email": username, "X-User-Token": authentication_token}
    response = requests.get(call, headers=headers)
    data = response.text
    return json.loads(data)

base = "https://api.strikeforcewireless.com/api/v2/"
request = "cameras"
parameters = ""
username = "bioreserve.cam@gmail.com"
authentication_token = "nWzMPu-ytLvatjyAL7kN"

data = getData(base, request, parameters, username, authentication_token)
pretty_json = json.dumps(data, indent=4)
print(pretty_json)
print(data)
cameras = []
list_of_cam_info = data['data']
for i in range(len(list_of_cam_info)):
    cameras = list_of_cam_info[i]['id']
    print(cameras)
