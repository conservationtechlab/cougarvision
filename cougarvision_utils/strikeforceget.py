'''Script for getting strikeforce auth_token'''
import requests
import json

username = "<insert strikeforce username>"
password = "<insert strikeforce password>"


base = "https://api.strikeforcewireless.com/api/v2/"
request = "users/sign-in/"
call = base + request
body = json.dumps({"user": {"email": username, "password": password}})
encode = 'json'
response = requests.post(url=call, data=body,
                         headers={"Content-Type": "application/json"})
response = response.text
response = json.loads(response)
authentication_token = response["meta"]["authentication_token"]
print(authentication_token)
