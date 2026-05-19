"""Script for getting strikeforce auth_token"""
import json
import requests

USERNAME = "<insert strikeforce username>"
PASSWORD = "<insert strikeforce password>"


BASE = "https://api.strikeforcewireless.com/api/v2/"
REQUEST = "users/sign-in/"
CALL = BASE + REQUEST
body = json.dumps({"user": {"email": USERNAME, "password": PASSWORD}})
ENCODE = 'json'
response = requests.post(url=CALL, data=body,
                         headers={"Content-Type":
                                  "application/json"},
                         timeout=20)
response = response.text
response = json.loads(response)
authentication_token = response["meta"]["authentication_token"]
print(authentication_token)
