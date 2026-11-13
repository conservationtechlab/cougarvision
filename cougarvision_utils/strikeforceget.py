"""Script for getting strikeforce auth_token"""
import json
import requests

USERNAME = "bioreserve.cam@gmail.com"
PASSWORD = "Cougar2022!"


BASE = "https://api.strikeforcewireless.com/api/v2/"
REQUEST = "users/sign-in/"


def get_token(username, password, base, request):
    """Function for retrieving strikeforce token

    Args:
        username(str): username for sf account
        password(str): password for sf account
        base (str): url to sf
        request (str): type or url request
    """
    call = base + request
    body = json.dumps({"user": {"email": username, "password": password}})
    response = requests.post(url=call, data=body,
                             headers={"Content-Type":
                                      "application/json"},
                             timeout=20)
    response = response.text
    response = json.loads(response)
    authentication_token = response["meta"]["authentication_token"]
    print(authentication_token)


if __name__ == "__main__":
    get_token(USERNAME, PASSWORD, BASE, REQUEST)
