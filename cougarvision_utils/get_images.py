"""Get Images

This module defines multiple functions including
request_strikeforce and fetch_image_api, fetch_and_alert.py
calls fetch_image_api which calls request_strikeforce inside
itself. request_strikeforce includes an api call to strikeforce
while fetch_image_api uses the info retrieved to label and
store each image and put it in an array with relevant info so
that it can later be classified. fetch_image_api depends on
last_id.txt as well, but it creates a new one if there is not
one currently present.

Examples:
    API request example of get list of cameras::
        request <- "cameras"
        parameters <- ""
"""

import json
import time
import urllib.request
import logging
import requests
import numpy as np

# pylint: disable=pointless-string-statement
"""
#request examples
#get list of camaras
request <- "cameras"
parameters <- ""

#recent photo count
request <- "photos/recent/count"
parameters <- ""

#get recent photos across cameras
request <- "photos/recent"
parameters <- "limit=100"

#get photos from specific camera (will need to loop through pages)
request <- "photos"
parameters <- "page=3&sort_date=desc&camera_id[]=59681"

#get photos from specific camera filtered by date (will need
# to loop through pages)
request <- "photos"
parameters <- "page=1&sort_date=desc&camera_id[]=
#60272&date_start=2022-09-01&date_end=2022-10-07"

#get subscriptions
request <- "subscriptions"
parameters <- ""
"""


def request_strikeforce(username, auth_token, base, request, parameters):
    """Strikeforce API call request.
    
    Takes in auth values and api call parameters and returns the data 
    about the specified images from strikeforce.

    Args:
        username (str): String strikeforce username.
        base (str): The main strikeforce api link.
        auth_token (str): Api token for strikeforce.
        request (str): The strikeforce api specific request type.
        parameters(str): Specifications for strikeforce about what exact 
            info is wanted from the api call that is made.

    Raises:
        Exception: A broad exception raised when connection to internet or api
            request fails.

    Returns:
        ndarray: A multi-dimensional array with the retrieved info of new images
          from strikeforce. Info for each photo is in the format of 
          ['photo id']['strikeforce url']['file path'].
    """
    call = base + request + "?" + parameters

    # if there is no internet connection try 5 times before raising exception
    max_retries = 5
    for attempt in range(max_retries):

        try:
            response = requests.get(call, headers={"X-User-Email": username,
                                                   "X-User-Token": auth_token},
                                    timeout=20)
            print(response.text)
            info = json.loads(response.text)
            return info

        except requests.exceptions.ConnectionError as e:
            logging.warning("Failed to connect attempt: %s error %s",
                            {attempt + 1},
                            {e})
            print(f'Connection Error {attempt + 1}: {e}')
            time.sleep(15)  # wait 15 seconds
        except requests.exceptions.Timeout as e:
            logging.warning("Failed to connect to StrikeForce attempt: %s error %s",
                            {attempt + 1},
                            {e})
            print(f'Timeout Error {attempt + 1}: {e}')
            time.sleep(15)  # wait 15 seconds

    logging.error("Failed to connect after multiple attempts.")
    raise Exception("Failed to connect after multiple attempts.") # pylint says too broad 


def fetch_image_api(config):  # pylint: disable=too-many-locals
    """Retrives new photo information.

    Takes in config values and returns info about each new photo
    on strikeforce since the last run of the program.

    Args:
        config (ConfigInfo): unpacked config 
            string values from fetch_and_alert.yml

    Returns: 
        Array: A nested array of information regarding each photo 
            that is to be run through the detector, includes only
            new photos since last run.
    """
    # id_path is the path to the id text file
    path = config.id_path
    # try creating file throw exception if it
    # does not exist
    try:
        with open(path, "x", encoding="utf-8") as f:
            f.write(str(0))  # write first ID from sf
    except FileExistsError:
        print(path, " already exists, exlusive creation aborted.")

    # read id from the .txt file
    with open(path, "r", encoding="utf-8") as f:
        last_id = int(f.read().strip())

    photos = []
# 5 second delay between captures, maximum 12 photos between checks
# using config object
    for account, token in zip(config.accounts, config.auth_token):
        data = request_strikeforce(account, token, config.base,
                                   "photos/recent", "limit=12")
        photos += data['photos']['data']

    new_photos = []
    for _, photo in enumerate(photos):
        if int(photo['id']) > last_id:
            info = photo['attributes']

            print(info)

            try:

                camera = config.camera_names[photo['relationships']
                                             ['camera']['data']['id']]
            except KeyError:
                logging.warning('Cannot retrieve photo from camera\
                as there is no asssociated ID in the config file')
                continue

            newname = config.save_dir + camera
            newname += "_" + info['file_thumb_filename']
            # native extension from strikeforce is .JPG.jpeg for some reason
            stripped_name = newname.replace(".JPG.jpeg", ".jpg")
            urllib.request.urlretrieve(info['file_thumb_url'], stripped_name)
            new_photos.append([photo['id'],
                               info['file_thumb_url'], stripped_name])

    new_photos = np.array(new_photos)
    if len(new_photos) > 0:  # update last image
        new_last = max(new_photos[:, 0])
        new_id = str(new_last)
        # write new id to .txt file
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(new_id)

    return new_photos
