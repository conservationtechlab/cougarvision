'''Get Images

This module defines multiple functions including
request_strikeforce and fetch_image_api, fetch_and_alert.py
calls fetch_image_api which calls request_strikeforce inside
itself. request_strikeforce includes an api call to strikeforce
while fetch_image_api uses the info retrieved to label and
store each image and put it in an array with relevant info so
that it can later be classified. fetch_image_api depends on
last_id.txt as well, but it creates a new one if there is not
one currently present.

'''

import json
import time
import urllib.request
import logging
import requests
import numpy as np

# pylint: disable=pointless-string-statement
'''
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
'''


def request_strikeforce(username, auth_token, base, request, parameters):
    '''
    Takes in auth values and api call parameters and returns the data about
    the specified images from strikeforce.

    Args:
    username: string strikeforce username
    base: the main strikeforce api link
    auth_token: api token for strikeforce
    request: the strikeforce api specific request type
    parameters: specifications for strikeforce about what exact info is
            wanted from whatever api call is made

    Returns: a json object with the retrieved info of new images from
        strikeforce
    '''
    call = base + request + "?" + parameters
    
    # Try request twice and catch timeout exceptions
    try:
        response = requests.get(call, headers={"X-User-Email": username,
                                           "X-User-Token": auth_token},
                                             timeout=10)
    except requests.exceptions.Timeout:
        print("Request timed out. Waiting 10 seconds then trying again.")
        time.sleep(10)
        try:
            response = requests.get(call, headers={"X-User-Email": username,
                                            "X-User-Token": auth_token},
                                             timeout=10)
        except requests.exceptions.Timeout:
            print("Request timed out for a second time.")

    print(response.text)
    info = json.loads(response.text)
    return info


def fetch_image_api(config):# pylint: disable=too-many-locals
    '''
    Takes in config values and returns info about each new photo
    on strikeforce since the last run of the program

    Args:
    config: unpacked config string values from fetch_and_alert.yml

    Returns: a nested array of information regarding each photo that is to be
        run through the detector, includes only new photos since last run
    '''

    # id_path is the path to the id text file
    path = config.id_path
    # try creating file throw exception if it
    # does not exist
    try:
        with open(path,"x", encoding= "utf-8") as f:
            f.write(str(0)) # write first ID from sf
    except FileExistsError:
        print(path," already exists, exlusive creation aborted.")

    # read id from the .txt file
    with open (path, "r", encoding= "utf-8") as f:
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
        with open(path, "w", encoding= "utf-8") as f:
            f.writelines(new_id)

    return new_photos
