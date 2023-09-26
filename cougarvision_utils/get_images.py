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
import urllib.request
import os.path
import requests
import numpy as np
import logging


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
    response = requests.get(call, headers={"X-User-Email": username,
                                           "X-User-Token": auth_token})
    print(response.text)
    info = json.loads(response.text)
    return info


def fetch_image_api(config):
    '''
    Takes in config values and returns info about each new photo
    on strikeforce since the last run of the program

    Args:
    config: unpacked config string values from fetch_and_alert.yml

    Returns: a nested array of information regarding each photo that is to be
        run through the detector, includes only new photos since last run
    '''
    camera_names = dict(config['camera_names'])
    base = config['strikeforce_api']
    accounts = config['username_scraper']
    tokens = config['auth_token']
    path = "./last_id.txt"
    password = config['password_scraper']
    checkfile = os.path.exists(path)
    if checkfile is False:
        new_file = open("last_id.txt", "x")
        new_file.close()
        first_id = str(0) # function to get the most recent id from sf)
        new_file = open('last_id.txt', 'w')
        new_file.writelines(first_id)
        new_file.close()
    id_file = open('last_id.txt', 'r')
    last_id = id_file.readlines()
    id_file.close()
    for line in last_id:
        line.strip()
    last_id = int(line)
    id_file.close()
    photos = []
# 5 second delay between captures, maximum 12 photos between checks
    for account, token in zip(accounts, tokens):
        data = request_strikeforce(account, token, base,
                                   "photos/recent", "limit=12")
        photos += data['photos']['data']

    new_photos = []
    for i in range(len(photos)):
        if int(photos[i]['id']) > last_id:
            info = photos[i]['attributes']
            print(info)
            try:
                camera = camera_names[photos[i]['relationships']
                                     ['camera']['data']['id']]
            except KeyError:
                logging.warning('Cannot retrieve photo from camera\
                as there is no asssociated ID in the config file')
                continue
            newname = config['save_dir'] + camera
            newname += "_" + info['file_thumb_filename']
            print(newname)
            urllib.request.urlretrieve(info['file_thumb_url'], newname)
            new_photos.append([photos[i]['id'],
                               info['file_thumb_url'], newname])

    new_photos = np.array(new_photos)
    if len(new_photos) > 0:  # update last image
        new_last = max(new_photos[:, 0])
        new_id = str(new_last)
        thefile = open('last_id.txt', 'w')
        thefile.writelines(new_id)
        thefile.close()
    return new_photos
