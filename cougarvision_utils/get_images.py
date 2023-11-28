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
import logging
import requests
import numpy as np
from cougarvision_visualize.visualize_helper import get_last_file_number
from cougarvision_visualize.visualize_helper import create_folder


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
    try:
        logging.info("Getting new image data from Strikeforce")
        response = requests.get(call, headers={"X-User-Email": username,
                                               "X-User-Token": auth_token})
        try:
            info = json.loads(response.text)
            return info
        except json.decoder.JSONDecodeError:
            logging.warning('An error occurred while decoding JSON')
            info = 0
            return info
    except requests.exceptions.ConnectionError:
        logging.warning("Connection Error, max retries exceeded")
        info = 0
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
    visualize_output = config['visualize_output']
    unlabeled_img = config['path_to_unlabeled_output']
    checkfile = os.path.exists(path)
    if checkfile is False:
        new_file = open("last_id.txt", "x")
        new_file.close()
        first_id = str(0)  # function to get the most recent id from sf)
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
        if data == 0:
            new_photos = []
            logging.warning('Returning to main loop after failed http request')
            return new_photos
        else:
            photos += data['photos']['data']

    new_photos = []
    photos = sorted(photos, key=lambda x: x['attributes']['original_datetime'])
    for i in range(len(photos)):
        if int(photos[i]['id']) > last_id:
            info = photos[i]['attributes']
            logging.info(info)
            try:
                camera = camera_names[photos[i]['relationships']
                                      ['camera']['data']['id']]
            except KeyError:
                logging.warning('skipped img: no associated cam ID')
                continue
            newname = config['save_dir'] + camera
            newname += "_" + info['file_thumb_filename']
            urllib.request.urlretrieve(info['file_thumb_url'], newname)
            new_photos.append([photos[i]['id'],
                               info['file_thumb_url'], newname])

            if visualize_output is True:
                file_path = create_folder(unlabeled_img)
                newname = file_path + 'image'
                new_file_num = get_last_file_number(file_path)
                new_file_num = new_file_num + 1
                new_file_num = str(new_file_num)
                newname += "_" + new_file_num
                urllib.request.urlretrieve(info['file_thumb_url'], newname)

    new_photos = np.array(new_photos)
    if len(new_photos) > 0:  # update last image
        new_last = max(new_photos[:, 0])
        new_id = str(new_last)
        thefile = open('last_id.txt', 'w')
        thefile.writelines(new_id)
        thefile.close()
    return new_photos
