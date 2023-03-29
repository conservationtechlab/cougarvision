import json
import requests
import urllib.request
import numpy as np
import os.path


'''
#################

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
    call = base + request + "?" + parameters
    response = requests.get(call, headers={"X-User-Email": username,
                                           "X-User-Token": auth_token})
    info = json.loads(response.text)
    return info


def fetch_image_api(config):
    camera_names = dict(config['camera_names'])
    base = config['strikeforce_api']
    username = config['username_scraper']
# password = config['password_scraper']
    auth_token = config['auth_token']
    path = "./last_id.txt"
    checkfile = os.path.exists(path)
    if checkfile == False:
        f = open("last_id.txt", "x")
        f.close()
        first_id = str(0) # function to get the most recent id from sf)
        f = open('last_id.txt', 'w')
        f.writelines(first_id)
        f.close()
    thefile = open('last_id.txt', 'r')
    last_id = thefile.readlines()
    thefile.close()
    for line in last_id:
        line.strip()
    last_id = int(line)
    print(last_id)
    thefile.close()    
#    last_id = int(config['last_id'])

# auth_token = get_token(config_path)
# print(auth_token)
# 5 second delay between captures, maximum 12 photos between checks
    data = request_strikeforce(username, auth_token, base,
                               "photos/recent", "limit=12")
    photos = data['photos']['data']

    new_photos = []
    for i in range(len(photos)):
        if int(photos[i]['id']) > last_id:
            info = photos[i]['attributes']
            print(info)
            camera = camera_names[photos[i]['relationships']
                                  ['camera']['data']['id']]
            newname = config['save_dir'] + camera
            newname += "_" + info['file_thumb_filename']
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
