import requests
import json
import urllib.request
import ruamel.yaml
import numpy as np


def get_token(config_path):
  yaml = ruamel.yaml.YAML()
  with open(config_path, 'r') as f:
      config = yaml.load(f)
  base = config['strikeforce_api'] # "https://api.strikeforcewireless.com/api/v2/"
  username = config['username_scraper']
  password = config['password_scraper']

  call = base + "users/sign-in/"

  body = {"user":{"email": username, "password":password}}

  response=requests.post(url=call,json=body)
  print(response)
  info = json.loads(response.text)

  auth_token= info['meta']['authentication_token']
  
  return auth_token

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

#get photos from specific camera filtered by date (will need to loop through pages)
request <- "photos"
parameters <- "page=1&sort_date=desc&camera_id[]=60272&date_start=2022-09-01&date_end=2022-10-07"

#get subscriptions
request <- "subscriptions"
parameters <- ""
'''


def request_strikeforce(username, auth_token, base, request, parameters):
  call = base + request + "?" + parameters
  response = requests.get(call, headers = {"X-User-Email":username, "X-User-Token":auth_token}) 
  info = json.loads(response.text)
  
  return info 




def fetch_images(config_path):
    yaml = ruamel.yaml.YAML()
    with open(config_path, 'r') as f:
        config = yaml.load(f)
        
    camera_names = dict(config['camera_names'])
    base = config['strikeforce_api'] 
    username = config['username_scraper']
    password = config['password_scraper']
    auth_token = config['auth_token']
    last_id = int(config['last_id'])

  #  auth_token = get_token(config_path)
  #  print(auth_token)
  # 5 second delay between captures, maximum 12 photos between checks
    data = request_strikeforce(username, auth_token, base, "photos/recent", "limit=12")
    photos = data['photos']['data']

    new_photos = []
    
    for i in range(len(photos)):
        if int(photos[i]['id']) > last_id:
            info = photos[i]['attributes']
            print(info)
            camera = camera_names[photos[i]['relationships']['camera']['data']['id']]
            newname = config['save_dir'] +  camera + "_" + info['file_thumb_filename']
         
            urllib.request.urlretrieve(info['file_thumb_url'], newname)
            new_photos.append([photos[i]['id'],info['file_thumb_url'],newname])

    new_photos= np.array(new_photos)
    if len(new_photos) > 0: # update last image
        new_last = max(new_photos[:,0])
        config['last_id'] = str(new_last)
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
    
    return new_photos
  


'''
def fetch_images(config_path):
    yaml = ruamel.yaml.YAML()
    with open(config_path, 'r') as f:
        config = yaml.load(f)
    site = config['site']
    site2 = config['site2']
    username = config['username_scraper']
    password = config['password_scraper']
    image_path = config['image_path']
    last_id = config['last_id']

    driver = webdriver.Chrome(ChromeDriverManager(chrome_type=ChromeType.CHROMIUM).install(), options=options)
    actions = ActionChains(driver)
    driver.maximize_window()
    driver.get(site)
    time.sleep(2)
    name = driver.find_element("id", "username")
    name.clear()
    name.send_keys(username)
    pw = driver.find_element("id", "password")
    pw.send_keys(password)
    pw.send_keys(Keys.RETURN)

    # wait to login
    time.sleep(2)

    # navigate to photos page
    driver.get(site2)
    # wait for javascript to load
    time.sleep(4)

    image_urls = driver.find_elements(By.CLASS_NAME, 'css-1vh28r')
    # make a dictionary with photo id as key and url as value
    element_dict = {}
    for image in image_urls:
        element_dict[image.get_attribute('data-photo-id')] = image

    # make a list of photo ids sorted from greatest to least
    photo_ids = [x.get_attribute('data-photo-id') for x in image_urls]
    photo_ids.sort(reverse=True)

    image_num = len(image_urls)
    df = pd.DataFrame(
        columns=['file', 'camera_name', 'time', 'date', 'temperature', 'moon', 'camera_id', 'alt', 'image_id', 'src'])
    skip = True
    first = True
    i = 0
    while i < image_num:
        picture_id = photo_ids[i]
        image_obj = element_dict[picture_id]
        if last_id == int(picture_id):
            print("Number of images found: " + str(i))
            driver.close()
            return df
        if first:
            first = False
            print("Looking for new images since last id: " + str(last_id))
            # Write the first image ID to file
            if skip:
                config['last_id'] = int(picture_id)
                with open(config_path, 'w') as f:
                    yaml.dump(config, f)
                skip = False
            # Get scroll height
            last_height = driver.execute_script("return document.body.scrollHeight")
            # Scrolls down to the bottom to dynamically load images
            SCROLL_PAUSE_TIME = 3.0
            while True:
                # Scroll down to bottom
                driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.PAGE_DOWN)
                # Wait to load page
                time.sleep(SCROLL_PAUSE_TIME)

                # Calculate new scroll height and compare with last scroll height
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                if last_id is not None:
                    image_urls = driver.find_elements(By.CLASS_NAME, 'css-1vh28r')
                    # make a list of photo ids sorted from greatest to least
                    photo_ids = [x.get_attribute('data-photo-id') for x in image_urls]
                    photo_ids.sort(reverse=True)
                    picture_id = photo_ids[-1]
                    if int(picture_id) < int(last_id):
                        picture_id = photo_ids[0]
                        break
                last_height = new_height
            # Go to the top of the page
            actions.send_keys(Keys.HOME)
            actions.perform()
            time.sleep(2)
            # load new number of images
            image_urls = driver.find_elements(By.CLASS_NAME, 'css-1vh28r')
            # make a dictionary with photo id as key and element as value
            element_dict = {}
            for image in image_urls:
                element_dict[image.get_attribute('data-photo-id')] = image
            # make a list of photo ids sorted from greatest to least
            photo_ids = [x.get_attribute('data-photo-id') for x in image_urls]
            photo_ids.sort(reverse=True)
            image_num = len(photo_ids)
            # Clicks on first image to bring up lightbox
            driver.find_element_by_xpath(
                '/html/body/div/section/main/div/div[2]/div/div/div/div/div[2]/div[1]/div/img').click()
            time.sleep(2)
        # Clicks arrow key to advance to next image
        else:
            driver.find_element_by_xpath('/html/body/div[4]/div/div/div/button[2]').click()
            time.sleep(2)
        # Get all information from image
        date_stamp = driver.find_element(By.CLASS_NAME, 'css-11c9ho7').text
        time_stamp = driver.find_element(By.CLASS_NAME, 'css-zz79lp').text
        # Converts time to 24 and correct timezone
        datetime_object = datetime.datetime.strptime(time_stamp, '%I:%M%p') + datetime.timedelta(hours=2)
        time_stamp = datetime_object.strftime("%H:%M:%S")
        temp_moon_stamps = driver.find_elements(By.CLASS_NAME, 'css-9kmbgq')
        temp_stamp, moon_stamp = temp_moon_stamps[0].text, temp_moon_stamps[1].text
        camera_id = image_obj.get_attribute('data-camera-id')
        alt = image_obj.get_attribute('alt')
        # Dictionary with mapping for camera names and id's
        src = image_obj.get_attribute('src')
        camera_names = {'62201': 'B018', '59760': 'B016', '59681': 'B013', '59758': 'B015', '60095': 'B014',
                        '60272': 'B017'}
        camera_name = camera_names[camera_id]
        # Download image
        path = f'{image_path}{camera_name}_{date_stamp.replace("/", "-")}_{picture_id}.png'
        urllib.request.urlretrieve(src, path)

        # add all information to dataframe
        df.loc[len(df.index)] = [path, camera_name, time_stamp, date_stamp, temp_stamp, moon_stamp, camera_id, alt,
                                 picture_id, src]
        i += 1
    driver.close()
    return df
'''
