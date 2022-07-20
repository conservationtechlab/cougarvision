import datetime
import time
import urllib.request

import pandas as pd
import yaml
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.core.utils import ChromeType

options = webdriver.ChromeOptions()
options.add_argument("--enable-javascript")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--no-sandbox")
options.add_argument("--headless")
options.add_argument("--window-size=1920,1080")


def fetch_images(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
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
    image_num = len(image_urls)
    df = pd.DataFrame(
        columns=['file', 'camera_name', 'time', 'date', 'temperature', 'moon', 'camera_id', 'alt', 'image_id', 'src'])
    skip = True
    first = True
    i = 0
    images_found = 0
    while i < image_num:
        image_obj = image_urls[i]
        picture_id = image_obj.get_attribute('data-photo-id')
        if first:
            first = False
            print("Looking for new images since last id: " + str(last_id))
            # If no new images since last run exit
            if last_id == picture_id:
                print("No new images since last run")
                driver.close()
                return df
            # Write the first image ID to file
            if skip:
                with open(config, 'w') as f:
                    config['last_id'] = int(picture_id)
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
                    image_obj = image_urls[-1]
                    picture_id = image_obj.get_attribute('data-photo-id')
                    if int(picture_id) < int(last_id):
                        break
                images_found += 1
                last_height = new_height
            # Go to the top of the page
            actions.send_keys(Keys.HOME)
            actions.perform()
            time.sleep(2)
            # load new number of images
            image_urls = driver.find_elements(By.CLASS_NAME, 'css-1vh28r')
            image_num = images_found
            print("Number of images found: " + str(image_num))
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
