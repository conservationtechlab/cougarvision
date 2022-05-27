from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.utils import ChromeType

import time
import datetime
import requests
from io import BytesIO
from PIL import Image
from PIL.ExifTags import TAGS

import re

options = webdriver.ChromeOptions()
options.add_argument("--enable-javascript")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--no-sandbox")
options.add_argument("--headless")

def readID():
    with open('checkpoint_id.txt', 'r') as f:
        checkpoint_id = f.read()
    return checkpoint_id

def writeID(picture_id):
    with open('checkpoint_id.txt', 'w+') as f:
        f.write(str(picture_id))


def nextPage(driver):
    next_page = driver.find_element_by_xpath("//li[@title='Next Page']")
    actions = ActionChains(driver)
    actions.click(next_page)
    actions.perform()
    

def fetch_images():
    driver = webdriver.Chrome(ChromeDriverManager(chrome_type=ChromeType.CHROMIUM).install(),options=options)
    actions = ActionChains(driver)
    driver.get("https://www.strikeforcewireless.com/login?redirect=%2Fphotos")
    name= driver.find_element_by_id("username")
    name.clear()
    name.send_keys("bioreserve.cam@gmail.com")
    pw = driver.find_element_by_id("password")
    pw.send_keys("Cougar2022!")
    pw.send_keys(Keys.RETURN)

    #wait to login
    time.sleep(2)

    #navigate to photos page
    driver.get("https://www.strikeforcewireless.com/photos")

    #wait for javascript to load
    time.sleep(2)

    images = []

    
    first = True
    first_id = None

    while(True):
        # driver.refresh()
        time.sleep(1)
        dates = driver.find_elements_by_class_name("css-1exq1ob")
        for date in dates:
            date_text = date.find_element_by_xpath("div[@class='css-1b8bd04']").text

            for image_obj in date.find_elements_by_tag_name('img'):
                actions = ActionChains(driver)
                actions.move_to_element(image_obj)
                actions.perform()
                time_text = re.split("-|\n",date.find_element_by_xpath("//div[@class='css-9akxad']").text)[1]
                time_string = date_text + " " + time_text


                print(time_string)
                timestamp = datetime.datetime.strptime(time_string, "%b %d, %Y %I:%M%p")
                print(timestamp)
 
                url = image_obj.get_attribute('src')
                camera_name = image_obj.get_attribute('data-camera-id')
                picture_id = image_obj.get_attribute('data-photo-id')
                # print(f"url:{url}")
                # print(f"camera_name:{camera_name}")
                # print(f"picture_id:{picture_id}")
                
                if first:
                    first = False
                    first_id = picture_id
                if picture_id == readID():
                    driver.close()
                    print("Detected checkpoint -> Stopping")
                    writeID(first_id)

                    return images


                                        

                r = requests.get(url)
                img = BytesIO(r.content)
                new_img = Image.open(img)


                images.append((img,camera_name,timestamp,picture_id))
                print("-----")
        nextPage(driver)
    driver.close()
    return images

if __name__ == "__main__":
    fetch_images()