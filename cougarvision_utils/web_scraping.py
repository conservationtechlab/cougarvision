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

options = webdriver.ChromeOptions()
options.add_argument("--enable-javascript")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--no-sandbox")
options.add_argument("--headless")

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

    image_urls = driver.find_elements_by_tag_name('img')
    print(len(image_urls))
    images = []
    skip = True
    first = True
    for image_obj in image_urls:
        print(image_obj)
        if skip:
            skip = False
            continue
        # Clicks on first image to bring up lightbox
        if first:
            first = False
            actions.move_to_element(image_obj)
            actions.click(image_obj)
            actions.perform()
        # Clicks arrow key to advance to next image
        else:
            button_element = driver.find_element_by_xpath("//button[@class='ril-next-button ril__navButtons ril__navButtonNext']")
            actions.move_to_element(button_element)
            actions.click(button_element)
            actions.perform()
            time.sleep(2)
        text = driver.find_element_by_xpath("//div[@class='ril-toolbar ril__toolbar']").text
        time_string = text.splitlines()[2]
        print(time_string)
        timestamp = datetime.datetime.strptime(time_string, "%b %d, %Y%I:%M%p")
        print(timestamp)
        
        url = image_obj.get_attribute('src')
        camera_name = image_obj.get_attribute('data-camera-id')
        picture_id = image_obj.get_attribute('data-photo-id')
        # print(f"url:{url}")
        # print(f"camera_name:{camera_name}")
        # print(f"picture_id:{picture_id}")

        
        r = requests.get(url)
        img = BytesIO(r.content)
        new_img = Image.open(img)
        images.append((img,camera_name,timestamp,picture_id))
        print("-----")
        time.sleep(2)

        
    driver.close()
    return images


if __name__ == "__main__":
    fetch_images()