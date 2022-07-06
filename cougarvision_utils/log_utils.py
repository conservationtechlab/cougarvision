import csv 
import os.path
import yaml
with open("config/fetch_and_alert.yml", 'r') as stream:
    config = yaml.safe_load(stream)
    LOG_DIR_PATH = config['log_path']
    CSV_PATH = config['csv_path']


# csv header format
# datetime,camera_id, image_file_name, top 1 classification, confidence

# Save classification
def log_classification(time,camera_id,img_file,classification,confidence):
    csv_file = LOG_DIR_PATH + CSV_PATH
    with open(csv_file, 'a') as csvfile:
        csv_writer = csv.writer(csvfile,delimiter=',')
        if os.stat(csv_file).st_size == 0:
            csv_writer.writerow(['datetime', 'camera_id','file_name','class','conf',])
        csv_writer.writerow([time, camera_id,img_file,classification,confidence])
# Saves image
def save_image(image,img_file):
    image.save(LOG_DIR_PATH + img_file)

