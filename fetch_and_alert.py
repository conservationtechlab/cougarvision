# Import local utilities
from cougarvision_utils.alert import smtp_setup, sendAlert
from cougarvision_utils.cropping import load_to_crop
from cougarvision_utils.fetch_emails import imap_setup, fetch_emails, extractAttachments

import json
import os
import schedule
import time
import json
import sys
import warnings
from tqdm import tqdm
from PIL import Image
from datetime import datetime
from concurrent import futures
from io import BytesIO

import torch
from torchvision import transforms



threads = 1

import tensorflow as tf

# Adds CameraTraps to Sys path, import specific utilities
sys.path.append('../CameraTraps')
from detection.run_tf_detector import ImagePathUtils, TFDetector
from detection.run_tf_detector_batch import load_and_run_detector_batch
import visualization.visualization_utils as viz_utils


# Numpy FutureWarnings from tensorflow import
warnings.filterwarnings('ignore', category=FutureWarning)

# Load sensitive/configuration information from local file
with open('config/email_config.txt') as file:
  keys = json.load(file)

# Set Email Variables for fetching
username = keys['username']
password = keys['password']
from_email = username
to_emails = keys['to_emails']
print(to_emails)
host = 'imap.gmail.com'



# Model Variables
detector_model = 'detector_models/md_v4.1.0.pb'
classifier_model = 'classifier_models/ig_resnext101_32x8d.pt'
detector_version = '4.1.0'
confidence_threshold=.5



# Load Labels for classifier
labels_map = json.load(open('labels/labels_map.txt'))
labels_map = [labels_map[str(i)] for i in range(1000)]


def runDetectionClassification(images, model_file):

    
    # Runs detection model
    results = load_and_run_detector_batch(model_file, images, checkpoint_path=None,
                                confidence_threshold=.5, checkpoint_frequency=-1,
                                results=None, n_cores=threads)
    detection_json = {
    'images': results,
    'detection_categories': TFDetector.DEFAULT_DETECTOR_LABEL_MAP,
    'info': {
        'detection_completion_time': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
        'format_version': '1.0'
        }
    }
    detections = {img['file']: img for img in detection_json['images']}
    return detections

def cropDetections(detections):
    crop_path_template = {
        True: '{img_path}___crop{n:>02d}.jpg',
        False: '{img_path}___crop{n:>02d}_' + f'mdv{detector_version}.jpg'
    }
    pool = futures.ThreadPoolExecutor(max_workers=threads)
    future_to_img_path = {}

    images_failed_download = []

    for img_path in tqdm(detections.keys()):
        # we already did all error checking above, so we don't do any here
        info_dict = detections[img_path]
        bbox_dicts = info_dict['detections']

        # get the image, either from disk or from Blob Storage
        future = pool.submit(
            load_to_crop, img_path, bbox_dicts = bbox_dicts,
            confidence_threshold = confidence_threshold)
        future_to_img_path[future] = img_path

    total = len(future_to_img_path)
    total_downloads = 0
    total_new_crops = 0
    print(f'Reading/downloading {total} images and cropping...')
    image_crops = []

    for future in tqdm(futures.as_completed(future_to_img_path), total=total):
        img_path = future_to_img_path[future]
        try:
            did_download, num_new_crops, crop_result = future.result()
            image_crops.append([crop_result, img_path])
            total_downloads += did_download
            total_new_crops += num_new_crops
        except Exception as e:  # pylint: disable=broad-except
            exception_type = type(e).__name__
            tqdm.write(f'{img_path} - generated {exception_type}: {e}')
            images_failed_download.append(img_path)

    print(f'Downloaded {total_downloads} images.')
    print(f'Made {total_new_crops} new crops.')

    return image_crops

def classify(image_crops, MODEL_PATH):

    classifications = []
    for image in image_crops:

        crop = image[0][0][0]
        # Preprocess Image
        tfms = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), 
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
        crop = tfms(crop).unsqueeze(0)

        # Load Model
        model = torch.jit.load(MODEL_PATH)
        model.eval()

        # Perform Inference
        with torch.no_grad():
            logits = model(crop)
        preds = torch.topk(logits, k=5).indices.squeeze(0).tolist()

        print('-----')
        
        # Print result
        for idx in preds:
            label = labels_map[idx]
            prob = torch.softmax(logits, dim=1)[0, idx].item()
            print('{:<75} ({:.2f}%)'.format(label, prob*100))
            classifications.append([idx,label,prob,image])

    
    return classifications

def check_humans_vehicles(detections,smtp_server):
    # Check Detections for Vehicles and Humans
    for img_path in tqdm(detections.keys()):
        # we already did all error checking above, so we don't do any here
        info_dict = detections[img_path]
    for detection in info_dict['detections']:
        if detection['category'] == '2':
            image = Image.open(info_dict['file'])
            box = detection['bbox']
            viz_utils.draw_bounding_box_on_image(image,
                               box[1], box[0], box[1] + box[3], box[0] + box[2],
                               clss=3,
                               thickness=4,
                               expansion=0,
                               display_str_list=['Human: {:.2f}%'.format(detection['conf']*100)],
                               use_normalized_coordinates=True,
                               label_font_size=16)
            imageBytes = BytesIO()
            image.save(imageBytes,format=image.format)
            sendAlert('Human', detection['conf'],imageBytes,smtp_server, from_email, to_emails)
        elif detection['category'] == '3':
            image = Image.open(info_dict['file'])

            box = detection['bbox']
            viz_utils.draw_bounding_box_on_image(image,
                               box[1], box[0], box[1] + box[3], box[0] + box[2],
                               clss=3,
                               thickness=4,
                               expansion=0,
                               display_str_list=['Vehicle: {:.2f}%'.format(detection['conf']*100)],
                               use_normalized_coordinates=True,
                               label_font_size=16)
            imageBytes = BytesIO()
            image.save(imageBytes,format=image.format)
            sendAlert('Vehicle', detection['conf'],imageBytes,smtp_server, from_email, to_emails)

def check_cougars(classifications,smtp_server):
    # Check Classifications for Vehicles and Humans
    for classification in classifications:
        conf = classification[2]
        if classification[0] == 287 or classification[0] == 291 and conf >= 0.01:
            image = classification[3][1]
            image = Image.open(image)
            box = classification[3][0][0][1]
            viz_utils.draw_bounding_box_on_image(image,
                               box[1], box[0], box[1] + box[3], box[0] + box[2],
                               clss=3,
                               thickness=4,
                               expansion=0,
                               display_str_list=['Cougar: {:.2f}%'.format(conf*100)],
                               use_normalized_coordinates=True,
                               label_font_size=16)
            imageBytes = BytesIO()
            image.save(imageBytes,format=image.format)
            sendAlert('Cougar', conf,imageBytes,smtp_server, from_email, to_emails)

def main():


    smtp_server = smtp_setup(username,password,host)

    # Gets a list of attachments from unread emails from bigfoot camera
    mail = imap_setup(host, username, password)
    images = extractAttachments(fetch_emails(mail),mail)
    if len(images) > 0:

        # Run Detector
        detections = runDetectionClassification(images, detector_model)
        # Check if detector found vehicles or humans
        check_humans_vehicles(detections,smtp_server)
        
        # Crop Detections
        crops = cropDetections(detections)
        
        # Run Classifier on cropped images and pass results into check cougars
        check_cougars(classify(crops, classifier_model),smtp_server)

        

main()

schedule.every(10).minutes.do(main)

while True:
    schedule.run_pending()
    time.sleep(1)
