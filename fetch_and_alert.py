# Import local utilities
from cougarvision_utils.alert import smtp_setup, sendAlert
from cougarvision_utils.cropping import load_to_crop, crop
from cougarvision_utils.fetch_emails import imap_setup, fetch_emails, extractAttachments
from cougarvision_utils.log_utils import log_classification,save_image
from cougarvision_utils.model_conversion import convert_and_load, convert_to_keras
import schedule
import time
import json
import sys
import warnings
import yaml
from tqdm import tqdm
from PIL import Image
from datetime import datetime
from concurrent import futures
from io import BytesIO
import uuid

import torch
from torchvision import transforms
import numpy as np
import humanfriendly


# Adds CameraTraps to Sys path, import specific utilities
with open("config/cameratraps.yml", 'r') as stream:
    camera_traps_config = yaml.safe_load(stream)
    sys.path.append(camera_traps_config['camera_traps_path'])

# noinspection PyUnresolvedReferences
from detection.run_tf_detector import TFDetector
from detection.run_tf_detector_batch import load_and_run_detector_batch
import visualization.visualization_utils as viz_utils


# Numpy FutureWarnings from tensorflow import
warnings.filterwarnings('ignore', category=FutureWarning)

# Load Configuration Settings from YML file
with open("config/fetch_and_alert.yml", 'r') as stream:
    config = yaml.safe_load(stream)
# Comment out if On GPU
force_cpu = True
if force_cpu:
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
threads = 1

import tensorflow as tf
from tensorflow import keras

# Set Email Variables for fetching
username = config['username']
password = config['password']
from_emails = config['from_emails']
to_emails = config['to_emails']
host = 'imap.gmail.com'


# Model Variables
detector_model = config['detector_model']
classifier_model = config['classifier_model']

# Model Setup
# Detector Model
start_time = time.time()
tf_detector = TFDetector(config['detector_model'])
print(f'Loaded detector model in {humanfriendly.format_timespan(time.time() - start_time)}')

# Classifier Model
start_time = time.time()

model = convert_to_keras(config['classifier_model'])
print(f'Loaded classifier model in {humanfriendly.format_timespan(time.time() - start_time)}')

# Set Confidence 
confidence_threshold = config['confidence']

# Set threads for load_and_crop
threads = config['threads']

# Load Labels for classifier
labels_map = json.load(open('labels/labels_map.txt'))
labels_map = [labels_map[str(i)] for i in range(1000)]

with open('/home/edgar/mnt/machinelearning/Models/Southwest/classes.txt', 'r') as f:
    data = f.read().splitlines()
southwest_labels = np.asarray(data)

timestamp = datetime.now()

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
    pool = futures.ThreadPoolExecutor(max_workers=threads)
    future_to_img_path = {}

    images_failed_download = []

    for img_path in tqdm(detections.keys()):
        # we already did all error checking above, so we don't do any here
        info_dict = detections[img_path]
        bbox_dicts = info_dict['detections']

        future = pool.submit(
            load_to_crop, img_path, bbox_dicts = bbox_dicts,
            confidence_threshold = confidence_threshold)
        future_to_img_path[future] = img_path

    total = len(future_to_img_path)

    print(f'Reading/downloading {total} images and cropping...')
    image_crops = []

    for future in tqdm(futures.as_completed(future_to_img_path), total=total):
        img_path = future_to_img_path[future]
        try:
            did_download, num_new_crops, crop_result = future.result()
            image_crops.append([crop_result, img_path])

        except Exception as e:  # pylint: disable=broad-except
            exception_type = type(e).__name__
            tqdm.write(f'{img_path} - generated {exception_type}: {e}')
            images_failed_download.append(img_path)

    return image_crops

def classify(image_crops, MODEL_PATH):

    classifications = []
    for image in image_crops:

        crop = image[0][0][0]
        # Preprocess Image
        tfms = transforms.Compose([transforms.Resize(456), transforms.CenterCrop(456), 
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
        crop = tfms(crop).unsqueeze(0)

        # Load Model
        # model = torch.jit.load(model_path)
        # model.eval()
        #model = convert_to_keras(MODEL_PATH)

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
        # Check if detection is human
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
            sendAlert('Human', detection['conf'],imageBytes,smtp_server, username, to_emails)
        # Check if detection is vehicle
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
            sendAlert('Vehicle', detection['conf'],imageBytes,smtp_server, username, to_emails)

def check_cougars(classifications,smtp_server):
    # Check Classifications for Cougars
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
            sendAlert('Cougar', conf,imageBytes,smtp_server, username, to_emails)

def process_image(image_data,smtp_server):
    # image_data = (image,camera_name,time_stamp,picture_id)
    frame = image_data[0]
    camera_name = image_data[1]
    timestamp = image_data[2]
    picture_id = image_data[3]
    image = Image.open(frame)

    t0 = time.time()
    result = tf_detector.generate_detections_one_image(
        image,
        '0',
        confidence_threshold
    )
    print(f'forward propagation time={(time.time())-t0}')

    print(result)
    foundDetection = False
    for detection in result['detections']:
        foundDetection = True
        # Crops each bbox from detector
        img = Image.open(frame)
        bbox = detection['bbox']
        crop_result = crop(img, bbox)
        

        # Preprocessing image for classifier
        tfms = transforms.Compose([transforms.Resize(456), transforms.CenterCrop(456),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
        crop_result = tfms(crop_result).unsqueeze(3)
        crop_result = tf.image.resize(crop_result,[456,456])
        
        crop_array = tf.keras.preprocessing.image.img_to_array(crop_result)
        test_img = Image.fromarray((crop_result.numpy()).astype(np.uint8)).convert('RGB')
        test_img.save("test.jpg")
        crop_array = tf.expand_dims(crop_array, 0)
        # Perform Inference
        t0 = time.time()
        # with torch.no_grad():
            # logits = model(crop_result)
        # preds = torch.topk(logits, k=5).indices.squeeze(0).tolist()
        preds = model(np.asarray(crop_array))

        print(f'time to perform classification={(time.time())-t0}')
        print('-----')
        
        print(f'preds{preds}')
        print(type(preds))
        idx = preds.numpy().argmax(axis=-1)[0]
        print(f'argmax id {idx}')
        # prob = torch.softmax(logits, dim=1)[0, preds[0]].item()
        # label = labels_map[preds[0]].split(',')[0]
        prob = preds.numpy()[0][idx]
        label = southwest_labels[idx]
        print(f'prob {prob}')
        print(f'label {label}')
        viz_utils.draw_bounding_box_on_image(img,
                bbox[1], bbox[0], bbox[1] + bbox[3], bbox[0] + bbox[2],
                clss=idx,
                thickness=4,
                expansion=0,
                display_str_list=f'{label} {prob*100}',
                use_normalized_coordinates=True,
                label_font_size=25)

        img_file = f"{camera_name}_{timestamp.strftime('%m-%d-%Y_%H%M%S')}.jpg"
        print(img_file)
        log_classification(datetime.now(),camera_name,picture_id,label,prob)
        

        # send alert based on labels
        imageBytes = BytesIO()
        img.save(imageBytes,format=img.format)
        if label == "cougar":
            pass
            # sendAlert(label, prob,imageBytes,smtp_server, username, to_emails)
    if foundDetection:
        save_image(image,img_file)

def main():


    smtp_server = smtp_setup(username,password,host)

    # Gets a list of attachments from unread emails from bigfoot camera
    mail = imap_setup(host, username, password)
    global timestamp
    images = extractAttachments(fetch_emails(mail,from_emails,timestamp),mail)
    print(images)

    # Reset Timestamp
    timestamp = datetime.now()
    print(timestamp)
    
    if len(images) > 0:

        # Run Detector
        detections = runDetectionClassification(images, detector_model)

        print(detections)
        # Check if detector found vehicles or humans
        check_humans_vehicles(detections,smtp_server)

        for image in images:
            process_image(image,smtp_server)
        
        # Crop Detections
        crops = cropDetections(detections)
        classify(crops,classifier_model)

        # Run Classifier on cropped images and pass results into check cougars
        check_cougars(classify(crops, classifier_model),smtp_server)

        
main()

schedule.every(10).minutes.do(main)

while True:
    schedule.run_pending()
    time.sleep(1)
