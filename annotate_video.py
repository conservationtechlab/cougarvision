import sys

import torch 
from torchvision import transforms
import cv2 
import yaml
import numpy as np
from PIL import Image
import json
import uuid

import time 
import humanfriendly

import cougarvision_utils.cropping as crop_util

# Adds CameraTraps to Sys path, import specific utilities
with open("config/cameratraps.yml", 'r') as stream:
    camera_traps_config = yaml.safe_load(stream)
    sys.path.append(camera_traps_config['camera_traps_path'])
from detection.run_tf_detector import TFDetector
import visualization.visualization_utils as viz_utils

# Load Configuration Settings from YML file
with open("config/annotate_video.yml", 'r') as stream:
    config = yaml.safe_load(stream)

# Load in classes, labels and color mapping
classes = open('labels/megadetector.names').read().strip().split('\n')
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')
image_net_labels = json.load(open("labels/labels_map.txt"))
image_net_labels = [image_net_labels[str(i)] for i in range(1000)]
# Loads in Label Category Mappings
with open("labels/label_categories.txt") as label_category:
    labels = json.load(label_category)

# Model Setup
# Detector Model
start_time = time.time()
tf_detector = TFDetector(config['detector_model'])
print(f'Loaded detector model in {humanfriendly.format_timespan(time.time() - start_time)}')

# Classifier Model
start_time = time.time()
model = torch.jit.load(config['classifier_model'])
model.eval()
print(f'Loaded classifier model in {humanfriendly.format_timespan(time.time() - start_time)}')

        
# Set confidence threshold
conf = config['confidence']

# Input Video
input_video = config['input_video']
cap = cv2.VideoCapture(input_video)

# Rendered Video Setup
frameSize = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 25.0 # or 30.0 for a better quality stream
video = cv2.VideoWriter(
        'annotated_{0}.avi'.format(input_video), 
        fourcc, 
        20.0, 
        frameSize )

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        # Run Megadetector
        t0 = time.time()
        result = tf_detector.generate_detections_one_image(
            Image.fromarray(frame),
            '0',
            conf
        )
        print(f'forward propagation time={(time.time())-t0}')
        # Take crop
        for detection in result['detections']:
            img = Image.fromarray(frame)
            bbox = detection['bbox']
            crop = crop_util.crop(img,bbox)

            # Run Classifier
            tfms = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), 
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
            crop = tfms(crop).unsqueeze(0)

            # Perform Inference
            t0 = time.time()
            with torch.no_grad():
                logits = model(crop)
            preds = torch.topk(logits, k=5).indices.squeeze(0).tolist()
            print(f'time to perform classification={(time.time())-t0}')
            print('-----')
            
            print(preds)

            label = image_net_labels[preds[0]].split(",")[0]
            prob = torch.softmax(logits, dim=1)[0, preds[0]].item()
            image = Image.fromarray(frame)
            if str(preds[0]) in labels['lizard']:
                label = 'lizard'
            viz_utils.draw_bounding_box_on_image(image,
                               bbox[1], bbox[0], bbox[1] + bbox[3], bbox[0] + bbox[2],
                               clss=preds[0],
                               thickness=4,
                               expansion=0,
                               display_str_list=['{:<75} ({:.2f}%)'.format(label, prob*100)],
                               use_normalized_coordinates=True,
                               label_font_size=16)
            frame = np.asarray(image)
        video.write(frame) 
            
    else:
        break

video.release()
cap.release()


