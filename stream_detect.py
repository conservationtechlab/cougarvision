import os
import sys

import tensorflow as tf
import torch 
from torchvision import transforms
import cv2 

import numpy as np
from PIL import Image
import json 
import uuid

import time 
import humanfriendly
import threading


from collections import deque
stack = deque()


# Adds CameraTraps to Sys path, import specific utilities
sys.path.append('../CameraTraps')
from detection.run_tf_detector import ImagePathUtils, TFDetector
import visualization.visualization_utils as viz_utils


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
detector_model = ('detector_models/md_v4.1.0.pb')
start_time = time.time()
tf_detector = TFDetector(detector_model)
elapsed = time.time() - start_time
print('Loaded detector model in {}'.format(humanfriendly.format_timespan(elapsed)))

# Classifier Model
start_time = time.time()
classifier_model = 'classifier_models/ig_resnext101_32x8d.pt'
model = torch.jit.load(classifier_model)
model.eval()        
print('Loaded detector model in {}'.format(humanfriendly.format_timespan(elapsed)))

        
# Set Confidence Threshold
conf = 0.5


# Set Stream Path
stream_path = str(sys.argv[1])

frame_countdown = 0

frame_size = (1280,720)

def receive_frame():
    
    ret,frame = cap.read()
    frame = cv2.resize(frame,frame_size,fx=0,fy=0, interpolation = cv2.INTER_CUBIC)

    stack.append(frame)

    while(ret and frame_countdown == 0):
        ret,frame = cap.read()
        frame = cv2.resize(frame,frame_size,fx=0,fy=0, interpolation = cv2.INTER_CUBIC)

        stack.append(frame)
        while len(stack) > 100 :
            stack.popleft()

def write_video():
    ret = True
    writer = cv2.VideoWriter(uuid.uuid1()+'.avi', 
                    cv2.VideoWriter_fourcc(*'MJPG'),
                    10, frame_size)
    while(ret and frame_countdown > 0):
        ret,frame = cap.read()
        frame = cv2.resize(frame,frame_size,fx=0,fy=0, interpolation = cv2.INTER_CUBIC)

        stack.append(frame)
        writer.write(frame)
        frame_countdown = frame_countdown - 1
        while len(stack) > 100 :
            stack.popleft()



def process_frame():
    
    while True:
        if len(stack) > 0:
            frame = stack.pop()
            if frame is None:
                pass
            else:
                print("Processing Frame Now")
            
                t0 = time.time()
                result = tf_detector.generate_detections_one_image(
                    Image.fromarray(frame),
                    '0',
                    conf
                )
                print(f'forward propagation time={(time.time())-t0}')

                print( result)
                # Take crop
                for detection in result['detections']:
                    img = Image.fromarray(frame)
                    bbox = detection['bbox']
                    img_w, img_h = img.size
                    xmin = int(bbox[0] * img_w)
                    ymin = int(bbox[1] * img_h)
                    box_w = int(bbox[2] * img_w)
                    box_h = int(bbox[3] * img_h)
                    crop = img.crop(
                        box=[xmin,
                        ymin, 
                        xmin + box_w,
                        ymin + box_h])

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

                    flag = 0 
                    if str(preds[0]) in labels['lizard']:
                        label = 'lizard'
                        flag = 1

                    elif str(preds[0]) in labels['cougar']:
                        label = 'cougar'
                        flag = 1

                    viz_utils.draw_bounding_box_on_image(image,
                                        bbox[1], bbox[0], bbox[1] + bbox[3], bbox[0] + bbox[2],
                                        clss=preds[0],
                                        thickness=4,
                                        expansion=0,
                                        display_str_list=['{:<75} ({:.2f}%)'.format(label, prob*100)],
                                        use_normalized_coordinates=True,
                                        label_font_size=16)
                    frame = np.asarray(image)


                    if flag and write_thread.is_alive():
                        frame_countdown = 100
                    elif flag and not write_thread.is_alive():
                        frame_countdown = 100
                        write_thread.start()



                ## Show annotated frame
                # print("Showing frame now")
                
                # cv2.imshow("window",frame)
                # cv2.waitKey(20)
                # cv2.destroyAllWindows
            




print("Starting Video Stream")
cap = cv2.VideoCapture(stream_path)

receive_thread = threading.Thread(target=receive_frame)
process_thread = threading.Thread(target=process_frame)
write_thread = threading.Thread(target=write_video)
receive_thread.start()
process_thread.start()

