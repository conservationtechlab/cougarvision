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
sys.path.append('/home/jaredm/Conservation_Fellowship/CameraTraps')
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
stream_path = 'rtsp://admin:NyalaChow22@192.168.1.64:8080'

def post_process(img, result, conf):
    H, W = img.shape[:2]

    boxes = []
    confidences = []
    classIDs = []

    for detection in result['detections']:
        classID = str(int(detection['category']) - 1)
        confidence = detection['conf']
        
        if confidence > conf:
            print(str(classes[int(classID)]) + ": " + str(confidence))
            x, y, w, h = detection['bbox'] * np.array([W, H, W, H])
            p0 = int(x - w//2), int(y - h//2)
            p1 = int(x + w//2), int(y + h//2)
            boxes.append([int(x), int(y), int(w), int(h)])
            confidences.append(float(confidence))
            classIDs.append(classID)
            # cv.rectangle(img, p0, p1, WHITE, 1)


    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf, conf-0.1)
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in colors[int(classIDs[i])]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(classes[int(classIDs[i])], confidences[i])
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def receive_frame():
    print("Starting Video Stream")
    cap = cv2.VideoCapture(stream_path)
    ret,frame = cap.read()
    stack.append(frame)

    while(ret):
        ret,frame = cap.read()

        stack.append(frame)

def process_frame():
    while True:
        if len(stack) > 0:
            frame = stack.pop()
            if frame is None:
                pass
            else:
                print("Processing Frame Now")
                stack.clear()
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


                    if str(preds[0]) in labels['lizard']:
                        cv2.imwrite("patio_lizards/"+str(uuid.uuid1())+".jpg", frame)

                ## Show annotated frame
                # print("Showing frame now")
                
                # cv2.imshow("window",frame)
                # cv2.waitKey(20)
                # cv2.destroyAllWindows
            



if __name__ == '__main__':
    

    p1 = threading.Thread(target=receive_frame)
    p2 = threading.Thread(target=process_frame)
    p1.start()
    p2.start()

