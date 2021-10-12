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

import cougarvision_utils.cropping as crop_util

from collections import deque
import cougarvision_utils
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
print('Loaded classifier model in {}'.format(humanfriendly.format_timespan(elapsed)))

        
# Set Confidence Threshold
conf = 0.5


# Set Stream Path
stream_path = str(sys.argv[1])

STACK_SIZE = 300
MAX_FRAME = 300

frame_size = (1280,720)

empty_c = 0
def receive_frame():
    
    ret,frame = cap.read()
    frame = cv2.resize(frame,frame_size,fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
    stack.append(frame)
    i = 0
    while(cap.isOpened()):
        if not writer_has_control.locked():
            ret,frame = cap.read()
            if not ret:
                print(f"False read on frame {i}")
                # Stopping condition to end the spam if continuous errors
                if empty_c >= 500:
                    break
                empty_c += 1
                continue
            else:
                empty_c=0
                if ( i % 50 == 0):
                    print(f"Succesfully read frame {i}")
                frame = cv2.resize(frame,frame_size,fx=0,fy=0, interpolation = cv2.INTER_CUBIC)

                stack.append(frame)
                while len(stack) > STACK_SIZE :
                    stack.popleft()
            i += 1
        

def write_video(buffer):
    # Acquires Lock - blocks receive frame
    writer_has_control.acquire()
    print("Writing Video Now, Lock acquired")
    
    global frame_countdown

    # Construct Video Writer
    writer = cv2.VideoWriter('detected_videos/{0}.avi'.format(uuid.uuid1()), 
                    cv2.VideoWriter_fourcc(*'XVID'),
                    20, frame_size)

    # Write previous 100 frames from the deque
    while(len(buffer) > 0):
        writer.write(buffer.popleft())

    # Set Ret = True to enter while loop
    ret = True
    # Write each next valid frame 
    while(ret and frame_countdown > 0):

        ret,frame = cap.read()
        if(frame is None):
            print("Found empty frame")
        else:
            frame = cv2.resize(frame,frame_size,fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
            stack.append(frame)
            writer.write(frame)
            frame_countdown += -1
            while len(stack) > STACK_SIZE :
                stack.popleft()

    # Release video writer and writer lock
    writer.release()
    writer_has_control.release()




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

                print(result)
                # Take crop
                flag = 0 
                for detection in result['detections']:
                    img = Image.fromarray(frame)
                    bbox = detection['bbox']
                    crop = crop_util.crop(img, bbox)

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

                # Write Thread Management

                # Continue current write thread
                global frame_countdown
                if flag and writer_has_control.locked():
                    frame_countdown = MAX_FRAME
                    
                # Start new write thread 
                elif flag and not writer_has_control.locked():
                    frame_countdown = MAX_FRAME
                    buffer = stack
                    stack.clear()
                    write_thread = threading.Thread(target=write_video,args = [buffer])
                    write_thread.start()


            
if __name__ == '__main__':

    frame_countdown = 0
    writer_has_control = threading.Lock()

    print("Starting Video Stream")
    cap = cv2.VideoCapture(stream_path)

    receive_thread = threading.Thread(target=receive_frame)
    process_thread = threading.Thread(target=process_frame)

    receive_thread.start()
    process_thread.start()

