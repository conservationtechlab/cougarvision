import sys
import threading
import time 
import humanfriendly
import json 
import uuid
import warnings

import yaml
import numpy as np
import torch 
from torchvision import transforms
import cv2 
from PIL import Image
import cougarvision_utils.alert as alert_util
import cougarvision_utils.cropping as crop_util
from io import BytesIO

# Adds CameraTraps to Sys path, import specific utilities
with open("config/cameratraps.yml", 'r') as stream:
    camera_traps_config = yaml.safe_load(stream)
    sys.path.append(camera_traps_config['camera_traps_path'])

from detection.run_tf_detector import TFDetector
import visualization.visualization_utils as viz_utils

from collections import deque

frames_deque = deque()

# Numpy FutureWarnings from tensorflow import
warnings.filterwarnings('ignore', category=FutureWarning)

# Load Configuration Settings from YML file
with open("config/stream_detect.yml", 'r') as stream:
    config = yaml.safe_load(stream)

# Loads in Label Category Mappings
with open("labels/label_categories.txt") as label_category:
    labels_category = json.load(label_category)

# Loads in Label Category Mappings
labels_map = json.load(open('labels/labels_map.txt'))
labels_map = [labels_map[str(i)] for i in range(1000)]

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

        
# Set Confidence Threshold
conf = config['confidence']

# Set Stream Path
stream_path = config['stream_path']

# Set Size Parameters
DEQUE_SIZE = config['DEQUE_SIZE']
MAX_FRAME = config['MAX_FRAME']
frame_size = tuple(config['frame_size'])

# Set Video Output Path
video_output_path = config['video_output_path']


# Set Email Variables for fetching
username = config['username']
password = config['password']
from_email = username
to_emails = config['to_emails']
host = 'imap.gmail.com'

def receive_frame():
    # Capture first frame
    ret,frame = cap.read()
    frame = cv2.resize(frame,frame_size,fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
    frames_deque.append(frame)

    # Reads frames while capture is online and write thread doesn't have control
    while(cap.isOpened()):
        if not writer_has_control.locked():
            ret,frame = cap.read()
            if not ret:
                print("False read on frame")
                continue
            else:
                frame = cv2.resize(frame,frame_size,fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
                frames_deque.append(frame)
                # Maintains deque size
                while len(frames_deque) > DEQUE_SIZE :
                    frames_deque.popleft()
      
def write_video():
    # Acquires Lock - blocks receive frame
    
    print("Writing Video Now, Lock acquired")
    
    

    # Construct Video Writer
    writer = cv2.VideoWriter(f'{video_output_path}/{uuid.uuid1()}.avi', 
                    cv2.VideoWriter_fourcc(*'XVID'),
                    20, frame_size)

    # Write previous 100 frames from the deque
    while(len(frames_deque) > 0):
        writer.write(frames_deque.popleft())

    # Set Ret = True to enter while loop
    ret = True
    # Tells thread that frame_countdown is a global variable
    global frame_countdown
    # Write each next valid frame 
    while(ret and frame_countdown > 0):

        ret,frame = cap.read()
        if(frame is None):
            print("Found empty frame")
        else:
            frame = cv2.resize(frame,frame_size,fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
            frames_deque.append(frame)
            writer.write(frame)
            frame_countdown += -1
            while len(frames_deque) > DEQUE_SIZE :
                frames_deque.popleft()

    # Release video writer and writer lock
    writer.release()
    writer_has_control.release()

def process_frame():
    while True:
        if len(frames_deque) > 0:
            frame = frames_deque.pop()
            if frame is None:
                pass
            else:
                print("Processing Frame Now")

                # Runs Megadetector on frame
                t0 = time.time()
                result = tf_detector.generate_detections_one_image(
                    Image.fromarray(frame),
                    '0',
                    conf
                )
                print(f'forward propagation time={(time.time())-t0}')

                print(result)
                
                flag = 0 
                
                for detection in result['detections']:
                    # Crops each bbox from detector
                    img = Image.fromarray(frame)
                    bbox = detection['bbox']
                    crop = crop_util.crop(img, bbox)

                    # Preprocessing image for classifier
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

                    prob = torch.softmax(logits, dim=1)[0, preds[0]].item()
                    # All labels less than 397 are animals
                    if(preds[0] <= 397) and prob > conf:
                        label = labels_map[preds[0]].split()[0]
                        viz_utils.draw_bounding_box_on_image(img,
                               bbox[1], bbox[0], bbox[1] + bbox[3], bbox[0] + bbox[2],
                               clss=preds[0],
                               thickness=4,
                               expansion=0,
                               display_str_list=['{:<75} ({:.2f}%)'.format(label, prob*100)],
                               use_normalized_coordinates=True,
                               label_font_size=16)
                        image = np.asarray(img)
                        cv2.imwrite(f"recorded_images/{label}-{uuid.uuid1()}.jpg",image)
                        imageBytes = BytesIO()
                        image.save(imageBytes,format=image.format)
                        alert_util.sendAlert("Found something!", conf,imageBytes,
                                    alert_util.smtp_setup(username,password,host),from_email, to_emails)

                    
                    if str(preds[0]) in labels_category['lizard'] and prob > conf:
                        label = 'lizard'
                        flag = 1

                    elif str(preds[0]) in labels_category['cougar'] and prob > conf:
                        label = 'cougar'
                        flag = 1


                # Write Thread Management

                # Continue current write thread / Reset frame_countdown
                global frame_countdown
                if flag and writer_has_control.locked():
                    frame_countdown = MAX_FRAME
                    
                # Start new write thread
                elif flag and not writer_has_control.locked():
                    writer_has_control.acquire()
                    frame_countdown = MAX_FRAME
                    write_thread = threading.Thread(target=write_video)
                    write_thread.start()
            
if __name__ == '__main__':

    frame_countdown = 0
    # init writer lock
    writer_has_control = threading.Lock()

    # start video capture
    print("Starting Video Stream")
    cap = cv2.VideoCapture(stream_path)

    # Construct and start main threads
    receive_thread = threading.Thread(target=receive_frame)
    process_thread = threading.Thread(target=process_frame)

    receive_thread.start()
    process_thread.start()