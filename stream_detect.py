import logging
import sys
import tempfile
import threading
import time
import uuid
import warnings
from collections import deque
from io import BytesIO

import cv2
import humanfriendly
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from PIL import Image
from animl import TFDetector
from tensorflow import keras

import cougarvision_utils.alert as alert_util
import cougarvision_utils.cropping as crop_util

frames_deque = deque()

# Numpy FutureWarnings from tensorflow import
warnings.filterwarnings('ignore', category=FutureWarning)

# Load Configuration Settings from YML file
with open("config/stream_detect.yml", 'r') as stream:
    config = yaml.safe_load(stream)

# Loads in Label Category Mappings

# Loads in Label Category Mappings
classes = config['classes']

# Loads in alert targets and general targets
alert_targets = config['alert_targets']
general_targets = config['general_targets']
recorded_images_dir = config['image_dir']

# Detector Model
start_time = time.time()
tf_detector = TFDetector.TFDetector(config['detector_model'])
print(f'Loaded detector model in {humanfriendly.format_timespan(time.time() - start_time)}')

# Classifier Model
classifier_model = config['classifier_model']
start_time = time.time()
model = keras.models.load_model(classifier_model)

print(f'Loaded classifier model in {humanfriendly.format_timespan(time.time() - start_time)}')

# Set Confidence Threshold
conf = config['confidence']
conf_alerts = config['confidence_alerts']

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

flag = 0
alert_list = []


def receive_frame():
    # Capture first frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 0)
    frame = cv2.resize(frame, frame_size, fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
    frames_deque.append(frame)

    # Reads frames while capture is online and write thread doesn't have control
    while cap.isOpened():
        if not writer_has_control.locked():
            try:
                ret, frame = cap.read()
                if not ret:
                    print("False read on frame")
                    continue
                else:
                    frame = cv2.flip(frame, 0)
                    frame = cv2.resize(frame, frame_size, fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
                    frames_deque.append(frame)
                    # Maintains deque size
                    while len(frames_deque) > DEQUE_SIZE:
                        frames_deque.popleft()
            except Exception as e:
                logging.error(f'Exception: {e} : {time.ctime(time.time())}')


def write_video():
    while True:
        global flag
        if flag:
            # Acquires Lock - blocks receive frame
            writer_has_control.acquire()
            print("Writing Video Now, Lock acquired")

            # Construct Video Writer
            writer = cv2.VideoWriter(f'{video_output_path}/{uuid.uuid1()}.avi',
                                     cv2.VideoWriter_fourcc(*'XVID'),
                                     20, frame_size)

            # Write previous 100 frames from the deque
            while len(frames_deque) > 0:
                writer.write(frames_deque.popleft())

            # Set Ret = True to enter while loop
            ret = True
            # Tells thread that frame_countdown is a global variable
            global frame_countdown
            # Write each next valid frame 
            while ret and frame_countdown > 0:
                try:
                    ret, frame = cap.read()
                    if frame is None:
                        print("Found empty frame")
                    else:
                        frame = cv2.flip(frame, 0)
                        frame = cv2.resize(frame, frame_size, fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
                        frames_deque.append(frame)
                        writer.write(frame)
                        frame_countdown += -1
                        print(frame_countdown)
                        while len(frames_deque) > DEQUE_SIZE:
                            frames_deque.popleft()
                except Exception as e:
                    logging.error(f'Exception: {e} : {time.ctime(time.time())}')

            # Release video writer and writer lock
            flag = 0
            writer.release()
            del writer
            writer_has_control.release()


def process_frame():
    while True:
        if len(frames_deque) > 0:
            frame = frames_deque.pop()
            print(sys.getsizeof(f"Size of deque: {frames_deque}"))
            if frame is None:
                pass
            else:
                print("Processing Frame Now")

                # Runs Megadetector on frame
                t0 = time.time()
                temp = tempfile.TemporaryFile()
                result = tf_detector.generate_detections_one_image(
                    Image.fromarray(frame),
                    temp.name,
                    conf
                )
                temp.close()
                print(f'forward propagation time={(time.time()) - t0}')

                print(result)

                for detection in result['detections']:
                    # Crops each bbox from detector
                    img = Image.fromarray(frame)
                    bbox = detection['bbox']
                    crop = crop_util.crop(img, bbox)
                    # Preprocess image
                    crop = tf.image.resize(crop, [456, 456])
                    crop = np.asarray([crop])
                    # Perform Inference
                    t0 = time.time()
                    predictions = model.predict(crop)
                    print(f'time to perform classification={(time.time()) - t0}')
                    print('-----')

                    global flag
                    # parse the prediction result
                    predictionsDataframe = pd.DataFrame(predictions)
                    label = predictionsDataframe.idxmax(axis=1).to_frame(name='class').values[0][0]
                    prob = predictionsDataframe[label].values[0]
                    table = pd.read_table(classes, sep=" ", index_col=0)
                    label = table['x'].values[label]
                    # check if our detection should be saved and alerted to
                    if (label in general_targets or label in alert_targets) and prob >= conf and not flag:
                        crop_util.draw_bounding_box_on_image(img,
                                                             bbox[1], bbox[0], bbox[1] + bbox[3], bbox[0] + bbox[2],
                                                             clss=None,
                                                             thickness=4,
                                                             expansion=0,
                                                             display_str_list=[
                                                                 '{:<75} ({:.2f}%)'.format(label, prob * 100)],
                                                             use_normalized_coordinates=True,
                                                             label_font_size=16)
                        cv2.imwrite(f"{recorded_images_dir}/{label}-{uuid.uuid1()}.jpg", np.asarray(img))
                        imageBytes = BytesIO()
                        img.save(imageBytes, format="JPEG")
                        logging.info(f'Found {label} with probability of {prob}:{time.ctime(time.time())}')
                        global frame_countdown
                        print(f'Prob of top predict = {prob}')
                        if label in alert_targets and prob >= conf_alerts:
                            x = (label, prob)
                            alert_list.append(x)

                    # check for alerts
                    if len(alert_list) > 0:
                        prob = 0
                        label = None
                        for x in alert_list:
                            if x[1] > prob:
                                prob = x[1]
                                label = x[0]
                        print(f"{label} WAS FOUND")
                        alert_list.clear()
                        alert_util.sendAlert(f"Found {label}", prob, imageBytes,
                                             alert_util.smtp_setup(username, password, host), from_email, to_emails)
                        flag = 1
                        frame_countdown = MAX_FRAME
                # Write Thread Management

                # # Start new write thread
                # elif flag and not writer_has_control.locked():

                #     frame_countdown = MAX_FRAME
                #     write_thread = threading.Thread(target=write_video)
                #     write_thread.start()


if __name__ == '__main__':
    # Init log
    logging.basicConfig(filename="stream.log", level=logging.DEBUG)

    frame_countdown = 0

    # init writer lock
    writer_has_control = threading.Lock()

    # start video capture
    print("Starting Video Stream")
    cap = cv2.VideoCapture(stream_path)

    # Construct and start main threads
    receive_thread = threading.Thread(target=receive_frame)
    process_thread = threading.Thread(target=process_frame)
    write_thread = threading.Thread(target=write_video)

    receive_thread.start()
    process_thread.start()
    write_thread.start()
