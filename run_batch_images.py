import yaml
import sys
import json
import os

from PIL import Image
import torch
from torchvision import transforms

# Adds CameraTraps to Sys path, import specific utilities
with open("config/cameratraps.yml", 'r') as stream:
    camera_traps_config = yaml.safe_load(stream)
    sys.path.append(camera_traps_config['camera_traps_path'])
# Load Configuration Settings from YML file
with open("config/run_batch_images.yml", 'r') as stream:
    config = yaml.safe_load(stream)

from detection.run_tf_detector_batch import main as detect_batch
from classification.crop_detections import main as crop_detections
from visualization.visualize_detector_output import visualize_detector_output
        
detector_model = config['detector_model']
classifier_model = config['classifier_model']

# single image, directory of images, or json of images
input_file = config['input_file']
# set output of detector/input to crop 
output_json = config['output_json']
# set directory of cropped images
crop_dir = config['crop_dir']

# set confidence
conf = config['confidence']

# Set Command-line arguments manually
sys.argv = f"x {detector_model} {input_file} {output_json}".split()
print(sys.argv)
detect_batch()

print(output_json)
crop_detections(output_json,crop_dir,"./",None,None,False,True,False,0.5,1,"logs/")


with open(output_json) as file:
  detections = json.load(file)

# Load Labels
labels_map = json.load(open("labels_map.txt"))
detections["detection_categories"] = labels_map
labels_map = [labels_map[str(i)] for i in range(1000)]

file_suffix_1 = "___crop0"
file_suffix_2 = "_mdvunknown.jpg"


# Classify with Selected Model
model = torch.jit.load(classifier_model)
model.eval()

for file in detections['images']:
  count = 0
  file_detections = file['detections']
  for detection in file_detections:
    if detection["conf"] <= conf: 

      print("Count is: " + str(count))
      file_detections.pop(count)
      print("Here")
      continue
    try:
      img = Image.open(crop_dir + '/' + str(file['file']) + file_suffix_1 + str(count) + file_suffix_2)
      
      # Preprocess image
      tfms = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), 
                              transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
      img = tfms(img).unsqueeze(0)

      with torch.no_grad():
          logits = model(img)
      preds = torch.topk(logits, k=5).indices.squeeze(0).tolist()

      print('-----')
      print('Filename: ' + file['file'])
      print(preds)
      id = 0
      max_prob = 0
      
      for idx in preds:
          label = labels_map[idx]
          prob = torch.softmax(logits, dim=1)[0, idx].item()
          if prob > max_prob:
            id = idx 
            max_prob = prob
          print('{:<75} ({:.2f}%)'.format(label, prob*100))

      print("ID: " + str(id) + " with prob: " + str(max_prob))

      detection['category'] = str(id)

      detection['conf'] = max_prob

      count = count + 1
    except Exception as Exception:
      file_detections.pop(count)

model_dict = { "model": os.path.splitext(os.path.basename(classifier_model))[0]}
detections.update(model_dict)

with open("classifications/" + os.path.basename(os.path.normpath(crop_dir) + '.json'), 'w') as json_file:
  json.dump(detections, json_file)

visualize_detector_output(
    detector_output_path="classifications/crops.json",
    out_dir="rendered_images/",
    confidence=conf,
    images_dir="./")
