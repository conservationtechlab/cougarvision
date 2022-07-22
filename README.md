# cougarvision 1.1.0

Tools to automatically analyze images and videos from telemetering field cameras and to take responsive action. Core
mechanism is combining [Megadetector](https://github.com/microsoft/CameraTraps), with interchangeable imagenet compiled
classifier models.

## Streaming

Stream_detect.py takes in a stream, runs a pipeline to detect various animals. An image is sent to specified
emails/phone numbers based on config/stream_detect.yml.
If a lizard or a cougar is detected then a video is recorded for as long as that animal remains to be detected.

## Fetch and alert

fetch_and_alert.py combines two functions to retrieve images from [Strikeforce](https://www.strikeforcewireless.com)
site and email addresses. A webscrapper is used to retrieve images from Strikeforce username and password can be changed
in the config file. Emails are accessed and images attached are extracted, email addresses may also be changed in the
same config file. Once extracted these images
are ran through both detector and classification models. And alerts (sends emails/texts) if a cougar is detected.

### Making a config file

An example config file is located at config/fetch_and_alert_config_example.yml you may use this config file as a
template. To create your own config simply make a .yml file then copy and paste the template and fill in the appropriate
fields.

```yaml
# Fetch_and_alert
home_dir:
detector_model: Path/To/MegaDetector/Model
classifier_model: Path/To/Classifier/Model
checkpoint_frequency: -1
checkpoint_path: Path/To/Checkpoint/Directory
csv_path: Path/To/SaveORLoad/CSV/File
classes: Path/To/Classifier/Classes
# Email Settings
username: Username
password: Password
from_emails: [ "Email1", "Email2" ]
to_emails: [ "Email1", "Email2" ]
confidence: 0.5
threads: 8
# Web Scraping
site: https://www.strikeforcewireless.com/login?redirect=%2Fphotos
site2: https://www.strikeforcewireless.com/photos
username_scraper: Username
password_scraper: Password
image_path: Path/To/Image/Directory
log_path: Path/To/Save/Log/File
last_id: LastIDNumber
camera_traps_path: Path/To/CameraTraps/Directory
```

- `home_dir` Should contain the path to this repositories root it will be prepended to the other paths in the config.
- `checkpoint_frequency` Determines the frequency of which to create a checkpoint for Megadetector by the number of
  images processed set to -1 to not create any checkpoints
- `username` `password` These email settings will be used to receive image attachments and send alert messages use a 
  gmail account with an App password
- `from_emails` Unread messages sent from these addresses will be searched for image attachments
- `to_emails` Alert messages will be sent to these email addresses
- `last_id` Keeps track of the latest image extracted by the web scrapper leave blank to run all images on the 
  Strikeforce site

### Running fetch_and_alert.py
Once you have setup your Conda environment following the steps [here](#setting-up-conda-environment) make sure you have
a config file ready to be used following the process previously mentioned. The fetch_and_alert.py script accepts a
config path as an argument. Simply type `python3 fetch_and_alert.py path/to/config.yml` into your console at the root
of this repository.

## Processing Batch Images
Run_batch_images.py can take in an input folder and apply the detector/classifier pipeline to render annotated images
along with an output.json file in directory classifications. The output.json contains the final classifications on the
cropped images and can be used to compare against a ground truth set. There is also an intermediate output.json which
holds the crop detections and is used by the script to crop images, this one can be moved by configuration in the yml
file.

## Setting up Conda Environment

[Instructions to install conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

The file **cougarvision_env.yml** describes the python version and various dependencies with specific version numbers.
To activate the environment

```
conda env create -f cougarvision_env.yml

conda activate cougarvision_env

conda env list

```

The first line creates the environment from the specifications file which only needs to be done once.

The second line activates the environment which may need to be done each time you restart the terminal.

The third line is to test the environment installation where it will list all dependencies for that environment

## Dependencies

These scripts rely on the [CameraTraps Repo](https://github.com/microsoft/CameraTraps). Download this repo and place it
anywhere. Add that path to config/cameratraps.yml. The CameraTraps repo will then be added to the python sys environment
variables upon loading an individual script.

## Models

Detection and inference models can be place anywhere. In each script's yml file (under the config directory) is a field
where the path variables for each model can be specified.


