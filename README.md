# cougarvision 1.1.0
Tools to automatically analyze images and videos from telemetering field cameras and to take responsive action. Core mechanism is combining [Megadetector](https://github.com/microsoft/CameraTraps), with interchangeable imagenet compiled classifier models.

## Origins

CougarVision was begun as Jared Macshane's project when he was a
graduate student fellow in the San Diego Zoo Wildlife Alliance's
Conservation Technology Lab's Fellows in Conservation Technology
program. Its name is an homage both to the species for which it was
originally developed and the mascot of Cal State San Marcos where
Jared was then studying, which is also the cougar.

# Setting up Conda Environment
First, you must install conda (whatever the latest version of Anaconda is) on your machine, install instructions can be found here:

[Instructions to install conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

```
git clone https://github.com/conservationtechlab/cougarvision.git

cd cougarvision
```
The file **cougarvision_env.yml** describes the python version and various dependencies with specific version numbers. To activate the enviroment

```
conda env create -f cougarvision_env.yml

conda activate cougarvision_env

conda env list
```

The first line creates the enviroment from the specifications file which only needs to be done once. 

The second line activates the environment which may need to be done each time you restart the terminal.

The third line is to test the environment installation where it will list all dependencies for that environment.

The environment must be activated anytime you wish to run the scripts within this package. 

# Models

Detection and inference models can be placed anywhere. In each script's yml file (under the config directory) is a field where the path variables for each model can be specified. Linked are the current updated models along with their class lists for [Peru](https://sandiegozoo.box.com/s/jfw7ih8xedzsn83to91pg6gvaq1nj5bl),[Peru class list](https://sandiegozoo.box.com/s/xng8erxrvw6nz98h8xjtk8avnopfz6ev), [Southwest](https://sandiegozoo.box.com/s/x63lnaxw8hag39mczeommqy9tw4t0ht9), [Southwest class list](https://sandiegozoo.box.com/s/hn8nput5pxjc3toao57gfn4h6zo1lyng) and [Kenya](https://sandiegozoo.box.com/s/cwn5wss9gjibvf57xop2zgfmlih512lt), [Kenya class list](https://sandiegozoo.box.com/s/f5athitml7bedix0ubnccyg8npvsr6ip). The detection model that is currently integrated with fetch_and_alert.py functionality is [MDv5 using pytorch](https://github.com/ecologize/CameraTraps/releases/download/v5.0/md_v5a.0.0.pt)

# Fetch and alert
Fetch_and_alert.py combines two functions to retrieve images from [Strikeforce](https://www.strikeforcewireless.com) site and email addresses. Strikeforce username and password can be changed in the config/fetch_and_alert.yml file. Emails are accessed and images attached are extracted, email addresses may also be changed in the same config file. Once extracted these images are ran through both detector and classification models. And alerts (sends emails/texts/EarthRanger event) if a cougar (or any target animal you define in config as long as they are an option on the class list for your classification model) is detected.
To run fetch_and_alert.py, the config/fetch_and_alert.yml must be configured according to the notes in the file. The command line script to run is:

```
python3 fetch_and_alert.py config/fetch_and_alert.yml

```
## Installing animl
Currently, animl must be downloaded from github and placed within /cougarvision.
```
git clone https://github.com/conservationtechlab/animl-py
cd animl-py/src/
mv animl ~/<path to cougarvision>
```
## Installing yolov5
Detector functionality depends on the [Yolov5 repo](https://github.com/ultralytics/yolov5). While cougarvision conda environment is activated, run the following lines:
```
git clone https://github.com/ultralytics/yolov5
cd yolov5
python -m pip install yolov5
```
## Obtaining Strikeforce camera dictionary/auth_token
In order to retrieve photos from strikeforce, you need the api auth token for your account. We currently have a python script within /cougarvision_utils/strikeforceget.py that can obtain this auth_token given the strikeforce account username and password. 
Once you have this token, place it in 'auth_token' within the config file. 
To obtain the unique camera IDs in Strikeforce, run the script within /cougarvision_utils/strikeforcegetcameras.py with your email and the auth_token obtained from strikekforceget.py and it will output the unique strikeforce ID along with your camera names. 
```
python3 strikeforceget.py
python3 strikeforcegetcameras.py
```
In this version of CougarVision, all camera names need to be 4 characters long, so if your camera names exceed this length, shorten them to 4 characters within the config directory. 

## EarthRanger integration
CougarVision can optionally send detections for a species of interest to their specific camera location in EarthRanger. 
For this functionality, the er_alerts config must be set to "True". 
You also need to include an authorization token from EarthRanger. This can be obtained from <your_instance>.pamdas.org/api/v1.0/docs/interactive/. If you run the first example (das_server) get>try it out>execute, you will see a 'Curl' example input. There will be an X-CSRF and Authorization token, copy and paste these values into 'token' and 'authorization' in the config file, respectively (including 'Bearer ' for the authorization). These tokens will expire relatively quickly. To make them last longer, head on over to <your_instance>.pamdas.org/admin under 'Django OAuth Toolkit'>access tokens and paste the authorization token into the search bar(not including 'Bearer '). Click on the option that matches, and change the expiration date for however far out you'd like it to expire. It's important to note that you should keep this token secret, especially if it will not expire for a while.

In order for this integration to work, the camera name in EarthRanger must be the same as the 4 digit name in the camera dictionary in the config file. For ease of adding your cameras in the same format we have, there is a script in our [EarthRanger integration API package](https://github.com/conservationtechlab/sageranger/blob/main/sageranger/post_camera_er.py) that will add the cameras correctly. Verify that the cameras are visible on your EarthRanger map instance before proceeding with this integration.

## Email alerts
In order to send email alerts, CougarVision needs to know from what email to send them. For this, any email will do, but you need to include the email and password to the email. We just created a gmail specifically for this purpose, and include the email and the password under 'username' and 'password' in the config file. 

## Additional setup
Create two directories anywhere, one called 'images' and one called 'logs'. Place their file paths in the config under 'save_dir' and 'log_dir'. This will be where all photos from strikeforce are stored locally and where all the dataframes containing detection information for each photo will live. 

# Streaming
Stream_detect.py takes in a stream, runs a pipeline to detect various animals. An image is sent to specified emails/phone numbers based on config/stream_detect.yml.
If a lizard or a cougar is detected then a video is recorded for as long as that animal remains to be detected.
These scripts rely on the [CameraTraps Repo](https://github.com/microsoft/CameraTraps). Download this repo and place it anywhere. Add that path to config. The CameraTraps repo will then be added to the python sys environment variables upon loading an individual script.

# Processing Batch Images
Run_batch_images.py can take in an input folder and apply the detector/classifier pipeline to render annotated images along with an output.json file in directory classifications. The output.json contains the final classifications on the cropped images and can be used to compare against a ground truth set. There is also an intermediate output.json which holds the crop detections and is used by the script to crop images, this one can be moved by configuration in the yml file.
These scripts rely on the [CameraTraps Repo](https://github.com/microsoft/CameraTraps). Download this repo and place it anywhere. Add that path to config. The CameraTraps repo will then be added to the python sys environment variables upon loading an individual script.

# Improvement Efforts
If you encounter any issues upon install or would like a feature added, create an issue in the issue tracker on this github and we will do our best to accomodate!
