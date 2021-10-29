# cougarvision
Tools to automatically analyze images and videos from telemetering field cameras and to take responsive action. Core mechanism is combining Megadetector [see](https://github.com/microsoft/CameraTraps), with interchangeable imagenet, torchscript compiled classifier models.

## Streaming
Stream_detect.py takes in a stream, runs a pipeline to detect various animals. An image is sent to specified emails/phone numbers based on config/stream_detect.yml.
If a lizard or a cougar is detected then a video is recorded for as long as that animal remains to be detected.

## Static Emails
Fetch_and_alert.py takes in an email address which periodically receives email with images attached. Minimum fetch period is 10 minutes while working with gmail. These images
are ran through the pipeline and alerts (sends emails/texts) if a human, vehicle, or cougar is detected.


## Setting up Conda Environment

[Instructions to install conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

The file **cougarvision_env.yml** describes the python version and various dependencies with specific version numbers. To activate the enviroment

```
conda env create -f cougarvision_env.yml

conda activate cougarvision_env

conda env list

```

The first line creates the enviroment from the specifications file which only needs to be done once. 

The second line activates the environment which may need to be done each time you restart the terminal.

The third line is to test the environment installation where it will list all dependencies for that environment

## CameraTraps Repo

These scripts rely on the [CameraTraps Repo](https://github.com/microsoft/CameraTraps). Download this repo and place it anywhere. Add that path to config/cameratraps.yml. The CameraTraps repo will then be added to the python sys environment variables upon loading an individual script.

## Models

Detection and inference models can be place anywhere. In each script's yml file (under the config directory) is a field where the path variables for each model can be specified.


