import yaml


with open("config/stream_detect.yml", 'r') as stream:
    config = yaml.safe_load(stream)

print(config['detector_model'])