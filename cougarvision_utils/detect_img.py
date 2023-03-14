from animl import FileManagement, ImageCropGenerator, DetectMD
from cougarvision_utils.cropping import draw_bounding_box_on_image
from cougarvision_utils.alert import smtp_setup, sendAlert
from io import BytesIO
from datetime import datetime as dt
from PIL import Image
import ruamel.yaml


def detect(images, config_file):
    yaml = ruamel.yaml.YAML()
    with open(config_file, 'r') as f:
        config = yaml.load(f)
    detector_model = config['detector_model']
    log_dir = config['log_dir']
    checkpoint_frequency = config['checkpoint_frequency']
    confidence_threshold = config['confidence']
    if len(images) > 0:
        # extract paths from dataframe
        image_paths = images[:, 2]
        # Run Detection
        results = DetectMD.load_and_run_detector_batch(image_paths,
                                                       detector_model,
                                                       log_dir,
                                                       confidence_threshold,
                                                       checkpoint_frequency,
                                                       [])
        # Parse results
        df = FileManagement.parseMD(results)
        # filter out all non animal detections
        if not df.empty:
            animalDataframe, otherDataframe = FileManagement.filterImages(df)
            # run classifier on animal detections if there are any
            if not animalDataframe.empty:
                # create generator for images
                generator = ImageCropGenerator.\
                    GenerateCropsFromFile(animalDataframe)
                # Run Classifier
                predictions = model.predict_generator(generator,
                                                      steps=len(generator),
                                                      verbose=1)
                # Parse results
                maxDataframe = FileManagement.parseCM(animalDataframe, None,
                                                      predictions, classes)
                # Creates a data frame with all relevant data
                cougars = maxDataframe[maxDataframe['class'].isin(targets)]
                # drops all detections with confidence less than threshold
                cougars = cougars[cougars['conf'] >= confidence_threshold]
                # reset dataframe index
                cougars = cougars.reset_index(drop=True)
                # Sends alert for each cougar detection
                for idx in range(len(cougars.index)):
                    label = cougars.at[idx, 'class']
                    prob = cougars.at[idx, 'conf']
                    img = Image.open(cougars.at[idx, 'file'])
                    draw_bounding_box_on_image(img,
                                               cougars.at[idx, 'bbox2'],
                                               cougars.at[idx, 'bbox1'],
                                               cougars.at[idx,
                                                          'bbox2'] +
                                               cougars.at[idx,
                                                          'bbox4'],
                                               cougars.at[idx,
                                                          'bbox1'] +
                                               cougars.at[idx,
                                                          'bbox3'],
                                               expansion=0,
                                               use_normalized_coordinates=True)
                    imageBytes = BytesIO()
                    img.save(imageBytes, format=img.format)
                    smtp_server = smtp_setup(username, password, host)
                    sendAlert(label, prob, imageBytes, smtp_server,
                              username, to_emails)
                # Write Dataframe to csv
                date = "%m-%d-%Y_%H:%M:%S"
                cougars.to_csv(f'{log_dir}dataframe_{dt.now().strftime(date)}')
