#!/bin/bash

# path to conda shell, may need to replace anaconda with miniconda
# depending on your installation
source /home/<user>/anaconda3/etc/profile.d/conda.sh

# activate environment
conda activate cougarvision

# run fetch and alert
python3 -u /home/<user>/cougarvision/fetch_and_alert.py /home/<user>/cougarvision/config/fetch_and_alert.yml
