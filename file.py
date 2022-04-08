import numpy as np
with open('/home/jared/cougarvision/labels/southwest_labels.txt', 'r') as f:
    data = f.read().splitlines()

print(np.asarray(data)[0])