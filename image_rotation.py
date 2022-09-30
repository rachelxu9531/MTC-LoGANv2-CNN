

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import numpy as np
import glob
import re
import os
import cv2
import Pillow

from PIL import Image

#%matplotlib inline

cwd = r"C:\Conditional-StyleGAN\ConditionalStyleGAN-master\Seabed info\feature 2 - Trial 2"

classes = {'Blocks','Extensional ridges', 'Grooves and striations', 'Individual flow', 'MTC material', 'Polygonally faults',
           'Scarps', 'Slump folds', 'Undisturbed'}

all_path =[]
all_label =[]
for index, name in enumerate(classes):
    class_path = cwd +'\\' + '\\'
    for img_name in os.listdir(class_path):
        img_path = class_path + img_name
        img = np.asarray(PIL.Image.open(img_path))
        if len(img.shape) > 2 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = Image.fromarray(img)
        img = img.convert("RGB")
        img.save(img_path, 'png')