
import glob
from fibsem.segmentation.model import SegmentationModel
import tifffile as tf
import matplotlib.pyplot as plt
from random import shuffle
import numpy as np 
import pandas as pd
# from autoscript_sdb_microscope_client.structures import AdornedImage

from fibsem.structures import Point
from fibsem.imaging import masks
from fibsem.detection.detection import NeedleTip, LamellaCentre, locate_shift_between_features_v2
import skimage

from pathlib import Path
from dataclasses import dataclass
from fibsem import conversions
from pprint import pprint
from fibsem.segmentation.model import load_model



### load evaluation folder 

eval_folder = r'C:\Users\rkan0039\Documents\detection_training\new_eval\label_00\{}'

filenames = glob.glob(eval_folder.format('*.tif'))

pprint(filenames)

image_names = []

for file in filenames:

    name = file.split('\\')[-1]
    pprint(name)
    image_names.append(name)

GT_Table = pd.read_csv(eval_folder.format('data.csv'))

print("hello")
