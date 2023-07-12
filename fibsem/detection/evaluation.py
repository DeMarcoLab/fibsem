
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob


from fibsem.structures import FibsemImage

from fibsem.detection import detection

import os
from copy import deepcopy
from pathlib import Path
import numpy as np

from fibsem.detection import detection
from fibsem.detection import utils as det_utils
from fibsem.detection.detection import DetectedFeatures
from fibsem.segmentation import model as fibsem_model
from fibsem.segmentation.model import load_model
from fibsem.structures import (
    BeamType,
    FibsemImage,
    Point,
)
import logging
from tqdm import tqdm
# from stqdm import stqdm

def run_eval(path: Path, checkpoint:str, encoder:str = "resnet34", num_classes: int = 3, show: bool = False):

    df = pd.read_csv(os.path.join(path, "data.csv"))

    model = load_model(checkpoint=checkpoint, encoder=encoder, nc=num_classes)


    # get filenames from image column
    filenames = df["image"].unique()

    data_list = []


    # USE_DATAFRAME = True
    # gt_model = load_model(checkpoint=None, encoder=encoder, nc=num_classes)
    # print(gt_model.device)
    # features = ["NeedleTip", "LamellaCentre", "LamellaLeftEdge", "LamellaRightEdge", "ImageCentre", "LandingPost"]

    for fname in tqdm(filenames):
        
        
        image = FibsemImage.load(os.path.join(path, f"{fname}.tif"))

        # should be able to get gt from model or from df

        # get ground truth
        gt = df[df["image"] == fname]
        features = gt["feature"].to_list()
        pixelsize = gt["pixelsize"].to_list()[0]
        features = [detection.get_feature(feature) for feature in features]
        gt_det = detection._det_from_df(gt, path, fname)

        det = detection.detect_features(image.data, model, features, pixelsize=pixelsize)

        for det_f in det.features:

            # compare the equivalent ground truth feature
            gt_f = gt_det.get_feature(det_f.name)

            # get the distance between the two features
            dist = det_f.px.euclidean(gt_f.px)
            sub = det_f.px - gt_f.px


            dat = {
                "fname": fname,
                "feature": det_f.name, 
                "f.px.x": det_f.px.x,
                "f.px.y": det_f.px.y,
                "gt.px.x": gt_f.px.x,
                "gt.px.y": gt_f.px.y,
                "dx": sub.x,
                "dy": sub.y,
                "distance": dist,
                "checkpoint": os.path.basename(checkpoint),
                "encoder": encoder, 
                }

            data_list.append(dat)
        if show:
            detection.plot_detections([gt_det, det], titles=["Ground Truth", "Prediction"])


    df_eval = pd.DataFrame(data_list)

    # save to csv
    df_eval.to_csv(os.path.join(path, "eval.csv"), index=False)
    
    return df_eval