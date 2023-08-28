
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


from fibsem.detection import detection
from fibsem.segmentation.model import load_model
from fibsem.structures import FibsemImage
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import os
from pprint import pprint


from fibsem.structures import Point
from fibsem.detection import detection
import pandas as pd
from copy import deepcopy



def _run_evaluation(path:Path, filenames: list[str], checkpoints: list[str], plot: bool = False):

    # ground truth 
    df = pd.read_csv(path)

    df_eval = pd.DataFrame(columns=["checkpoint", "fname", "feature", "p.x", "p.y", "pixelsize", "gt.p.x", "gt.p.y", "gt.pixelsize", "distance","_is_corrected"])
    df_eval["_is_corrected"] = df_eval["_is_corrected"].astype(bool)

    _prog = tqdm(filenames)
    for i, fname in enumerate(_prog):

        image_fname = os.path.basename(fname)[:-7]
        df_filt = df[df["image"]==image_fname]

        if len(df_filt) == 0:
            print(f"no ground truth for {fname}")
            continue

        _is_corrected = df_filt["corrected"].values[0]

        # extract each row
        gt_features = []
        pixelsize = 25e-9
        for _, row in df_filt.iterrows():

            feature = detection.get_feature(row["feature"])
            feature.px = Point(row["p.x"], row["p.y"])
            pixelsize= row["pixelsize"]

            gt_features.append(feature)
        

        image = FibsemImage.load(fname)

        # plot
        # TODO: updat when get gt mask / rgb
        gt_det = detection.DetectedFeatures(
            features = gt_features,
            image=image.data,
            mask=None,
            rgb=None,
            pixelsize=pixelsize, 
            fibsem_image=image)

        dets = [gt_det]
        for k, CHECKPOINT in enumerate(checkpoints):

            _prog.set_description(f"file ({i+1}/{len(filenames)}) {os.path.basename(fname)} - checkpoint: ({k+1}/{len(checkpoints)}) {os.path.basename(CHECKPOINT)}")
            model = load_model(CHECKPOINT, encoder="resnet34", nc=3)

            det  = detection.detect_features(image.data, model, features=deepcopy(gt_features), pixelsize=pixelsize, filter=True)
            dets.append(det)
        
        if plot:
            fig = detection.plot_detections(dets, titles=["Ground Truth"] + [os.path.basename(checkpoint) for checkpoint in checkpoints])

        # evaluation against ground truth
        for checkpoint, det in zip(checkpoints, dets[1:]):

            for feature in det.features:
                gt_feat = gt_det.get_feature(feature.name)
                dpx = feature.px - gt_feat.px
                    
                df_eval2 = pd.DataFrame([{
                    "checkpoint": os.path.basename(checkpoint),
                    "fname": os.path.basename(fname),
                    "gt_fname": image_fname,
                    "feature": feature.name,
                    "p.x": feature.px.x,
                    "p.y": feature.px.y,
                    "pixelsize": pixelsize,
                    "gt.p.x": gt_feat.px.x,
                    "gt.p.y": gt_feat.px.y,
                    "gt.pixelsize": pixelsize,
                    "distance": feature.px.euclidean(gt_feat.px),
                    "d.p.x": dpx.x,
                    "d.p.y": dpx.y,
                    "_is_corrected": bool(_is_corrected), 
                }])

                # convert to bool type
                df_eval2["_is_corrected"] = df_eval2["_is_corrected"].astype(bool)
                df_eval = pd.concat([df_eval, df_eval2], axis=0, ignore_index=True)
        
        # print("-"*80, "\n")
        # if i==3:
            # break

    return df_eval