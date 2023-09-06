
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

from tqdm import tqdm

def _run_evaluation(path:Path, filenames: list[str], checkpoints: list[dict], plot: bool = False, _clip: bool=True, _FEATURES_TO_IGNORE: list[str] = ["ImageCentre", "LandingPost"], save: bool = False, save_path: Path = None):

    # ground truth 
    df = pd.read_csv(path)

    df_eval = pd.DataFrame(columns=["checkpoint", "fname", "feature", "p.x", "p.y", "pixelsize", "gt.p.x", "gt.p.y", "gt.pixelsize", "distance","_is_corrected"])
    df_eval["_is_corrected"] = df_eval["_is_corrected"].astype(bool)

    _prog = tqdm(filenames)
    for i, fname in enumerate(_prog):
        
        image_fname = os.path.basename(fname).removesuffix(".tif")
        if _clip:
            image_fname = image_fname[:-3]

        df_filt = df[df["image"] == image_fname]

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
            
            if feature.name in _FEATURES_TO_IGNORE:
                continue

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
        for k, CKPT in enumerate(checkpoints):

            CHECKPOINT = CKPT["checkpoint"]
            ENCODER = CKPT["encoder"]
            NC = CKPT["nc"]
            

            _prog.set_description(f"file ({i+1}/{len(filenames)}) {os.path.basename(fname)} - checkpoint: ({k+1}/{len(checkpoints)}) {os.path.basename(CHECKPOINT)}")
            model = load_model(CHECKPOINT, encoder=ENCODER, nc=NC)

            det  = detection.detect_features(image.data, model, features=deepcopy(gt_features), pixelsize=pixelsize, filter=True)
            dets.append(det)
        
        if plot:
            fig = detection.plot_detections(dets, titles=["Ground Truth"] + [os.path.basename(ckpt["checkpoint"].removesuffix(".pt")) for ckpt in checkpoints])

            plt.show()

            if save:
                os.makedirs(save_path, exist_ok=True)
                fig.savefig(os.path.join(save_path, f"{image_fname}.png"), dpi=300)
                plt.close(fig)

        # evaluation against ground truth
        for ckpt, det in zip(checkpoints, dets[1:]):
            checkpoint = ckpt["checkpoint"]
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

    df_eval["checkpoint"] = df_eval["checkpoint"].str.replace(".pt", "")

    if save:
        df_eval.to_csv(os.path.join(save_path, "eval.csv"), index=False)

    return df_eval


import plotly.express as px

def _plot_evalution_data(df: pd.DataFrame, category_orders: dict, show: bool = True, thresholds: list[int] =  [50, 25, 10]):
    # scatter plot
    fig = px.scatter(df, x="d.p.x", y="d.p.y", color="feature", facet_col="checkpoint", 
        # facet_row="dataset", 
        category_orders=category_orders,
        title="Distance from Ground Truth", hover_data=df.columns)
    if show:
        fig.show()


    # plot histogram
    fig = px.histogram(df, x="distance", color="feature", facet_col="checkpoint", nbins=100, title="Distance from Ground Truth", hover_data=df.columns, 
                    category_orders=category_orders)
    if show:
        fig.show()


    # plot box plot
    fig = px.box(df, x="feature", y="distance", color="feature", facet_col="checkpoint", title="Distance from Ground Truth", hover_data=df.columns,
                    category_orders=category_orders)

    if show:
        fig.show()

    # # calculate the mean and standard deviation in one aggregation per checkpoint, feature
    df_agg = df.groupby(["checkpoint", "feature"]).agg({"distance": ["mean", "median", "std"]}).reset_index()

    # display(df_agg)
    
    # plot percentage of distance under threshold
    df_ret = pd.DataFrame()
    for threshold in thresholds:

        df[f"under_threshold_{threshold}px"] = df["distance"] < threshold

        df_group = df.groupby(["checkpoint", "feature", f"under_threshold_{threshold}px"]).count().reset_index()
        # # pivot table
        df_group = df_group.pivot(index=["checkpoint", "feature"], columns=f"under_threshold_{threshold}px", values="distance").reset_index()
        
        df_group["threshold"] = threshold

        # add false if not present
        if False not in df_group.columns:
            df_group[False] = 0

        # add true if not present
        if True not in df_group.columns:
            df_group[True] = 0

        # fill na with 0
        df_group.fillna(0, inplace=True)

        # # calc percentage
        df_group[f"under_threshold"] = df_group[True] / (df_group[True] + df_group[False])
        
        # concat to df_ret
        df_ret = pd.concat([df_ret, df_group], axis=0)


    df_ret = df_ret[["checkpoint", "feature", "threshold", "under_threshold"]]

    # plot on bar chart with plotly express

    # sort column by category order
    df_ret["feature"] = pd.Categorical(df_ret["feature"], category_orders["feature"])
    df_ret["checkpoint"] = pd.Categorical(df_ret["checkpoint"], category_orders["checkpoint"])
    df_ret.sort_values(["checkpoint", "feature"], inplace=True)
    df_ret["threshold"] = df_ret["threshold"].astype(str)

    fig = px.bar(df_ret, x="threshold", y="under_threshold", color="feature", facet_col="checkpoint", barmode="group",                    
                        text=df_ret[f"under_threshold"].apply(lambda x: f"{x:.2f}"),
                        category_orders=category_orders, 
                        hover_data=df_ret.columns,
                        title="Percentage of Distance under threshold (px)")
    
    if show:
        fig.show()

    df_ret["threshold"] = df_ret["threshold"].astype(int)

    fig = px.line(df_ret, x="threshold", y="under_threshold", 
                  color="feature",
                    facet_col="checkpoint", 
                    # #facet_col="checkpoint", 
                    # text=df_ret[f"under_threshold"].apply(lambda x: f"{x:.2f}"),
                    category_orders=category_orders,
                    title="Percentage of Distance under threshold (px)")
    
    # reverse order of x axis
    fig.update_xaxes(autorange="reversed")
    if show:
        fig.show()


    # display(df_ret)

    # return 
