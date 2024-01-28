
import glob
import os
from copy import deepcopy
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from fibsem.detection import detection
from fibsem.segmentation.model import load_model
from fibsem.segmentation.utils import decode_segmap_v2
from fibsem.structures import FibsemImage, Point

import tifffile as tff


def _run_evaluation(path:Path, image_path: Path, checkpoints: list[dict], plot: bool = False, _FEATURES_TO_IGNORE: list[str] = ["ImageCentre", "LandingPost"], save: bool = False, save_path: Path = None):

    # ground truth 
    df = pd.read_csv(path)

    filename_col = "image" if "image" in df.columns else "filename"
    image_fnames = df[filename_col].unique().tolist()

    # glob each image in the image_path and add to filenames
    filenames = []
    
    for fname in image_fnames:
        fname = fname.removesuffix(".tif")
        filenames += glob.glob(os.path.join(image_path, f"{fname}*.tif"))
    
    print(f"Found {len(filenames)} images (test)")

    # setup eval dataframe
    px_col = "p.x" if "p.x" in df.columns else "px.x"
    py_col = "p.y" if "p.y" in df.columns else "px.y"
    df_eval = pd.DataFrame(columns=["checkpoint", "fname", "feature", px_col, py_col, "pixelsize", "gt.p.x", "gt.p.y", "gt.pixelsize", "distance","_is_corrected"])
    df_eval["_is_corrected"] = df_eval["_is_corrected"].astype(bool)

    _prog = tqdm(sorted(filenames))
    for i, fname in enumerate(_prog):
        
        image_fname = os.path.basename(fname).removesuffix(".tif")

        # if suffix is either _eb or _ib, remove it
        if image_fname.endswith("_eb") or image_fname.endswith("_ib"):
            image_fname = image_fname[:-3]

        df_filt = df[df[filename_col] == image_fname]

        if len(df_filt) == 0:
            print(f"no ground truth for {fname}")
            continue

        _is_corrected = df_filt["corrected"].values[0]

        # extract each row
        gt_features = []
        pixelsize = 25e-9
        for _, row in df_filt.iterrows():

            feature = detection.get_feature(row["feature"])
            feature.px = Point(row[px_col], row[py_col])
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
        
        if plot or save:
            fig = detection.plot_detections(dets, titles=["Ground Truth"] + [os.path.basename(ckpt["checkpoint"].removesuffix(".pt")) for ckpt in checkpoints])
            if plot:
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



def run_evaluation_v2(path:Path, image_path: Path, checkpoints: list[dict], labels_path: Path = None, 
                      plot: bool = False, _FEATURES_TO_IGNORE: list[str] = ["ImageCentre"], 
                      save: bool = False, save_path: Path = None):

    # ground truth 
    df = pd.read_csv(path)

    image_fnames = df["filename"].unique().tolist()

    # glob each image in the image_path and add to filenames
    filenames = []
    
    # TODO: keep track of whether .tif ext was recorded in the csv

    for fname in image_fnames:
        fname = fname.removesuffix(".tif")
        filenames += glob.glob(os.path.join(image_path, f"{fname}*.tif"))
    
    print(f"Found {len(filenames)} images (test)")

    if labels_path is None:
        labels_path = os.path.join(image_path, "labels")

    # setup eval dataframe
    df_eval = pd.DataFrame(columns=["checkpoint", "fname", "feature", "px.x", "px.y", "pixelsize", "gt.p.x", "gt.p.y", "gt.pixelsize", "distance","_is_corrected"])
    df_eval["_is_corrected"] = df_eval["_is_corrected"].astype(bool)

    _prog = tqdm(sorted(filenames))
    for i, fname in enumerate(_prog):
        
        image_fname = os.path.basename(fname)#.removesuffix(".tif")

        # if suffix is either _eb or _ib, remove it
        if image_fname.endswith("_eb") or image_fname.endswith("_ib"):
            image_fname = image_fname[:-3]

        df_filt = df[df["filename"] == image_fname]

        if len(df_filt) == 0:
            print(f"no ground truth for {fname}")
            continue

        _is_corrected = df_filt["corrected"].values[0]

        # extract each row
        gt_features = []
        pixelsize = 25e-9
        for _, row in df_filt.iterrows():

            # if entries in row are null, go to next image
            if pd.isnull(row["feature"]):
                continue

            feature = detection.get_feature(row["feature"])
            feature.px = Point(row["px.x"], row["px.y"])
            pixelsize= row["pixelsize"]
            
            if feature.name in _FEATURES_TO_IGNORE:
                continue

            gt_features.append(feature)
        

        image = FibsemImage.load(fname)

        # plot
        mask, rgb = None, None
        label_fname = os.path.join(labels_path, f"{image_fname}")
        if os.path.exists(label_fname):
            mask = tff.imread(os.path.join(labels_path, f"{image_fname}"))
            rgb = decode_segmap_v2(mask)
        gt_det = detection.DetectedFeatures(
            features = gt_features,
            image=image.data,
            mask=mask,
            rgb=rgb,
            pixelsize=pixelsize, 
            fibsem_image=image)

        dets = [gt_det]
        for k, CKPT in enumerate(checkpoints):

            CHECKPOINT = CKPT["checkpoint"]           

            _prog.set_description(f"file ({i+1}/{len(filenames)}) {os.path.basename(fname)} - checkpoint: ({k+1}/{len(checkpoints)}) {os.path.basename(CHECKPOINT)}")
            model = load_model(CHECKPOINT) 

            det  = detection.detect_features(image.data, model, features=deepcopy(gt_features), pixelsize=pixelsize, filter=True)
            dets.append(det)
        
        if plot or save:
            fig = detection.plot_detections(dets, titles=["Ground Truth"] + [os.path.basename(ckpt["checkpoint"].removesuffix(".pt")) for ckpt in checkpoints])
            if plot:
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


def plot_evaluation_data(df: pd.DataFrame, category_orders: dict, show: bool = True, 
                            thresholds: list[int] =  [50, 25, 10], save: bool=False, save_path: Path = None):
    # scatter plot
    fig = px.scatter(df, x="d.p.x", y="d.p.y", color="feature", facet_col="checkpoint", 
        # facet_row="dataset", 
        category_orders=category_orders,
        title="Distance from Ground Truth", hover_data=df.columns)
    
    if save:
        # save fig as png
        fig.write_image(os.path.join(save_path, "eval00.png"))
    
    if show:
        fig.show()


    # plot histogram
    fig = px.histogram(df, x="distance", color="feature", facet_col="checkpoint", nbins=100, title="Distance from Ground Truth", hover_data=df.columns, 
                    category_orders=category_orders)
    if show:
        fig.show()

    if save:
        # save fig as png
        fig.write_image(os.path.join(save_path, "eval01.png"))
    

    # plot box plot
    fig = px.box(df, x="feature", y="distance", color="feature", facet_col="checkpoint", title="Distance from Ground Truth", hover_data=df.columns,
                    category_orders=category_orders)

    if show:
        fig.show()
    
    if save:
        # save fig as png
        fig.write_image(os.path.join(save_path, "eval02.png"))
    
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
    
    if save:
        # save fig as png
        fig.write_image(os.path.join(save_path, "eval03.png"))
    

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

    if save:
        # save fig as png
        fig.write_image(os.path.join(save_path, "eval04.png"))
    

    # display(df_ret)
        
    # calculate the accuracy at each threshold, save to csv
    df_ret["accuracy"] = df_ret["under_threshold"] * 100
    df_ret["accuracy"] = df_ret["accuracy"]
    df_ret["threshold"] = df_ret["threshold"].astype(int)

    df_ret.to_csv(os.path.join(save_path, "eval_accuracy.csv"), index=False)


    # return 
