import glob
import logging
import os
import re
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from fibsem import config as cfg
from fibsem import utils
from fibsem.detection.detection import DetectedFeatures
from fibsem.structures import FibsemImage, FibsemImageMetadata, Point

def decode_segmap(image, nc=3):

    """ Decode segmentation class mask into an RGB image mask"""

    # 0=background, 1=lamella, 2= needle
    label_colors = np.array([(0, 0, 0),
                                (255, 0, 0),
                                (0, 255, 0)])

    # pre-allocate r, g, b channels as zero
    r = np.zeros_like(image, dtype=np.uint8)
    g = np.zeros_like(image, dtype=np.uint8)
    b = np.zeros_like(image, dtype=np.uint8)

    # apply the class label colours to each pixel
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    # stack rgb channels to form an image
    rgb_mask = np.stack([r, g, b], axis=2)
    return rgb_mask

def coordinate_distance(p1: Point, p2: Point):
    """Calculate the distance between two points in each coordinate"""

    return p2.x - p1.x, p2.y - p1.y

def scale_pixel_coordinates(px: Point, from_image: FibsemImage, to_image: FibsemImage) -> Point:
    """Scale the pixel coordinate from one image to another"""

    invariant_pt = get_scale_invariant_coordinates(px, from_image.data.shape)

    scaled_px = scale_coordinate_to_image(invariant_pt, to_image.data.shape)

    return scaled_px


def get_scale_invariant_coordinates(point: Point, shape: tuple) -> Point:
    """Convert the point coordinates from image coordinates to scale invariant coordinates"""
    scaled_pt = Point(x=point.x / shape[1], y=point.y / shape[0])

    return scaled_pt


def scale_coordinate_to_image(point: Point, shape: tuple) -> Point:
    """Scale invariant coordinates to image shape"""
    scaled_pt = Point(x=int(point.x * shape[1]), y=int(point.y * shape[0]))

    return scaled_pt

def parse_metadata(filename):

    # FIB meta data key is 34682, comes as a string
    img = Image.open(filename)
    img_metadata = img.tag[34682][0]

    # parse metadata
    parsed_metadata = img_metadata.split("\r\n")

    metadata_dict = {}
    for item in parsed_metadata:

        if item == "":
            # skip blank lines
            pass
        elif re.match(r"\[(.*?)\]", item):
            # find category, dont add to dict
            category = item
        else:
            # meta data point
            datum = item.split("=")

            # save to dictionary
            metadata_dict[category + "." + datum[0]] = datum[1]

    # add filename to metadata
    metadata_dict["filename"] = filename

    # convert to pandas df
    df = pd.DataFrame.from_dict(metadata_dict, orient="index").T

    return df


def extract_img_for_labelling(path, show=False):
    """Extract all the images that have been identified for retraining.

    path: path to directory containing logged images

    """
    import datetime
    import random
    import time

    import liftout
    import matplotlib.pyplot as plt
    from PIL import Image

    # mkdir for copying images to
    data_path = os.path.join(os.path.dirname(liftout.__file__), "data", "retrain")
    os.makedirs(data_path, exist_ok=True)
    print(f"Searching in {path} for retraining images...")

    # find all files for retraining (with _label postfix
    filenames = glob.glob(os.path.join(path, "/**/*label*.tif"), recursive=True)
    print(f"{len(filenames)} images found for relabelling")
    print(f"Copying images to {data_path}...")

    for i, fname in enumerate(filenames):
        # tqdm?
        print(f"Copying {i}/{len(filenames)}")
        # basename = os.path.basename(fname)
        datetime_str = datetime.datetime.fromtimestamp(time.time()).strftime(
            "%Y%m%d.%H%M%S"
        )
        basename = f"{datetime_str}.{random.random()}.tif"  # use a random number to prevent duplicates at seconds time resolution
        # print(fname, basename)
        if show:
            img = Image.open(fname)
            plt.imshow(img, cmap="gray")
            plt.show()

        source_path = os.path.join(fname)
        destination_path = os.path.join(data_path, basename)
        # print(f"Source: {source_path}")
        # print(f"Destination: {destination_path}")
        print("-" * 50)
        shutil.copyfile(source_path, destination_path)

    # zip the image folder
    # shutil.make_archive(f"{path}/images", 'zip', label_dir)

def write_data_to_csv(path: Path, info: list) -> None:

    dataframe_path = os.path.join(path, "data.csv")

    cols = ["label", "p1.type", "p1.x", "p1.y", "p2.type", "p2.x", "p2.y"]
    df_tmp = pd.DataFrame([info], columns=cols)
    if os.path.exists(dataframe_path):
        df = pd.read_csv(dataframe_path)
        df = pd.concat([df, df_tmp], axis=0, ignore_index=True)
    else:
        df = df_tmp
    df.to_csv(dataframe_path,index=False)

    logging.info(f"Logged data to {dataframe_path}.")


def write_data_to_disk(path: Path, detected_features) -> None:
    from fibsem import utils
    from fibsem.detection.detection import DetectedFeatures
    label = utils.current_timestamp() + "_label"

    # utils.save_image(
    # image=detected_features.image,
    # save_path=path,
    # label=label,
    # )

    import os

    import tifffile as tf 
    os.makedirs(path, exist_ok=True)
    tf.imsave(os.path.join(path, f"{label}.tif"), detected_features.image)

    # get scale invariant coords
    shape = detected_features.image.shape
    scaled_p0 = get_scale_invariant_coordinates(detected_features.features[0].px, shape=shape)
    scaled_p1 = get_scale_invariant_coordinates(detected_features.features[1].px, shape=shape)

    # get info
    info = [label, 
        detected_features.features[0].type.name, 
        scaled_p0.x, 
        scaled_p0.y, 
        detected_features.features[1].type.name, 
        scaled_p1.x, 
        scaled_p1.y
        ]

    write_data_to_csv(path, info)


def save_data(det: DetectedFeatures, corrected: bool = False, fname: str = None) -> None:

    image = det.fibsem_image if det.fibsem_image is not None else det.image
    if not isinstance(image, FibsemImage):
        image = FibsemImage(image, None)
    
    if fname is None:
        fname = f"{utils.current_timestamp_v2()}"
    fname = os.path.join(cfg.DATA_ML_PATH, f"{fname}")

    idx = 1
    while os.path.exists(fname):
        fname = f"{fname}_{idx}"
        idx += 1

    image.save(fname) # type: ignore 
    logging.info(f"Saved image to {fname}")

    # save mask to disk
    os.makedirs(os.path.join(cfg.DATA_ML_PATH, "mask"), exist_ok=True)
    mask_fname = os.path.join(cfg.DATA_ML_PATH, "mask", os.path.basename(fname))
    mask_fname = Path(mask_fname).with_suffix(".tif")
    im = Image.fromarray(det.mask) 
    im.save(mask_fname)


    # save coordinates for testing    
    # save the feature_type, px coordinates for each feature into a pandas dataframe
    feat_list = []
    for i, feature in enumerate(det.features):

        dat = {"feature": feature.name, 
                        "p.x": feature.px.x, 
                        "p.y": feature.px.y, 
                    "beam_type": "ELECTRON", 
                    "image": os.path.basename(fname), 
                    "pixelsize": det.pixelsize,
                    "corrected": corrected} # TODO: beamtype
        feat_list.append(dat)

    df = pd.DataFrame(feat_list)
    
    # save the dataframe to a csv file, append if the file already exists
    DATAFRAME_PATH = os.path.join(cfg.DATA_ML_PATH, "data.csv")
    if os.path.exists(DATAFRAME_PATH):
        df_tmp = pd.read_csv(DATAFRAME_PATH)
        df = pd.concat([df_tmp, df], axis=0, ignore_index=True)
    
    # logging.info(f"{df.tail(10)}")
    df.to_csv(DATAFRAME_PATH, index=False)