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
from autoscript_sdb_microscope_client.structures import AdornedImage
from fibsem import conversions
from fibsem.structures import Point
from PIL import Image

# TODO: rename from detection to features, e.g. FeatureType

class FeatureType(Enum):
    LamellaCentre = 1
    NeedleTip = 2
    LamellaRightEdge = 3
    LamellaLeftEdge = 4
    LandingPost = 5
    ImageCentre = 6


@dataclass
class Feature:
    detection_type: FeatureType
    feature_px: Point  = None # x, y (image)
    feature_m: Point = None # x, y (microscope image coord)


@dataclass
class DetectionResult:
    features: list[Feature]
    adorned_image: AdornedImage
    display_image: np.ndarray
    distance_metres: Point = Point(0, 0)  # x, y
    microscope_coordinate: list[Point] = None


# detection colour map
DETECTION_TYPE_COLOURS = {
    FeatureType.LamellaCentre: (1, 0, 0, 1),
    FeatureType.NeedleTip: (0, 1, 0, 1),
    FeatureType.LamellaLeftEdge: (1, 0.5, 0.5, 1),
    FeatureType.LamellaRightEdge: (1, 0.5, 0, 1),
    FeatureType.LandingPost: (0, 1, 1, 1),
    FeatureType.ImageCentre: (1, 1, 1, 1)
}

DETECTION_TYPE_COLOURS_v2 = {
    FeatureType.LamellaCentre: "red",
    FeatureType.NeedleTip: "green",
    FeatureType.LamellaLeftEdge: "magenta",
    FeatureType.LamellaRightEdge: "orange",
    FeatureType.LandingPost: "cyan",
    FeatureType.ImageCentre: "white"
}

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






# TODO: refactor usage to distance between two points
def convert_pixel_distance_to_metres(p1: Point, p2: Point, adorned_image: AdornedImage):
    """Convert from pixel coordinates to distance in metres """

    # convert pixel coordinate to realspace coordinate
    x1_real, y1_real = conversions.pixel_to_realspace_coordinate(
        (p1.x, p1.y), adorned_image
    )
    x2_real, y2_real = conversions.pixel_to_realspace_coordinate(
        (p2.x, p2.y), adorned_image
    )

    p1_real = Point(x1_real, y1_real)
    p2_real = Point(x2_real, y2_real)

    # calculate distance between points along each axis
    x_distance_m, y_distance_m = coordinate_distance(p1_real, p2_real)

    return x_distance_m, y_distance_m

def coordinate_distance(p1: Point, p2: Point):
    """Calculate the distance between two points in each coordinate"""

    return p2.x - p1.x, p2.y - p1.y

def scale_pixel_coordinates(px: Point, from_image: AdornedImage, to_image: AdornedImage) -> Point:
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



def load_detection_result(path: Path, data) -> DetectionResult:
    """Read detection result from dataframe row, and return"""

    label = data["label"]
    p1_type = FeatureType[data["p1.type"]]
    p1 = Point(x=data["p1.x"], y=data["p1.y"])
    p2_type = FeatureType[data["p2.type"]]
    p2 = Point(x=data["p2.x"], y=data["p2.y"])

    fname = glob.glob(os.path.join(path, f"*{label}*.tif"))[0]
    img = AdornedImage.load(fname)

    p1 = scale_coordinate_to_image(p1, img.data.shape)
    p2 = scale_coordinate_to_image(p2, img.data.shape)

    det = DetectionResult(
        features=[
            Feature(detection_type=p1_type, feature_px=p1),
            Feature(detection_type=p2_type, feature_px=p2),
        ],
        adorned_image=img,
        display_image=None,
    )

    return det


def plot_detection_result(det_result: DetectionResult):
    """Plot the Detection Result using matplotlib using the full scale image and coordinates"""
    from fibsem.detection.utils import DETECTION_TYPE_COLOURS

    # TODO: consolidate this with what is in detection_window

    p1 = det_result.features[0].feature_px
    p2 = det_result.features[1].feature_px

    c1 = DETECTION_TYPE_COLOURS[det_result.features[0].detection_type]
    c2 = DETECTION_TYPE_COLOURS[det_result.features[1].detection_type]

    if det_result.display_image is None:
        display_image = det_result.adorned_image.data
    else:
        display_image = det_result.display_image

    fig = plt.figure(figsize=(15, 15))
    plt.title(f"{det_result.features[0].detection_type.name} to {det_result.features[1].detection_type.name}")
    plt.imshow(display_image, cmap="gray")
    plt.plot(p1.x, p1.y, color=c1, marker="+", ms=50, markeredgewidth=2)
    plt.plot(p2.x, p2.y, color=c2, marker="+", ms=50, markeredgewidth=2)
    plt.plot((p1.x, p2.x),(p1.y, p2.y), color="white", ms=50, markeredgewidth=2) # line between

    # legend
    patch_one = mpatches.Patch(color=c1, label=det_result.features[0].detection_type.name)
    patch_two = mpatches.Patch(color=c2, label=det_result.features[1].detection_type.name)
    plt.legend(handles=[patch_one, patch_two])

    return fig


def write_data_to_disk(path: Path, detected_features) -> None:
    from fibsem.detection.detection import DetectedFeatures

    from fibsem import utils
    label = utils.current_timestamp() + "_label"

    # utils.save_image(
    # image=detected_features.image,
    # save_path=path,
    # label=label,
    # )

    import tifffile as tf
    import os 
    os.makedirs(path, exist_ok=True)
    tf.imsave(os.path.join(path, f"{label}.tif"), detected_features.image)

    # get scale invariant coords
    shape = detected_features.image.shape
    scaled_p0 = get_scale_invariant_coordinates(detected_features.features[0].feature_px, shape=shape)
    scaled_p1 = get_scale_invariant_coordinates(detected_features.features[1].feature_px, shape=shape)

    # get info
    logging.info(f"Label: {label}")
    info = [label, 
        detected_features.features[0].detection_type.name, 
        scaled_p0.x, 
        scaled_p0.y, 
        detected_features.features[1].detection_type.name, 
        scaled_p1.x, 
        scaled_p1.y
        ]

    write_data_to_csv(path, info)

