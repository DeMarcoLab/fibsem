#!/usr/bin/env python3
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.spatial import distance
from skimage import feature, measure

from fibsem import conversions
from fibsem.imaging import masks
from fibsem.microscope import FibsemMicroscope
from fibsem.segmentation.model import SegmentationModel
from fibsem.structures import FibsemImage, BeamType, MicroscopeSettings, Point
import matplotlib.pyplot as plt

@dataclass
class Feature(ABC):
    px: Point 
    feature_m: Point
    _color_UINT8: tuple = (255,255,255)
    color = "white"
    name: str = None

    @abstractmethod
    def detect(self, img: np.ndarray, mask: np.ndarray=None, point:Point=None) -> 'Feature':
        pass

@dataclass
class ImageCentre(Feature):
    feature_m: Point = None
    px: Point = None
    _color_UINT8: tuple = (255,255,255)
    color = "white"
    name: str = "ImageCentre"

    def detect(self, img: np.ndarray, mask: np.ndarray=None, point:Point=None) -> 'ImageCentre':
        self.px = Point(x=img.shape[1] // 2, y=img.shape[0] // 2)
        return self.px



@dataclass
class NeedleTip(Feature):
    feature_m: Point = None
    px: Point = None
    _color_UINT8: tuple = (0,255,0)
    color = "green"
    name: str = "NeedleTip"

    def detect(self, img: np.ndarray, mask: np.ndarray = None, point:Point=None) -> 'NeedleTip':
        self.px = detect_needle_v4(mask)
        return self.px



@dataclass
class LamellaCentre(Feature):
    feature_m: Point = None
    px: Point = None
    _color_UINT8: tuple = (255,0,0)
    color = "red"
    name: str = "LamellaCentre"

    def detect(self, img: np.ndarray, mask: np.ndarray = None, point:Point=None) -> 'LamellaCentre':
        self.px = detect_lamella(mask, self)
        return self.px


@dataclass
class LamellaLeftEdge(Feature):
    feature_m: Point = None
    px: Point = None
    _color_UINT8: tuple = (255,0,255)
    color = "magenta"
    name: str = "LamellaLeftEdge"

    def detect(self, img: np.ndarray, mask: np.ndarray = None, point:Point=None) -> 'LamellaLeftEdge':
        self.px = detect_lamella(mask, self)
        return self.px


@dataclass
class LamellaRightEdge(Feature):
    feature_m: Point = None
    px: Point = None
    _color_UINT8: tuple = (255,165,0)
    color = "orange"
    name: str = "LamellaRightEdge"

    def detect(self, img: np.ndarray, mask: np.ndarray = None, point:Point=None) -> 'LamellaRightEdge':
        self.px = detect_lamella(mask, self)
        return self.px

@dataclass
class LamellaTopEdge(Feature):
    feature_m: Point = None
    px: Point = None
    _color_UINT8: tuple = (255,0,255)
    color = "hotpink"
    name: str = "LamellaTopEdge"

    def detect(self, img: np.ndarray, mask: np.ndarray = None, point:Point=None) -> 'LamellaTopEdge':
        self.px = detect_lamella(mask, self)
        return self.px

@dataclass
class LamellaBottomEdge(Feature):
    feature_m: Point = None
    px: Point = None
    _color_UINT8: tuple = (0, 0, 255)
    color = "blue"
    name: str = "LamellaBottomEdge"

    def detect(self, img: np.ndarray, mask: np.ndarray = None, point:Point=None) -> 'LamellaBottomEdge':
        self.px = detect_lamella(mask, self)
        return self.px

@dataclass
class LandingPost(Feature):
    feature_m: Point = None
    px: Point = None
    _color_UINT8: tuple = (255,255,255)
    color = "cyan"
    name: str = "LandingPost"

    def detect(self, img: np.ndarray, mask: np.ndarray = None, point:Point=None) -> 'LandingPost':
        self.px = detect_landing_post_v3(img, point)
        return self.px


@dataclass
class CoreFeature(Feature):
    feature_m: Point = None
    px: Point = None
    _color_UINT8: tuple = (50, 205, 50)
    color = "lime"
    name: str = "CoreFeature"

    def detect(self, img: np.ndarray, mask: np.ndarray = None, point:Point=None) -> 'CoreFature':
        self.px = detect_core_feature(mask, self)
        return self.px


__FEATURES__ = [ImageCentre, NeedleTip, LamellaCentre, LamellaLeftEdge, LamellaRightEdge, LandingPost, CoreFeature]
 


# Detection and Drawing Tools
def extract_class_pixels(mask, color):
    """ Extract only the pixels that are classified as the desired class (color)

    args:
        mask: detection mask containing all detection classes (np.array)
        color: the color of the specified class in the mask (rgb tuple)

    return:
        class_mask: the mask containing only the selected class
        idx: the indexes of the detected class in the mask

    """
    # TODO: should the class masks be binary?? probs easier

    # extract only label pixels to find edges
    class_mask = np.zeros_like(mask)
    idx = np.where(np.all(mask == color, axis=-1))
    class_mask[idx] = color

    return class_mask, idx


def detect_landing_post_v3(img: np.ndarray, landing_pt: Point = None, sigma=3) -> Point:
    if landing_pt is None:
        landing_pt = Point(x=img.shape[1] // 2, y=img.shape[0] // 2)
    edge = edge_detection(img, sigma=sigma)
    px = detect_closest_edge_v2(edge, landing_pt)
    return px


def detect_centre_point(mask: np.ndarray, threshold: int = 500) -> Point:
    """ Detect the centre (mean) point of the mask for a given color (label)

    args:
        mask: the detection mask (PIL.Image)
        idx: the index of the desired class in the mask (int)
        threshold: the minimum number of required pixels for a detection to count (int)

    return:

        centre_px: the pixel coordinates of the centre point of the feature (tuple)
    """
    centre_px = Point(x=0, y=0)
    # get mask px coordinates
    idx = np.where(mask)

    # only return a centre point if detection is above a threshold
    if len(idx[0]) > threshold:
        # get the centre point of each coordinate
        y_mid = int(np.mean(idx[0]))
        x_mid = int(np.mean(idx[1]))

        # centre coordinate as tuple
        centre_px = Point(x=x_mid, y=y_mid)
    return centre_px


def detect_corner(
    mask: np.ndarray, threshold=25, left: bool = False, bottom: bool = False
) -> Point:

    # get mask px coordinates
    edge_mask = np.where(mask)
    edge_px = (0, 0)

    # only return an edge point if detection is above a threshold
    if len(edge_mask[0]) > threshold:

        # right_most: max(x_coord), left_most: min(x_coord)
        px = np.max(edge_mask[1])
        if left:
            px = np.min(edge_mask[1])

        # get all px corresponding to value
        h_idx = np.where(edge_mask[1] == px)
        coords = edge_mask[0][h_idx], edge_mask[1][h_idx]

        # get the coordinates of the median vertical index
        py = int(np.median(coords[0]))
        edge_px = (py, px)

    return Point(x=int(edge_px[1]), y=int(edge_px[0]))


def detect_median_edge(mask: np.ndarray, edge: str, threshold: int = 250) -> Point:

    # get mask px coordindates
    edge_mask = np.where(mask)
    edge_px = Point(0, 0)

    if len(edge_mask[0]) > threshold:
        try:
            # get the centre point of each coordinate
            py = int(np.mean(edge_mask[0]))
            px = int(np.mean(edge_mask[1]))

            # get the min and max x and y coordinates at these median points
            x_min = np.min(edge_mask[1][edge_mask[0] == py])
            x_max = np.max(edge_mask[1][edge_mask[0] == py])
            y_min = np.min(edge_mask[0][edge_mask[1] == px])
            y_max = np.max(edge_mask[0][edge_mask[1] == px])


            if edge == "left":
                edge_px = Point(x=x_min, y=py)
            if edge == "right":
                edge_px = Point(x=x_max, y=py)
            if edge == "top":
                edge_px = Point(x=px, y=y_min)
            if edge == "bottom":
                edge_px = Point(x=px, y=y_max)
        except:
            pass
    return Point(x=int(edge_px.x), y=int(edge_px.y))

def detect_core_feature(
    mask: np.ndarray,
    feature: Feature,
    mask_radius: int = 512,
    idx: int = 1,
) -> Point:
    px = detect_centre_point(mask == idx, threshold=500)
    return px

def detect_lamella(
    mask: np.ndarray,
    feature: Feature,
    mask_radius: int = 512,
    idx: int = 1,
) -> Point:

    lamella_mask = mask == idx
    # lamella_mask = masks.apply_circular_mask(lamella_mask, radius=mask_radius)

    if isinstance(feature, LamellaCentre):
        px = detect_centre_point(lamella_mask)

    if isinstance(feature, LamellaLeftEdge):
        px = detect_corner(lamella_mask, left=True)

    if isinstance(feature, LamellaRightEdge):
        px = detect_corner(lamella_mask, left=False)


    if isinstance(feature, LamellaTopEdge):
        px = detect_median_edge(lamella_mask, edge="top")

    if isinstance(feature, LamellaBottomEdge):
        px = detect_median_edge(lamella_mask, edge="bottom")
    
    return px


def detect_needle_v4(mask: np.ndarray, idx:int=2) -> Point:
    needle_mask = mask == idx
    return detect_corner(needle_mask, threshold=100)


def edge_detection(img: np.ndarray, sigma=3) -> np.ndarray:
    return feature.canny(img, sigma=sigma)  # sigma higher usually better


def detect_closest_edge_v2(
    mask: np.ndarray, landing_pt: Point
) -> tuple[Point, np.ndarray]:
    """ Identify the closest edge point to the initially selected point

    args:
        img: base image (np.ndarray)
        landing_px: the initial landing point pixel (tuple) (y, x) format
    return:
        landing_edge_pt: the closest edge point to the intitially selected point (tuple)
        edges: the edge mask (np.array)
    """

    # identify edge pixels
    landing_px = (landing_pt.y, landing_pt.x)
    edge_mask = np.where(mask)
    edge_px = list(zip(edge_mask[0], edge_mask[1]))

    # set min distance
    min_dst = np.inf

    # TODO: vectorise this like
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html

    landing_edge_px = (0, 0)
    for px in edge_px:

        # distance between edges and landing point
        dst = distance.euclidean(landing_px, px)

        # select point with min
        if dst < min_dst:
            min_dst = dst
            landing_edge_px = px

    return Point(x=int(landing_edge_px[1]), y=int(landing_edge_px[0]))


def detect_bounding_box(mask, color, threshold=25):
    """ Detect the bounding edge points of the mask for a given color (label)

    args:
        mask: the detection mask (PIL.Image)
        color: the color of the label for the feature to detect (rgb tuple)
        threshold: the minimum number of required pixels for a detection to count (int)

    return:

        edge_px: the pixel coordinates of the edge points of the feature list((tuple))
    """

    top_px, bottom_px, left_px, right_px = (0, 0), (0, 0), (0, 0), (0, 0)

    # extract class pixels
    class_mask, idx = extract_class_pixels(mask, color)

    # only return an edge point if detection is above a threshold

    if len(idx[0]) > threshold:
        # convert mask to coordinates
        px = list(zip(idx[0], idx[1]))

        # get index of each value
        top_idx = np.argmin(idx[0])
        bottom_idx = np.argmax(idx[0])
        left_idx = np.argmin(idx[1])
        right_idx = np.argmax(idx[1])

        # pixel coordinates
        top_px = px[top_idx]
        bottom_px = px[bottom_idx]
        left_px = px[left_idx]
        right_px = px[right_idx]

    # bbox should be (x0, y0), (x1, y1)
    x0 = top_px[0]
    y0 = left_px[1]
    x1 = bottom_px[0]
    y1 = right_px[1]

    bbox = (x0, y0, x1, y1)

    return bbox

from typing import Union
### v2

@dataclass
class DetectedFeatures:
    features: list[Feature]
    image: np.ndarray # TODO: convert or add FIBSEMImage
    mask: np.ndarray # class binary mask
    rgb: np.ndarray # rgb mask
    pixelsize: float
    _distance: Point = Point(x=0, y=0)
    fibsem_image: FibsemImage = None

    @property
    def distance(self):
        assert len(self.features) >= 2, "Need at least two features to calculate distance"
        return self.features[0].px.distance(self.features[1].px)._to_metres(self.pixelsize)
        
    @distance.setter
    def distance(self, value: Point) -> None:
        self._distance = value

    def get_feature(self, ftype: Union[str, Feature]) -> Feature:
        for feature in self.features:
            if feature.name == ftype or isinstance(feature, type(ftype)):
                return feature

def detect_features_v2(
    img: np.ndarray, mask: np.ndarray, features: tuple[Feature], filter: bool = True, point: Point = None
) -> list[Feature]:

    detection_features = []

    for feature in features:
        
        if isinstance(feature, (LamellaCentre, LamellaLeftEdge, LamellaRightEdge, LamellaTopEdge, LamellaBottomEdge, CoreFeature)):
            feature = detect_multi_features(img, mask, feature)
            if filter:
                feature = filter_best_feature(mask, feature, 
                                              method="closest", 
                                              point=point)
        else:
            feature.detect(img=img,mask=mask) 

        if isinstance(feature, list):
            detection_features.extend(feature)
        else:
            detection_features.append(feature)

        
    return detection_features


def detect_features(
    image: Union[np.ndarray, FibsemImage],
    model: SegmentationModel,
    features: tuple[Feature],
    pixelsize: float,
    filter: bool = True,
    point: Point = None
) -> DetectedFeatures:

    if isinstance(image, FibsemImage):
        fibsem_image = deepcopy(image)
        image = image.data
    else:
        fibsem_image = None

    # model inference
    mask = model.inference(image, rgb=False)
    rgb = model.postprocess(mask, model.num_classes)
    mask = mask[0] # remove channel dim

    # detect features
    features = detect_features_v2(img=image, 
                                  mask=mask, 
                                  features=features, 
                                  filter=filter, point=point)

    det = DetectedFeatures(
        features=features, # type: ignore
        image=image,
        mask=mask,
        rgb=rgb,
        pixelsize=pixelsize,
        fibsem_image=fibsem_image
    )

    # distance in metres (from centre)
    for feature in det.features:
        feature.feature_m = conversions.image_to_microscope_image_coordinates(
            feature.px, det.image.data, det.pixelsize
        )

    return det


def take_image_and_detect_features(
    microscope: FibsemMicroscope,
    settings: MicroscopeSettings,
    features: tuple[Feature],
) -> DetectedFeatures:
    
    from fibsem import acquire, utils
    from fibsem.segmentation.model import load_model

    if settings.image.reduced_area is not None:
        logging.info(
            f"Reduced area is not compatible with model detection, disabling..."
        )
        settings.image.reduced_area = None
    
    settings.image.label = f"ml-{utils.current_timestamp_v2()}"
    settings.image.save = True

    # take new image
    image = acquire.new_image(microscope, settings.image)

    # load model
    ml_protocol = settings.protocol.get("ml", {})
    checkpoint = ml_protocol.get("checkpoint", None)
    encoder = ml_protocol.get("encoder", "resnet34")
    num_classes = int(ml_protocol.get("num_classes", 3))
    model = load_model(checkpoint=checkpoint, encoder=encoder, nc=num_classes)

    # detect features
    det = detect_features(
        deepcopy(image), model, features=features, pixelsize=image.metadata.pixel_size.x
    )
    return det



def plot_detection(det: DetectedFeatures):

    """Plotting image with detected features

    Args:
        det (DetectedFeatures): detected features type
        inverse (bool, optional): Inverses the colour of the centre crosshair of the feature. Defaults to True.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    plot_det(det, ax)


def plot_det(det: DetectedFeatures, ax: plt.Axes, title: str = "Prediction", show: bool = True):
    ax.imshow(det.image, cmap="gray")
    ax.imshow(det.rgb, alpha=0.3)
    ax.set_title(title)
    
    for f in det.features:
        ax.plot(f.px.x, f.px.y, 
                "o",  color=f.color, 
                markersize=5, markeredgecolor="w", 
                label=f.name)
    if len(det.features) < 10:
        ax.legend(loc="best")

    if len(det.features) == 2:
        # plot white line between features
        ax.plot([det.features[0].px.x, det.features[1].px.x],
                [det.features[0].px.y, det.features[1].px.y], 
                color="w", linestyle="--")

    if show:
        plt.show()


def plot_detections(dets: list[DetectedFeatures], titles: list[str] = None) -> plt.Figure:
    """Plotting image with detected features

    Args:
        det (DetectedFeatures): detected features type
        inverse (bool, optional): Inverses the colour of the centre crosshair of the feature. Defaults to True.
    """
    import matplotlib.pyplot as plt

    if titles is None:
        titles = [f"Prediction {i}" for i in range(len(dets))]

    fig, ax = plt.subplots(1, len(dets), figsize=(15, 7))

    for i, det in enumerate(dets):

        plot_det(det, ax[i], title=titles[i], show=False)
    
    plt.show()

    return fig


def move_based_on_detection(
    microscope: FibsemMicroscope,
    settings: MicroscopeSettings,
    det: DetectedFeatures,
    beam_type: BeamType,
    move_x: bool = True,
    move_y: bool = True,
):

    from fibsem import movement

    # nulify movements in unsupported axes
    if not move_x:
        det.distance.x = 0
    if not move_y:
        det.distance.y = 0

    # f1 = move from, f2 = move to
    f1 = det.features[0]
    f2 = det.features[1]

    logging.debug(f"move_x: {move_x}, move_y: {move_y}")
    logging.debug(f"movement: x={det.distance.x:.2e}, y={det.distance.y:.2e}")
    logging.debug(f"features: {f1}, {f2}, beam_type: {beam_type}")

    # these movements move the needle...
    if isinstance(f1, NeedleTip) or isinstance(f1, LamellaRightEdge):

        # electron: neg = down, ion: neg = up
        if beam_type is BeamType.ION:
            det.distance.y *= -1

        microscope.move_manipulator_corrected(
            dx=det.distance.x,
            dy=det.distance.y,
            beam_type=beam_type,
        )

    if isinstance(f1, LamellaCentre) and isinstance(f2, ImageCentre):


            # need to reverse the direction to move correctly. investigate if this is to do with scan rotation?
            microscope.stable_move(
                settings=settings,
                dx=-det.distance.x,
                dy=det.distance.y,
                beam_type=beam_type,
            )

            # TODO: support other movements?
    return


import glob
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import tifffile as tff

from fibsem.segmentation import utils as seg_utils
from fibsem.structures import FibsemImage, Point, FibsemStagePosition


def _det_from_df(df: pd.DataFrame, path: Path, fname: str) -> Optional[DetectedFeatures]:
        
    # glob for the image in the path
    img_fname = glob.glob(os.path.join(path, f"{fname}.*"))
    mask_fname = glob.glob(os.path.join(path, "mask", f"{fname}.*"))

    if img_fname == [] or mask_fname == []:
        return None

    img = FibsemImage.load(img_fname[0])
    mask = tff.imread(mask_fname[0])

    df_filt = df[df["image"] == fname]

    def _from_df(df: pd.DataFrame) -> list[Feature]:

        features = []
        for feat_name in df["feature"].unique():
            
            # create feature from name
            idx = [i for i, feat in enumerate(__FEATURES__) if feat.__name__ == feat_name][0]
            feature = __FEATURES__[idx](
                px=Point(x=df[df["feature"] == feat_name]["p.x"].values[0],
                        y=df[df["feature"] == feat_name]["p.y"].values[0]),
            )

            features.append(feature)


        return features

    features = _from_df(df_filt)
    det = DetectedFeatures(features=features, 
                        image=img.data, 
                        mask=mask, 
                        rgb=seg_utils.decode_segmap(mask),
                        pixelsize=df_filt["pixelsize"].values[0],
                        )
    
    return det




def mask_contours(image):
    # Find contours
    contours = measure.find_contours(image, 0.5)

    # Create a mask with the same shape as the input image
    mask = np.zeros_like(image, dtype=np.uint8)

    # Mask the area inside each contour
    for i, contour in enumerate(contours):
        # Convert the contour coordinates to integers
        contour = np.round(contour).astype(int)

        # Create a polygonal mask for this contour
        rr, cc = np.meshgrid(np.arange(mask.shape[0]), np.arange(mask.shape[1]), indexing='ij')
        inside_contour = np.zeros_like(mask, dtype=np.uint8)
        inside_contour[rr, cc] = 0
        inside_contour[contour[:, 0], contour[:, 1]] = i+1

        # mask the area inside the contour
        ymin, ymax = np.min(contour[:, 0]), np.max(contour[:, 0])
        xmin, xmax = np.min(contour[:, 1]), np.max(contour[:, 1])

        inside_contour[ymin:ymax, xmin:xmax] = i+1

        # Apply the mask for this contour
        mask = mask + inside_contour

    return mask

from copy import deepcopy
# TODO: need passthrough for the params
def detect_multi_features(image: np.ndarray, mask: np.ndarray, feature: Feature, class_idx: int = 1):
    
    mask = mask == class_idx # filter to class 
    mask = mask_contours(mask)
    idxs = np.unique(mask)

    features = []
    for idx in idxs:
        if idx==0:
            continue

        # create a new image
        feature_mask = np.zeros_like(mask)
        feature_mask[mask==idx] = 1

        # detect features
        feature.detect(image, feature_mask)
        features.append(deepcopy(feature))

    return features


def filter_best_feature(mask: np.ndarray, features: list[Feature], method: str = "closest", point: Point = None):
    if method == "closest":
        # plot feature closest to point
        if point is None:
            point = Point(mask.shape[1]/2, mask.shape[0]/2)

        distances = []
        for feature in features:
            distances.append(np.linalg.norm(point - feature.px))
        idx = np.argmin(distances)

        return features[idx]

    else:
        raise ValueError(f"method {method} not recognised")

def get_feature(name: str) -> Feature:

    idx = [i for i, feat in enumerate(__FEATURES__) if feat.__name__ == name][0]
    feature = __FEATURES__[idx]()

    return feature



# v4 intersection features
from fibsem.microscope import FibsemMicroscope, MicroscopeSettings
import numpy as np
from fibsem.imaging import _tile

def _detect_positions(microscope: FibsemMicroscope, settings: MicroscopeSettings, image: FibsemImage, mask:np.ndarray, features: list[Feature]) -> list[FibsemStagePosition]:

    # detect features
    features = detect_features_v2(img=image, 
                                mask=mask, 
                                features=features, 
                                filter=False, point=None)

    # convert image coordinates to microscope coordinates # TODO: check why we reverse the points axis?
    positions = _tile._convert_image_coords_to_positions(microscope, settings, image, [Point(f.px.y, f.px.x) for f in features])

    return positions, features

def _calculate_intersection(masks: list[np.ndarray]) -> np.ndarray:

    intersection = masks[0]
    for mask in masks[1:]:
        intersection = intersection + mask
        intersection[intersection < 2] = 0
        intersection[intersection == 2] = 1
    return intersection