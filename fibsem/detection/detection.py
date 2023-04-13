#!/usr/bin/env python3
import logging
from dataclasses import dataclass
from enum import Enum
import numpy as np

from fibsem.microscope import FibsemMicroscope
from scipy.spatial import distance
from skimage import feature

from fibsem import conversions
# from fibsem.detection.utils import Feature, FeatureType
from fibsem.imaging import masks
from fibsem.segmentation.model import SegmentationModel
from fibsem.structures import BeamType, MicroscopeSettings, Point
from abc import ABC, abstractmethod


@dataclass
class Feature(ABC):
    feature_px: Point 
    feature_m: Point
    _color_UINT8: None
    name: str = None

    @abstractmethod
    def detect(self, img: np.ndarray, mask: np.ndarray=None, point:Point=None) -> 'Feature':
        pass

@dataclass
class ImageCentre(Feature):
    feature_m: Point = None
    feature_px: Point = None
    _color_UINT8: tuple = (255,255,255)
    name: str = "ImageCentre"

    def detect(self, img: np.ndarray, mask: np.ndarray=None, point:Point=None) -> 'ImageCentre':
        self.feature_px = Point(x=img.shape[1] / 2, y=img.shape[0] / 2)
        return self.feature_px

    

@dataclass
class NeedleTip(Feature):
    feature_m: Point = None
    feature_px: Point = None
    _color_UINT8: tuple = (0,255,0)
    name: str = "NeedleTip"

    def detect(self, img: np.ndarray, mask: np.ndarray = None, point:Point=None) -> 'NeedleTip':
        self.feature_px = detect_needle_v4(mask)
        return self.feature_px
    
    

@dataclass
class LamellaCentre(Feature):
    feature_m: Point = None
    feature_px: Point = None
    _color_UINT8: tuple = (255,0,0)
    name: str = "LamellaCentre"

    def detect(self, img: np.ndarray, mask: np.ndarray = None, point:Point=None) -> 'LamellaCentre':
        self.feature_px = detect_lamella(mask, self.name)

    
@dataclass
class LamellaLeftEdge(Feature):
    feature_m: Point = None
    feature_px: Point = None
    _color_UINT8: tuple = (255,0,0)
    name: str = "LamellaLeftEdge"

    def detect(self, img: np.ndarray, mask: np.ndarray = None, point:Point=None) -> 'LamellaLeftEdge':
        self.feature_px = detect_lamella(mask, self.name)
        return self.feature_px
    

@dataclass
class LamellaRightEdge(Feature):
    feature_m: Point = None
    feature_px: Point = None
    _color_UINT8: tuple = (255,0,255)
    name: str = "LamellaRightEdge"

    def detect(self, img: np.ndarray, mask: np.ndarray = None, point:Point=None) -> 'LamellaRightEdge':
        self.feature_px = detect_lamella(mask, self.name)
        return self.feature_px
    

@dataclass
class LandingPost(Feature):
    feature_m: Point = None
    feature_px: Point = None
    _color_UINT8: tuple = (255,255,255)
    name: str = "LandingPost"

    def detect(self, img: np.ndarray, mask: np.ndarray = None, point:Point=None) -> 'LandingPost':
        self.feature_px = detect_landing_post_v3(img, point)
        return self.feature_px
    

__FEATURES__ = [ImageCentre, NeedleTip, LamellaCentre, LamellaLeftEdge, LamellaRightEdge, LandingPost]

 


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
    feature_px = detect_closest_edge_v2(edge, landing_pt)
    return feature_px


def detect_centre_point(mask: np.ndarray, color: tuple, threshold: int = 25) -> Point:
    """ Detect the centre (mean) point of the mask for a given color (label)

    args:
        mask: the detection mask (PIL.Image)
        color: the color of the label for the feature to detect (rgb tuple)
        threshold: the minimum number of required pixels for a detection to count (int)

    return:

        centre_px: the pixel coordinates of the centre point of the feature (tuple)
    """
    centre_px = Point(x=0, y=0)

    # extract class pixels
    class_mask, idx = extract_class_pixels(mask, color)

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

        # get vertical index
        v_idx = np.argmin(coords[0])
        if bottom:
            v_idx = np.argmax(coords[0])

        edge_px = coords[0][v_idx], coords[1][v_idx]

    return Point(x=edge_px[1], y=edge_px[0])


def detect_lamella(
    mask: np.ndarray,
    feature_type: str,
    color: tuple = LamellaCentre._color_UINT8,
    mask_radius: int = 512,
) -> Point:

    lamella_mask, _ = extract_class_pixels(mask, color)
    lamella_mask = masks.apply_circular_mask(lamella_mask, radius=mask_radius)
    lamella_centre = detect_centre_point(lamella_mask, color=color)

    if feature_type == "LamellaCentre":
        feature_px = detect_centre_point(lamella_mask, color=color)

    if feature_type == "LamellaLeftEdge":
        feature_px = detect_corner(lamella_mask, left=True)

    if feature_type == "LamellaRightEdge":
        feature_px = detect_corner(lamella_mask, left=False)

    return feature_px


def detect_needle_v4(mask: np.ndarray,) -> Point:
    needle_mask, _ = extract_class_pixels(mask, NeedleTip._color_UINT8)
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

    return Point(x=landing_edge_px[1], y=landing_edge_px[0])


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


### v2


@dataclass
class DetectedFeatures:
    features: list[Feature]
    image: np.ndarray
    mask: np.ndarray
    pixelsize: float
    distance: Point


def detect_features_v2(
    img: np.ndarray, mask: np.ndarray, features: tuple[Feature]
) -> list[Feature]:

    detection_features = []

    for feature in features:
        
        feature.detect(img=img,mask=mask) 

        detection_features.append(feature)

    return detection_features


def locate_shift_between_features_v2(
    image: np.ndarray,
    model: SegmentationModel,
    features: tuple[Feature],
    pixelsize: float,
) -> DetectedFeatures:

    # model inference
    mask = model.inference(image)

    # detect features
    feature_1, feature_2 = detect_features_v2(image, mask, features)

    # calculate distance between features
    distance_px = conversions.distance_between_points(
        feature_1.feature_px, feature_2.feature_px
    )
    distance_m = conversions.convert_point_from_pixel_to_metres(distance_px, pixelsize)

    det = DetectedFeatures(
        features=[feature_1, feature_2],
        image=image,
        mask=mask,
        distance=distance_m,
        pixelsize=pixelsize,
    )

    return det


def plot_det_result_v2(det: DetectedFeatures,inverse: bool = True ):
    """Plotting image with detected features

    Args:
        det (DetectedFeatures): detected features type
        inverse (bool, optional): Inverses the colour of the centre crosshair of the feature. Defaults to True.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 7))

    # convert rgb 255 range to 0-1 tuple
    if inverse:
        # plotting crosshairs are contrasted against feature
        c1 = ((255-det.features[0]._color_UINT8[0])/255,
            (255-det.features[0]._color_UINT8[1])/255,
            (255-det.features[0]._color_UINT8[2])/255)
        
        c2 = ((255-det.features[1]._color_UINT8[0])/255,
            (255-det.features[1]._color_UINT8[1])/255,
            (255-det.features[1]._color_UINT8[2])/255)
        
    else:

        c1 = ((det.features[0]._color_UINT8[0])/255,
            (det.features[0]._color_UINT8[1])/255,
            (det.features[0]._color_UINT8[2])/255)
        
        c2 = ((det.features[1]._color_UINT8[0])/255,
            (det.features[1]._color_UINT8[1])/255,
            (det.features[1]._color_UINT8[2])/255)
        
        

    ax[0].imshow(det.image, cmap="gray")
    ax[0].set_title(f"Image")
    ax[1].imshow(det.mask)
    plt.imsave("mask.tiff",det.mask)
    ax[1].set_title("Prediction")
    ax[1].plot(
        det.features[0].feature_px.x,
        det.features[0].feature_px.y,
        color=c1,
        marker="+",
        ms=20,
        label=det.features[0].name,
    )
    ax[1].plot(
        det.features[1].feature_px.x,
        det.features[1].feature_px.y,
        color=c2,
        marker="+",
        ms=20,
        label=det.features[1].name,
    )
    ax[1].plot(
        [det.features[0].feature_px.x, det.features[1].feature_px.x],
        [det.features[0].feature_px.y, det.features[1].feature_px.y],
        "w--",
    )
    ax[1].legend(loc="best")
    plt.show()




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
    if f1.name in ["NeedleTip", "LamellaRightEdge"]:

        # electron: neg = down, ion: neg = up
        if beam_type is BeamType.ELECTRON:
            det.distance.y *= -1

        microscope.move_manipulator_corrected(
            dx=det.distance.x,
            dy=det.distance.y,
            beam_type=beam_type,
        )

    if f1.name == "LamellaCentre" and f2.name == "ImageCentre":

            # need to reverse the direction to move correctly. investigate if this is to do with scan rotation?
            microscope.stable_move(
                settings=settings,
                dx=-det.distance.x,
                dy=-det.distance.y,
                beam_type=beam_type,
            )

            # TODO: support other movements?
    return

