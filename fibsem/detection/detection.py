#!/usr/bin/env python3
import logging
from dataclasses import dataclass

import numpy as np
from autoscript_sdb_microscope_client import SdbMicroscopeClient
from scipy.spatial import distance
from skimage import feature

from fibsem import conversions
from fibsem.detection.utils import Feature, FeatureType
from fibsem.imaging import masks
from fibsem.segmentation.model import SegmentationModel
from fibsem.structures import BeamType, MicroscopeSettings, Point

FEATURE_COLOURS_UINT8 = {
    FeatureType.ImageCentre: (255, 255, 255),
    FeatureType.LamellaCentre: (255, 0, 0),
    FeatureType.LamellaLeftEdge: (255, 0, 0),
    FeatureType.LamellaRightEdge: (255, 0, 0),
    FeatureType.NeedleTip: (0, 255, 0),
    FeatureType.LandingPost: (255, 255, 255),
}


def filter_selected_masks(
    mask: np.ndarray, shift_type: tuple[FeatureType]
) -> np.ndarray:
    """Combine only the masks for the selected detection types"""
    c1 = FEATURE_COLOURS_UINT8[shift_type[0]]
    c2 = FEATURE_COLOURS_UINT8[shift_type[1]]

    # get mask for first detection type
    mask1, _ = extract_class_pixels(mask, color=c1)
    # get mask for second detection type
    mask2, _ = extract_class_pixels(mask, color=c2)

    # combine masks
    mask_combined = mask1 + mask2

    return mask_combined


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
    feature_type: FeatureType,
    color: tuple = (255, 0, 0),
    mask_radius: int = 512,
) -> Point:

    lamella_mask, _ = extract_class_pixels(mask, color)
    lamella_mask = masks.apply_circular_mask(lamella_mask, radius=mask_radius)
    lamella_centre = detect_centre_point(lamella_mask, color=color)

    if feature_type is FeatureType.LamellaCentre:
        feature_px = detect_centre_point(lamella_mask, color=color)

    if feature_type is FeatureType.LamellaLeftEdge:
        feature_px = detect_corner(lamella_mask, left=True)

    if feature_type is FeatureType.LamellaRightEdge:
        feature_px = detect_corner(lamella_mask, left=False)

    return feature_px


def detect_needle_v4(mask: np.ndarray,) -> Point:
    needle_mask, _ = extract_class_pixels(mask, (0, 255, 0))
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

        det_type = feature.type
        initial_point = feature.feature_px

        if not isinstance(det_type, FeatureType):
            raise TypeError(f"Detection Type {det_type} is not supported.")

        # get the initial position estimate
        if initial_point is None:
            initial_point = Point(x=img.shape[1] // 2, y=img.shape[0] // 2)

        if det_type == FeatureType.ImageCentre:
            feature_px = initial_point

        if det_type == FeatureType.NeedleTip:
            feature_px = detect_needle_v4(mask)

        if det_type in [
            FeatureType.LamellaCentre,
            FeatureType.LamellaLeftEdge,
            FeatureType.LamellaRightEdge,
        ]:
            feature_px = detect_lamella(mask, det_type)

        if det_type == FeatureType.LandingPost:
            feature_px = detect_landing_post_v3(img, initial_point)

        detection_features.append(Feature(type=det_type, feature_px=feature_px))

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


def plot_det_result_v2(det: DetectedFeatures):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 7))
    ax[0].imshow(det.image, cmap="gray")
    ax[0].set_title(f"Image")
    ax[1].imshow(det.mask)
    ax[1].set_title("Prediction")
    ax[1].plot(
        det.features[0].feature_px.x,
        det.features[0].feature_px.y,
        "g+",
        ms=20,
        label=det.features[0].type.name,
    )
    ax[1].plot(
        det.features[1].feature_px.x,
        det.features[1].feature_px.y,
        "w+",
        ms=20,
        label=det.features[1].type.name,
    )
    ax[1].plot(
        [det.features[0].feature_px.x, det.features[1].feature_px.x],
        [det.features[0].feature_px.y, det.features[1].feature_px.y],
        "w--",
    )
    ax[1].legend(loc="best")
    plt.show()




def move_based_on_detection(
    microscope: SdbMicroscopeClient,
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
    if f1.type in [FeatureType.NeedleTip, FeatureType.LamellaRightEdge]:

        # electron: neg = down, ion: neg = up
        if beam_type is BeamType.ELECTRON:
            det.distance.y *= -1

        movement.move_needle_relative_with_corrected_movement(
            microscope=microscope,
            dx=det.distance.x,
            dy=det.distance.y,
            beam_type=beam_type,
        )

    if f1.type is FeatureType.LamellaCentre:
        if f2.type is FeatureType.ImageCentre:

            # need to reverse the direction to move correctly. investigate if this is to do with scan rotation?
            movement.move_stage_relative_with_corrected_movement(
                microscope=microscope,
                settings=settings,
                dx=-det.distance_metres.x,
                dy=-det.distance_metres.y,
                beam_type=beam_type,
            )

            # TODO: support other movements?
    return



# convert to grayscale mask
def convert_rgb_to_binary_mask(rgb):
    mask = np.sum(rgb, axis=2).astype(bool)

    return mask

def get_mask_point(arr: np.ndarray, hor: str = "centre", vert: str = "centre") -> Point:
    """Get a point from a mask"""

    # get all pixels equal to 1
    if arr.ndim == 3:
        arr = convert_rgb_to_binary_mask(arr)
    
    mask = arr == 1

    # get the x and y coordinates of the pixels
    x, y = np.where(mask)

    # get the bounding box
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)

    # get the width and height of the bounding box
    w = (xmax - xmin)
    h = (ymax - ymin)

    # get the centre of the bounding box
    xc = xmin + w // 2
    yc = ymin + h // 2

    if hor == "left":
        px = xmin
    if hor == "right":
        px = xmax
    if hor == "centre":
        px = xc

    if vert == "upper":
        py = ymin
    if vert == "lower":
        py = ymax
    if vert =="centre":
        py = yc

    return Point(px, py)