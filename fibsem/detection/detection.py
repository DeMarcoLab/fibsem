#!/usr/bin/env python3


import logging

import numpy as np
import PIL
import scipy.ndimage as ndi
from autoscript_sdb_microscope_client.structures import AdornedImage
from fibsem import calibration
from fibsem.structures import Point
from fibsem.detection import utils as det_utils
from fibsem.detection.utils import (DetectionFeature, DetectionResult,
                                     DetectionType)
from scipy.spatial import distance
from skimage import feature

# TODO:
# rename DetectionResult to Detection

# TODO: determine if there is a better way to figure out where the needle is without looking at it...
# do the coordinates helP?
# would be better to move into centre image.
# probably need to do full transformation matrix to get it working properly.. long temr


DETECTION_COLOURS_UINT8 = {
    DetectionType.ImageCentre: (255, 255, 255),
    DetectionType.LamellaCentre: (255, 0, 0),
    DetectionType.LamellaEdge: (255, 165, 0),
    DetectionType.NeedleTip: (0, 255, 0),
    DetectionType.LandingPost: (255, 255, 255),
}

def detect_features(img: AdornedImage, ref_image:AdornedImage, features: tuple[DetectionFeature]) -> list[DetectionFeature]:
    """

    args:
        img: the input img (AdornedImage)
        ref_image: the reference image (AdornedImage)
        shift_type: the type of feature detection to run (tuple)

    return:
        detection_features [DetectionFeature, DetectionFeature]: the detected feature coordinates and types
    """

    detection_features = []

    for feature in features:
        
        det_type = feature.detection_type
        initial_point = feature.feature_px

        if not isinstance(det_type, DetectionType):
            raise TypeError(f"Detection Type {det_type} is not supported.")

        # get the initial position estimate
        if initial_point is None:
            initial_point = get_initial_position(img, det_type)

        if det_type == DetectionType.ImageCentre:
            feature_px = initial_point

        if det_type == DetectionType.NeedleTip:
            feature_px = detect_needle_tip_v3(img, initial_point)

        if det_type == DetectionType.LamellaCentre:
            feature_px = initial_point # TODO

        if det_type == DetectionType.LamellaEdge:
            feature_px = detect_lamella_edge(img)

        if det_type == DetectionType.LandingPost:
            feature_px = detect_landing_post_v2(img, initial_point)

        detection_features.append(
            DetectionFeature(detection_type=det_type, feature_px=feature_px)
        )

    return detection_features

def locate_shift_between_features(adorned_img: AdornedImage, ref_image: AdornedImage, features: tuple[DetectionFeature]):
    """
    Calculate the distance between two features in the image coordinate system.

    args:
        adorned_img: input image (AdornedImage)
        ref_img: the reference image to align (AdornedImage)
        shift_type: the type of feature detection shift to calculation

    return:
        detection_result (DetectionResult): The detection result containing the feature coordinates, and images

    """
    # detect features for calculation
    feature_1, feature_2 = detect_features(adorned_img, ref_image, features)

    # calculate movement distance
    x_distance_m, y_distance_m = det_utils.convert_pixel_distance_to_metres(
        feature_1.feature_px, feature_2.feature_px, adorned_img
    )

    detection_result = DetectionResult(
        features=[feature_1, feature_2],
        adorned_image=adorned_img,
        display_image=adorned_img.data,
        distance_metres=Point(x_distance_m, y_distance_m),
        microscope_coordinate=[Point(0, 0), Point(0, 0)],
    )

    return detection_result

def get_initial_position(img: AdornedImage, det_type: DetectionType) -> Point:

    beam_type = img.metadata.acquisition.beam_type

    if det_type in [DetectionType.ImageCentre, DetectionType.LamellaCentre, DetectionType.LandingPost, DetectionType.NeedleTip]:
        point = Point(x=img.data.shape[1] // 2, y=img.data.shape[0] // 2)  # midpoint

    if det_type == DetectionType.LamellaEdge:
        point = Point(0, 0)

    # if det_type == DetectionType.NeedleTip:
        
    #     if beam_type == "Electron": 
    #         pt = (400, 200) 
    #     if beam_type == "Ion":
    #         pt = (800, 400) 

    #     point = Point(x=pt[0], y=pt[1])

    logging.info(f"{det_type}: {point}")
   
    return point

def filter_selected_masks(mask: np.ndarray, shift_type: tuple[DetectionType]) -> np.ndarray:
    """Combine only the masks for the selected detection types"""
    c1 = DETECTION_COLOURS_UINT8[shift_type[0]]
    c2 = DETECTION_COLOURS_UINT8[shift_type[1]]
    
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
    # extract only label pixels to find edges
    class_mask = np.zeros_like(mask)
    idx = np.where(np.all(mask == color, axis=-1))
    class_mask[idx] = color

    return class_mask, idx

def draw_overlay(img: np.ndarray, mask: np.ndarray, alpha:float=0.2) -> np.ndarray:
    """ Draw the detection overlay onto base image. Required to blend mixed grayscale / rgb images

    args:
        img: orignal image (np.array or PIL.Image)
        mask: detection mask (np.array or PIL.Image)

    returns:
        alpha_blend: mask overlaid with original image (PIL.Image)

    """

    # convert to PIL Image from np.array
    if isinstance(mask, np.ndarray):
        mask = PIL.Image.fromarray(mask)
    if isinstance(img, np.ndarray):
        img = PIL.Image.fromarray(img)

    # resize to same size as mask
    img = img.resize((mask.size[0], mask.size[1]), resample=PIL.Image.BILINEAR)

    # required for blending
    img = img.convert("RGB")
    mask = mask.convert("RGB")

    # blend images together
    alpha_blend = PIL.Image.blend(img, mask, alpha)

    return np.array(alpha_blend)

def detect_lamella_edge(img:AdornedImage):
    
    beam_type = img.metadata.acquisition.beam_type

    if beam_type == "Electron":
        pt = Point(x=int(img.data.shape[1] // 2.4), y=int(img.data.shape[0]*0.47)) # eb mask
    if beam_type == "Ion":
        pt = Point(x=int(img.data.shape[1] // 2.2), y=int(img.data.shape[0]*0.3)) # ib mask

    # ib mask
    mask = np.zeros_like(img.data)
    mask[pt.y:, :pt.x] = 1
    
    edge = edge_detection(img.data, sigma=3)  
    edge_mask = edge * mask
    lamella_edge = detect_right_edge_v2(edge_mask)

    return lamella_edge


def detect_needle_tip_v2(ref_image: AdornedImage, new_image:AdornedImage, initial_point: Point = Point(400, 200) ) -> Point:

    minus = calibration.normalise_image(ref_image) - calibration.normalise_image(new_image)

    filt = ndi.filters.gaussian_filter(minus, sigma=12)

    edge = edge_detection(filt, sigma=3)  # edges

    # mask after edge detection
    mask = np.zeros_like(ref_image.data)
    
    beam_type = ref_image.metadata.acquisition.beam_type
    if beam_type == "Electron":
        pt = Point(x=int(ref_image.data.shape[1] * 0.3), y=int(ref_image.data.shape[0]*0.3)) # eb mask, top left corner
        mask[:pt.y, :pt.x] = 1
    if beam_type == "Ion":
        pt = Point(x=int(ref_image.data.shape[1] * 0.5), y=int(ref_image.data.shape[0]*0.5)) # ib mask, bottom, left corner
        mask[pt.y:, :pt.x] = 1
        
    mask = np.ones_like(ref_image.data)
    edge_mask = edge * mask

    if beam_type == "Electron":
        needle = detect_closest_edge_v2(edge_mask, initial_point)  # closest edge  
    if beam_type == "Ion":
        needle = detect_right_edge_v2(edge_mask)   # right most edge
    
    # centre = Point(x=ref_image.data.shape[1]//2, y=ref_image.data.shape[0]//2)
    # top_needle = detection.detect_closest_edge_v2(edge_mask, Point(x=400, y=200))  # closest edge  
    # bottom_needle = detection.detect_closest_edge_v2(edge_mask, centre)  # closest edge  
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 3, figsize=(30, 30))
    # ax[0].imshow(filt, cmap="gray")
    # ax[1].imshow(minus * mask, cmap="gray")
    # ax[2].imshow(edge_mask, cmap="turbo", )
    # plt.show()

    return needle 

def detect_needle_tip_v3(image:AdornedImage, initial_point: Point = None) -> Point:

    edge = edge_detection(image.data, sigma=3)  # edges

    # needle = detect_closest_edge_v2(edge, initial_point)
    needle = detect_right_edge_v2(edge)   # right most edge
    # TODO: top point detection

    return needle

def detect_landing_post_v2(img: AdornedImage, landing_pt: Point) -> Point:
    landing_pt = Point(x=img.data.shape[1] // 2, y=img.data.shape[0] // 2)
    edge = edge_detection(img.data, sigma=3)
    feature_px = detect_closest_edge_v2(edge, landing_pt)
    return feature_px


def detect_centre_point(mask: np.ndarray, color: tuple, threshold:int=25) -> Point:
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
        centre_px = Point(x=x_mid, y=y_mid )

    return centre_px

def detect_right_edge_v2(mask: np.ndarray, threshold=25, left=False) -> Point:

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
        idx = np.where(edge_mask[1] == px)
        if left:
            idx = np.where(edge_mask[1]==px)
        coords = edge_mask[0][idx], edge_mask[1][idx]

        # get top value  (most vertical) to break ties
        top_idx = np.argmax(coords[1])

        edge_px = coords[0][top_idx], coords[1][top_idx]

    return Point(x=edge_px[1], y=edge_px[0])


def edge_detection(img: np.ndarray, sigma=3) -> np.ndarray:
    return feature.canny(img, sigma=sigma)  # sigma higher usually better


def detect_closest_edge_v2(mask: np.ndarray, landing_pt: Point) -> tuple[Point, np.ndarray]:
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

