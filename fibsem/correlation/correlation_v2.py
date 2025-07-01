import contextlib
import datetime
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

from fibsem.correlation.pyto.rigid_3d import Rigid3D # NOTE: this is still a 3DCT dependency, migrate

DEFAULT_OPTIMIZATION_PARAMETERS = {
    'random_rotations': True,
    'rotation_init': 'gl2',
    'restrict_rotations': 0.1,
    'scale': None,
    'random_scale': True,
    'scale_init': 'gl2',
    'ninit': 10
}

def correlate(
    markers_3d: np.ndarray,
    markers_2d: np.ndarray,
    poi_3d: np.ndarray,
    rotation_center: List[float],
    imageProps: list = None,
    optimiser_params: Dict = DEFAULT_OPTIMIZATION_PARAMETERS
) -> dict:
    """
    Iteratively calculate the correlation between 3D and 2D markers and reproject the points of interest (POI) into the 2D image

    Args: 
        markers_3d: array of correlation marker positions for 3D image
        markers_2d: array of correlation marker positions for 2D image
        poi_3d:     array of points of interest for 3D image
        rotation_center: center of rotation for the 3D image (x,y,z)
        imageProps: properties of the images
            ([2d_image_shape, 2d_image_pixel_size_um, 3d_image_shape])
        optimiser_params: dictionary with optimization parameters
            {
                'random_rotations': bool,      # random rotations
                'rotation_init': float,        # initial rotation in degrees
                'restrict_rotations': float,   # restrict rotations
                'scale': float,                # scale
                'random_scale': bool,          # random scale
                'scale_init': float,           # initial scale
                'ninit': float                 # number of iterations
            }
    Returns:
        Dictionary with input and output data:
            input: {
                "markers_3d": np.ndarray[float],    # 3D marker positions
                "markers_2d": np.ndarray[float],    # 2D marker positions
                "poi_3d": np.ndarray[float],        # 3D point of interest positions
                "rotation_center": list[float],     # center of rotation for the 3D image
                "imageProps": list                  # properties of the images
            },
            output: {
                "transform": Rigid3D,                               # transformation object
                "reprojected_3d_coordinates": np.ndarray[float],    # reprojected 3D marker positions in 2D image
                "reprojected_2d_poi": np.ndarray[float],            # reprojected 3D poi in 2D image
                "reprojection_error": np.ndarray[float],            # reprojection error between reprojected 3D markers and 2D markers
                "center_of_mass_3d_markers": list[float],           # center of mass of 3D markers
                "modified_translation": list[float]                 # modified translation (rotation center not at 0,0,0)

    """
    # TODO: convert imageProps to a dataclass or dict?
    
    # read optimization parameters
    random_rotations = optimiser_params.get('random_rotations', DEFAULT_OPTIMIZATION_PARAMETERS['random_rotations'])
    rotation_init = optimiser_params.get('rotation_init', DEFAULT_OPTIMIZATION_PARAMETERS['rotation_init'])
    restrict_rotations = optimiser_params.get('restrict_rotations', DEFAULT_OPTIMIZATION_PARAMETERS['restrict_rotations'])
    scale = optimiser_params.get('scale', DEFAULT_OPTIMIZATION_PARAMETERS['scale'])
    random_scale = optimiser_params.get('random_scale', DEFAULT_OPTIMIZATION_PARAMETERS['random_scale'])
    scale_init = optimiser_params.get('scale_init', DEFAULT_OPTIMIZATION_PARAMETERS['scale_init'])
    ninit: float = optimiser_params.get('ninit', DEFAULT_OPTIMIZATION_PARAMETERS['ninit'])
    
    assert markers_3d.shape[1] == 3, "Markers 3D do not have 3 dimensions"
    
    # coordinate arrays
    mark_3d = markers_3d.T          # fm markers (3D)
    mark_2d = markers_2d[:,:2].T    # fib markers (2D) 
    poi_3d = poi_3d.T               # points of interest (3D)
    
    # convert Eulers in degrees to Caley-Klein params
    if (rotation_init is not None) and (rotation_init != 'gl2'):
        rotation_init_rad = rotation_init * np.pi / 180
        einit = Rigid3D.euler_to_ck(angles=rotation_init_rad, mode='x')
    else:
        einit = rotation_init

    # establish correlation
    # Suppress stdout and stderr
    with open(os.devnull, 'w') as fnull, contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
        transf = Rigid3D.find_32(
            x=mark_3d, y=mark_2d, scale=scale,
            randome=random_rotations, einit=einit, einit_dist=restrict_rotations,
            randoms=random_scale, sinit=scale_init, ninit=ninit)

    if imageProps:
        
        # establish correlation for cubic rotation (offset added to coordinates)
        shape_2d, pixel_size, shape_3d = imageProps
        offset = (max(shape_3d) - np.array(shape_3d)) * 0.5

        mark_3d_cube = np.copy(mark_3d) + offset[::-1, np.newaxis]
        # Suppress stdout and stderr
        with open(os.devnull, 'w') as fnull, contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
            transf_cube = Rigid3D.find_32(
                x=mark_3d_cube, y=mark_2d, scale=scale,
                randome=random_rotations, einit=einit, einit_dist=restrict_rotations,
                randoms=random_scale, sinit=scale_init, ninit=ninit)
    else:
        transf_cube = transf

    # reproject_points of interest
    reprojected_poi_2d = None
    if len(poi_3d) > 0:
        reprojected_poi_2d = transf.transform(x=poi_3d)

    # transform markers
    reprojected_coordinates_3d = transf.transform(x=mark_3d)

    # calculate translation if rotation center is not at (0,0,0)
    modified_translation = transf_cube.recalculate_translation(
        rotation_center=rotation_center)

    # center of mass of 3D markers
    cm_3D_markers = mark_3d.mean(axis=-1).tolist()

    # delta calc,real
    reprojection_error = reprojected_coordinates_3d[:2,:] - mark_2d

    return {
        "input": {
            "markers_3d": mark_3d,
            "markers_2d": mark_2d,
            "poi_3d": poi_3d,
            "rotation_center": rotation_center,
            "imageProps": imageProps,
        },
        "output": {
            "transform": transf,
            "reprojected_3d_coordinates": reprojected_coordinates_3d,
            "reprojected_2d_poi": reprojected_poi_2d,
            "reprojection_error": reprojection_error,
            "center_of_mass_3d_markers": cm_3D_markers,
            "modified_translation": modified_translation,
        }
    }


def save_results(correlation_results: dict, results_file: str):
    """
    Save the results of the correlation to a file (old .txt format)
    """
    from tdct.correlation import write_results
    # write transformation params and correlation
    write_results(
        transf=correlation_results["output"]["transform"], 
        res_file_name=results_file,
        spots_3d=correlation_results["input"]["poi_3d"], 
        spots_2d=correlation_results["output"]["reprojected_2d_poi"],
        markers_3d=correlation_results["input"]["markers_3d"], 
        transformed_3d=correlation_results["output"]["reprojected_3d_coordinates"], 
        markers_2d=correlation_results["input"]["markers_2d"],
        rotation_center=correlation_results["input"]["rotation_center"], 
        modified_translation=correlation_results["output"]["modified_translation"],
        imageProps=correlation_results["input"]["imageProps"]
        )

def run_correlation(
    fib_coords: np.ndarray,
    fm_coords: np.ndarray,
    poi_coords: np.ndarray,
    image_props: tuple,
    rotation_center: tuple,
    path: Optional[str] = None,
    fib_image_filename: str = "",
    fm_image_filename: str = "",
) -> dict:
    """Run the correlation between the FIB and FM images"""
    # run the correlation
    correlation_results = correlate(
        markers_3d=fm_coords,
        markers_2d=fib_coords,
        poi_3d=poi_coords,
        rotation_center=rotation_center,
        imageProps=image_props,
    )

    # input data
    input_data = {
        "fib_coordinates": fib_coords.tolist(),
        "fm_coordinates": fm_coords.tolist(),
        "poi_coordinates": poi_coords.tolist(),
        "image_properties": {
            "fib_image_filename": fib_image_filename,
            "fib_image_shape": list(image_props[0]),
            "fib_pixel_size_um": float(image_props[1]),
            "fm_image_filename": fm_image_filename,
            "fm_image_shape": list(image_props[2]),
        },
        "rotation_center": list(rotation_center),
        "rotation_center_custom": list(rotation_center),
        "method": "mulit-point",
    }

    # output data
    correlation_data = parse_correlation_result_v2(
        cor_ret=correlation_results, 
        input_data=input_data
    )

    # full correlation data
    full_correlation_data = {
        "metadata": {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "data_path": path,
            "csv_path": os.path.join(path, "data.csv"),
            "project_path": path, # TODO: add project path
        },
        "correlation": correlation_data,
    }
    if path is not None:
        save_correlation_data(full_correlation_data, path)

    return correlation_data

##### CORRELATION RESULTS #####

# convert 2D image coordinates to microscope image coordinates
def convert_poi_to_microscope_coordinates(
    poi_coordinates: np.ndarray, fib_image_shape: tuple, pixel_size_um: float
) -> list:
    # image centre
    cx = float(fib_image_shape[1] * 0.5)
    cy = float(fib_image_shape[0] * 0.5)

    poi_image_coordinates: list = []

    for i in range(poi_coordinates.shape[1]):
        px = poi_coordinates[:, i]  # (x, y, z) in pixel coordinates
        px = [float(px[0]), float(px[1])]
        px_x, px_y = (
            px[0] - cx,
            cy - px[1],
        )  # point in microscope image coordinates (px)
        pt_um = (
            px_x * pixel_size_um,
            px_y * pixel_size_um,
        )  # point in microscope image coordinates (um)
        poi_image_coordinates.append(
            {"image_px": px, 
             "px": [px_x, px_y], 
             "px_um": [pt_um[0], pt_um[1]],                 # micrometers
             "px_m": [pt_um[0] * 1e-6, pt_um[1] * 1e-6]}    # meters
        )

    return poi_image_coordinates

def _convert_poi_to_microscope_image_coordinate(
    poi_coordinates: Tuple[float, float], 
    fib_image_shape: Tuple[int, int], 
    pixel_size_um: float
) -> Dict[str, List[float]]:
    """Convert a single point of interest to microscope image coordinates. 
    Point of Interest (POI) is in 2D image coordinates (x, y) in image pixels.
    Microscope Image Coordinates are centred at the image centre (0, 0) and in micrometers.
    Args:
        poi_coordinates (Tuple[float, float]): Point of Interest (POI) in 2D image coordinates (x, y) in image pixels.
        fib_image_shape (Tuple[int, int]): Shape of the FIBSEM image (height, width).
        pixel_size_um (float): Pixel size in micrometers.
    Returns:
        Dict[str, List[float]]: Dictionary containing the POI in different coordinate systems."""
    
    # image centre
    cx = float(fib_image_shape[1] * 0.5)
    cy = float(fib_image_shape[0] * 0.5)

    px = [float(poi_coordinates[0]), float(poi_coordinates[1])] # (x, y) in pixel coordinates
    px_x, px_y = (px[0] - cx, cy - px[1])  # point in microscope image coordinates (px)
    pt_um = (px_x * pixel_size_um,px_y * pixel_size_um)  # point in microscope image coordinates (um)
    poi_image_coordinates = {"image_px": px, "px": [px_x, px_y], "px_um": [pt_um[0], pt_um[1]]}

    return poi_image_coordinates

def extract_transformation_data(transf, mod_translation, reproj_3d, delta_2d) -> dict:
    # extract eulers in degrees
    eulers = transf.extract_euler(r=transf.q, mode="x", ret="one")
    eulers = eulers * 180 / np.pi

    # RMS error
    rms_error = transf.rmsError

    # difference between points after transforming 3D points to 2D
    delta_2d_mean_abs_err = np.absolute(delta_2d).mean(axis=1)

    transformation_data = {
        "transformation": {
            "scale": float(transf.s_scalar),
            "rotation_eulers": eulers.tolist(),
            "rotation_quaternion": transf.q.tolist(),
            "translation_around_rotation_center_custom": mod_translation.tolist(),
            "translation_around_rotation_center_zero": transf.d.tolist(),
        },
        "error": {
            "reprojected_3d": reproj_3d.tolist(),
            "delta_2d": delta_2d.tolist(),
            "mean_absolute_error": delta_2d_mean_abs_err.tolist(),
            "rms_error": float(rms_error),
        },
    }

    return transformation_data

def parse_correlation_result(cor_ret: list, input_data: dict) -> dict:
    # point of interest data
    spots_2d = cor_ret[2]  # (points of interest in 2D image)
    fib_image_shape = input_data["image_properties"]["fib_image_shape"]
    pixel_size_um = input_data["image_properties"]["fib_pixel_size_um"]

    poi_image_coordinates = convert_poi_to_microscope_coordinates(
        spots_2d, fib_image_shape, pixel_size_um
    )

    # transformation data
    transf = cor_ret[0]     # transformation matrix
    reproj_3d = cor_ret[1]  # reprojected 3D points to 2D points
    delta_2d = cor_ret[3]   # difference between reprojected 3D points and 2D points (in pixels)
    mod_translation = cor_ret[5]  # translation around rotation center
    transformation_data = extract_transformation_data(transf=transf, 
                                                      mod_translation=mod_translation, 
                                                      reproj_3d=reproj_3d, 
                                                      delta_2d=delta_2d)

    correlation_data = {"input": input_data, "output": {}}
    correlation_data["output"].update(transformation_data)
    correlation_data["output"].update({"poi": poi_image_coordinates})

    return correlation_data

def parse_correlation_result_v2(cor_ret: dict, input_data: dict) -> dict:
    # point of interest data
    spots_2d = cor_ret["output"]["reprojected_2d_poi"]  # (points of interest in 2D image)
    fib_image_shape = input_data["image_properties"]["fib_image_shape"]
    pixel_size_um = input_data["image_properties"]["fib_pixel_size_um"]

    poi_image_coordinates = convert_poi_to_microscope_coordinates(
        spots_2d, fib_image_shape, pixel_size_um
    )

    # transformation data
    transf = cor_ret["output"]["transform"]     # transformation matrix
    reproj_3d = cor_ret["output"]["reprojected_3d_coordinates"]  # reprojected 3D points to 2D points
    delta_2d = cor_ret["output"]["reprojection_error"]   # difference between reprojected 3D points and 2D points (in pixels)
    mod_translation = cor_ret["output"]["modified_translation"]  # translation around rotation center
    transformation_data = extract_transformation_data(transf=transf, 
                                                      mod_translation=mod_translation, 
                                                      reproj_3d=reproj_3d, 
                                                      delta_2d=delta_2d)

    correlation_data = {"input": input_data, "output": {}}
    correlation_data["output"].update(transformation_data)
    correlation_data["output"].update({"poi": poi_image_coordinates})

    return correlation_data

def save_correlation_data(data: dict, path: str) -> None:
    correlation_data_filename = os.path.join(path, "correlation_data.yaml")
    with open(correlation_data_filename, "w") as file:
        yaml.dump(data, file)

    logging.info(f"Correlation data saved to: {correlation_data_filename}")

    return correlation_data_filename