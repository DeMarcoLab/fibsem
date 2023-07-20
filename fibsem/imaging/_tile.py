

from fibsem import utils, acquire
from fibsem.structures import BeamType, FibsemStagePosition

import matplotlib.pyplot as plt
import os
import glob
import logging
from copy import deepcopy


def _tile_image_collection(microscope, settings, grid_size, tile_size) -> dict: 

    n_rows, n_cols = int(grid_size // tile_size), int(grid_size // tile_size)
    # TODO: OVERLAP + STITCH
    dx, dy = settings.image.hfw, settings.image.hfw 

    dy *= -1 # need to invert y-axis

    print(f"Taking n_rows={n_rows}, n_cols={n_cols} ({n_rows*n_cols}) images. Grid Size = {grid_size*1e6} um, Tile Size = {tile_size*1e6} um")
    print(f"dx: {dx*1e6} um, dy: {dy*1e6} um")

    # start in the middle of the grid
    start_state = microscope.get_current_microscope_state()
    settings.image.hfw = grid_size
    prev_label = settings.image.label
    settings.image.label = "big_image"
    big_image = acquire.new_image(microscope, settings.image)
    
    # TOP LEFT CORNER START
    settings.image.hfw = tile_size
    settings.image.label = prev_label
    start_move = grid_size / 2 - tile_size / 2
    dxg, dyg = start_move, start_move
    dyg *= -1

    microscope.stable_move(settings=settings, dx=-dxg, dy=-dyg, beam_type=settings.image.beam_type)
    state = microscope.get_current_microscope_state()
    images = []

    for i in range(n_rows):

        microscope._safe_absolute_stage_movement(state.absolute_position) # TODO: change this to absolute move, should be faster
        
        img_row = []
        microscope.stable_move(
            settings=settings,
            dx=0,
            dy=i*dy, 
            beam_type=settings.image.beam_type)


        for j in range(n_cols):
            settings.image.label = f"tile_{i}_{j}"
            microscope.stable_move(settings=settings, dx=dx*(j!=0),  dy=0, beam_type=settings.image.beam_type) # dont move on the first tile?

            logging.info(f"ACQUIRING IMAGE {i}, {j}")
            image = acquire.new_image(microscope, settings.image)
            logging.info(f"WORKING_DISTANCE: {image.metadata.microscope_state.eb_settings.working_distance}")
            img_row.append(image)
        images.append(img_row)

    # restore initial state
    microscope.set_microscope_state(start_state)

    ddict = {"grid_size": grid_size, "tile_size": tile_size, "n_rows": n_rows, "n_cols": n_cols, 
            "image_settings": settings.image, 
            "dx": dx, "dy": dy, 
            "start_state": start_state, "prev-label": prev_label, "start_move": start_move, "dxg": dxg, "dyg": dyg,
            "images": images, "big_image": big_image }
    # from pprint import pprint
    # pprint(ddict)

    return ddict


import numpy as np
from skimage import transform
from fibsem.structures import FibsemImage, FibsemImageMetadata


def _stitch_images(images, ddict: dict, overlap=0) -> FibsemImage:

    arr = np.array(images)
    n_rows, n_cols = arr.shape[0], arr.shape[1]
    shape = arr[0, 0].data.shape

    arr = np.zeros(shape=(n_rows*shape[0], n_cols*shape[1]), dtype=np.uint8)

    for i in range(n_rows):
        for j in range(n_cols):
            arr[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = images[i][j].data
    
    # convert to fibsem image
    image = FibsemImage(data=arr, metadata=images[0][0].metadata)
    image.metadata.microscope_state = deepcopy(ddict["start_state"])
    image.metadata.image_settings.hfw = ddict["grid_size"]
    image.metadata.image_settings = ddict["image_settings"]

    image.save(os.path.join(image.metadata.image_settings.save_path, 
        f"stitched_image-{settings.image.beam_type.name.lower()}.tif"))

    # save ddict as yaml
    # utils.save_yaml(os.path.join(image.metadata.image_settings.save_path, 
                                #  f"{image.metadata.image_settings.label.replace('.tif', '')}.yaml"), ddict) # TODO: remove images from dict

    return image


def _tile_image_collection_stitch(microscope, settings, grid_size, tile_size, overlap=0) -> FibsemImage:

    ddict = _tile_image_collection(microscope, settings, grid_size, tile_size)
    image = _stitch_images(ddict["images"], ddict, overlap=overlap)

    return image