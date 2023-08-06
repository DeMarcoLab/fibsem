import glob
import logging
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from skimage import transform

from fibsem import acquire, conversions, utils
from fibsem.structures import (BeamType, FibsemImage, FibsemImageMetadata, MicroscopeSettings,
                               FibsemStagePosition, Point)
from fibsem.microscope import FibsemMicroscope

def _tile_image_collection(microscope: FibsemMicroscope, settings: MicroscopeSettings, grid_size:float, tile_size:float, overlap: float = 0.0, cryo: bool = True) -> dict: 

    # TODO: OVERLAP + STITCH
    n_rows, n_cols = int(grid_size // tile_size), int(grid_size // tile_size)
    dx, dy = settings.image.hfw, settings.image.hfw 

    dy *= -1 # need to invert y-axis

    # fixed image settings
    settings.image.resolution = [1024, 1024]
    settings.image.dwell_time = 1e-6
    settings.image.autocontrast = cryo # required for cryo
    settings.image.gamma_enabled = False

    logging.info(f"TILE COLLECTION: {settings.image.label}")
    logging.info(f"Taking n_rows={n_rows}, n_cols={n_cols} ({n_rows*n_cols}) images. Grid Size = {grid_size*1e6} um, Tile Size = {tile_size*1e6} um")
    logging.info(f"dx: {dx*1e6} um, dy: {dy*1e6} um")

    # start in the middle of the grid

    start_state = microscope.get_current_microscope_state()
    
    # we save all intermediates into a folder with the same name as the label, then save the stitched image into the parent folder
    prev_path = settings.image.save_path
    settings.image.save_path = os.path.join(settings.image.save_path, settings.image.label)
    os.makedirs(settings.image.save_path, exist_ok=True)
    prev_label = settings.image.label
    
    # BIG_IMAGE FOR DEBUGGING ONLY
    settings.image.hfw = grid_size
    settings.image.label = "big_image"
    big_image = acquire.new_image(microscope, settings.image)
    
    # TOP LEFT CORNER START
    settings.image.hfw = tile_size
    settings.image.label = prev_label
    settings.image.autocontrast = False # required for cryo
    start_move = grid_size / 2 - tile_size / 2
    dxg, dyg = start_move, start_move
    dyg *= -1

    microscope.stable_move(settings=settings, dx=-dxg, dy=-dyg, beam_type=settings.image.beam_type, _fixed=True)
    state = microscope.get_current_microscope_state()
    images = []

    for i in range(n_rows):

        microscope._safe_absolute_stage_movement(state.absolute_position)
        
        img_row = []
        microscope.stable_move(
            settings=settings,
            dx=0,
            dy=i*dy, 
            beam_type=settings.image.beam_type, 
            _fixed=True)


        for j in range(n_cols):
            settings.image.label = f"tile_{i}_{j}"
            microscope.stable_move(settings=settings, dx=dx*(j!=0),  dy=0, beam_type=settings.image.beam_type) # dont move on the first tile?

            logging.info(f"ACQUIRING IMAGE {i}, {j}")
            image = acquire.new_image(microscope, settings.image)
            img_row.append(image)
        images.append(img_row)

    # restore initial state
    microscope.set_microscope_state(start_state)
    settings.image.save_path = prev_path

    ddict = {"grid_size": grid_size, "tile_size": tile_size, "n_rows": n_rows, "n_cols": n_cols, 
            "image_settings": settings.image, 
            "dx": dx, "dy": dy, "cryo": cryo,
            "start_state": start_state, "prev-label": prev_label, "start_move": start_move, "dxg": dxg, "dyg": dyg,
            "images": images, "big_image": big_image }

    return ddict



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
    image.metadata.image_settings = ddict["image_settings"]
    image.metadata.image_settings.hfw = deepcopy(float(ddict["grid_size"]))
    image.metadata.image_settings.resolution = deepcopy([arr.shape[0], arr.shape[1]])

    filename = os.path.join(image.metadata.image_settings.save_path, f'{ddict["prev-label"]}')
    image.save(filename)

    # for cryo need to histogram equalise
    if ddict.get("cryo", False):
        image = acquire.auto_gamma(image, method="autogamma")

    filename = os.path.join(image.metadata.image_settings.save_path, f'{ddict["prev-label"]}-autogamma')
    image.save(filename)

    # save ddict as yaml
    del ddict["images"]
    del ddict["big_image"]

    ddict["image_settings"] = ddict["image_settings"].__to_dict__()
    ddict["start_state"] = ddict["start_state"].__to_dict__()
    filename = os.path.join(filename, f'{ddict["prev-label"]}') # subdir
    utils.save_yaml(filename, ddict) 

    return image


def _tile_image_collection_stitch(microscope, settings, grid_size, tile_size, overlap=0) -> FibsemImage:

    ddict = _tile_image_collection(microscope, settings, grid_size, tile_size)
    image = _stitch_images(ddict["images"], ddict, overlap=overlap)

    return image


def _stitch_arr(images, dtype=np.uint8):

    arr = np.array(images)
    n_rows, n_cols = arr.shape[0], arr.shape[1]
    shape = arr[0, 0].data.shape

    arr = np.zeros(shape=(n_rows*shape[0], n_cols*shape[1]), dtype=dtype)

    for i in range(n_rows):
        for j in range(n_cols):
            arr[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = images[i][j].data  

    return arr



def _create_tiles(image: np.ndarray, n_rows, n_cols, tile_size, overlap=0):
    # create tiles
    tiles = []
    for i in range(n_rows):

        for j in range(n_cols):

            # get tile
            tile = image[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size]

            # append to list
            tiles.append(tile)

    tiles = np.array(tiles)

    return tiles



def _calculate_repojection(image: FibsemImage, pos: FibsemStagePosition):

    # difference between current position and image position
    delta = pos - image.metadata.microscope_state.absolute_position

    # projection of the positions onto the image
    dx = delta.x
    dy = np.sqrt(delta.y**2 + delta.z**2) # TODO: correct for perspective here
    dy = dy if (delta.y<0) else -dy

    pt_delta = Point(dx, dy)
    px_delta = pt_delta._to_pixels(image.metadata.pixel_size.x)

    image_centre = Point(x=image.data.shape[1]/2, y=image.data.shape[0]/2)
    point = image_centre + px_delta

    # NB: there is a small reprojection error that grows with distance from centre
    # print(f"ERROR: dy: {dy}, delta_y: {delta.y}, delta_z: {delta.z}")

    return point


def _reproject_positions(image:FibsemImage, positions: list[FibsemStagePosition]):
    # reprojection of positions onto image coordinates
    points = []
    for pos in positions:
        points.append(_calculate_repojection(image, pos))
    
    return points




def _plot_positions(image: FibsemImage, positions: list[FibsemStagePosition], show:bool=False) -> plt.Figure:

    points = _reproject_positions(image, positions)

    # plot on matplotlib
    fig = plt.figure(figsize=(15, 15))
    plt.imshow(image.data, cmap="gray")

    COLOURS = ["lime", "blue", "cyan", "magenta", 
        "hotpink", "yellow", "orange", "red"]
    for i, (pos, pt) in enumerate(zip(positions, points)):
        c =COLOURS[i%len(COLOURS)]
        plt.plot(pt.x, pt.y, ms=20, c=c, marker="+", markeredgewidth=2, label=f"{pos.name}")

        # draw label next to point
        plt.text(pt.x-225, pt.y-50, pos.name, fontsize=14, color=c, alpha=0.75)

    plt.axis("off")
    if show:
        plt.show()

    return fig


# TODO: these probs should be somewhere else
def _convert_image_coord_to_position(microscope, settings, image, coords) -> FibsemStagePosition:



    # microscope coordinates
    point = conversions.image_to_microscope_image_coordinates(
        Point(x=coords[1], y=coords[0]), image.data, image.metadata.pixel_size.x,
    )

    _new_position = microscope._calculate_new_position( 
            settings=settings, 
            dx=point.x, dy=point.y, 
            beam_type=image.metadata.image_settings.beam_type, 
            base_position=image.metadata.microscope_state.absolute_position)

    return _new_position


def _convert_image_coords_to_positions(microscope, settings, image, coords) -> list[FibsemStagePosition]:

    positions = []
    for i, coord in enumerate(coords):
        positions.append(_convert_image_coord_to_position(microscope, settings, image, coord))
        positions[i].name = f"Position {i:02d}"
    return positions