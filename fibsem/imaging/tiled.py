
import logging
import os
import time
import datetime
from copy import deepcopy
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from fibsem import acquire, conversions
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemStagePosition,
    ImageSettings,
    Point,
)
from fibsem.ui.napari.utilities import is_inside_image_bounds

POSITION_COLOURS = ["lime", "blue", "cyan", "magenta", "hotpink", "yellow", "orange", "red"]

##### TILED ACQUISITION
def tiled_image_acquisition(
    microscope: FibsemMicroscope,
    image_settings: ImageSettings,
    grid_size: float,
    tile_size: float,
    overlap: float = 0.0,
    cryo: bool = True,
    parent_ui=None,
) -> dict: 
    """Tiled image acquisition. Currently only supports square grids with no overlap.
    Args:
        microscope: The microscope connection.
        image_settings: The image settings.
        grid_size: The size of the entire final image.
        tile_size: The size of the tiles.
        overlap: The overlap between tiles in pixels. Currently not supported.
        cryo: Whether to use cryo mode (histogram equalisation).
        parent_ui: The parent UI for progress updates.
    Returns:
        A dictionary containing the acquisition details for stitching."""

    # TODO: OVERLAP + STITCH
    n_rows, n_cols = int(grid_size // tile_size), int(grid_size // tile_size)
    dx, dy = image_settings.hfw, image_settings.hfw

    dy *= -1 # need to invert y-axis

    # fixed image settings
    image_settings.autogamma = False

    logging.info(f"TILE COLLECTION: {image_settings.filename}")
    logging.info(f"Taking nrows={n_rows}, ncols={n_cols} ({n_rows*n_cols}) images. TotalFoV={grid_size*1e6} um, TileFoV={tile_size*1e6} um")
    logging.info(f"dx: {dx*1e6} um, dy: {dy*1e6} um")

    # start in the middle of the grid

    start_state = microscope.get_microscope_state()
    
    # we save all intermediates into a folder with the same name as the filename, then save the stitched image into the parent folder
    prev_path = image_settings.path
    image_settings.path = os.path.join(image_settings.path, image_settings.filename)
    os.makedirs(image_settings.path, exist_ok=True)
    prev_label = image_settings.filename
    
    # BIG_IMAGE FOR DEBUGGING ONLY
    image_settings.hfw = grid_size
    image_settings.filename = "big_image"
    image_settings.save = False
    big_image = acquire.new_image(microscope, image_settings)

    # TOP LEFT CORNER START
    image_settings.hfw = tile_size
    image_settings.filename = prev_label
    image_settings.autocontrast = False # required for cryo
    image_settings.save = True
    start_move = grid_size / 2 - tile_size / 2
    dxg, dyg = start_move, start_move
    dyg *= -1

    microscope.stable_move(dx=-dxg, dy=-dyg, beam_type=image_settings.beam_type, static_wd=True)
    start_position = microscope.get_stage_position()
    images = []

    # stitched image
    shape = image_settings.resolution
    full_shape = (shape[0]*n_rows, shape[1]*n_cols)
    arr = np.zeros(shape=(full_shape), dtype=np.uint8)
    n_tiles_acquired = 0
    total_tiles = n_rows*n_cols
    for i in range(n_rows):

        microscope.safe_absolute_stage_movement(start_position)
        
        img_row = []
        microscope.stable_move(
            dx=0,
            dy=i*dy, 
            beam_type=image_settings.beam_type, 
            static_wd=True)

        for j in range(n_cols):
            image_settings.filename = f"tile_{i}_{j}"
            microscope.stable_move(dx=dx*(j!=0),  dy=0, beam_type=image_settings.beam_type) # dont move on the first tile?

            if parent_ui:
                if parent_ui.STOP_ACQUISITION:
                    raise Exception("User Stopped Acquisition")

            logging.info(f"Acquiring Tilet {i}, {j}")
            image = acquire.new_image(microscope, image_settings)

            # stitch image
            arr[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = image.data
            
            if parent_ui:
                n_tiles_acquired+=1
                parent_ui.tile_acquisition_progress_signal.emit(
                    {
                        "msg": "Tile Collected",
                        "i": i,
                        "j": j,
                        "n_rows": n_rows,
                        "n_cols": n_cols,
                        "image": arr,
                        "counter": n_tiles_acquired,
                        "total": total_tiles,
                    }
                )
                time.sleep(1)

            img_row.append(image)
        images.append(img_row)

    # restore initial state
    microscope.set_microscope_state(start_state)
    image_settings.path = prev_path

    ddict = {"grid_size": grid_size, "tile_size": tile_size, "n_rows": n_rows, "n_cols": n_cols, 
            "image_settings": image_settings, 
            "dx": dx, "dy": dy, "cryo": cryo,
            "start_state": start_state, "prev-filename": prev_label, "start_move": start_move, "dxg": dxg, "dyg": dyg,
            "images": images, "big_image": big_image, "stitched_image": arr}

    return ddict

# TODO: stitch while collecting
def stitch_images(images: List[List[FibsemImage]], ddict: dict, parent_ui = None) -> FibsemImage:
    """Stitch an array (2D) of images together. Assumes images are ordered in a grid with no overlap.
    Args:
        images: The images.
        parent_ui: The parent UI for progress updates.
    Returns:
        The stitched image."""    
    if parent_ui:
        total = ddict["n_rows"] * ddict["n_cols"]
        parent_ui.tile_acquisition_progress_signal.emit({"msg": "Stitching Tiles", "counter": total, "total": total})
    arr = ddict["stitched_image"]

    # convert to fibsem image
    image = FibsemImage(data=arr, metadata=images[0][0].metadata)
    image.metadata.microscope_state = deepcopy(ddict["start_state"])
    image.metadata.image_settings = ddict["image_settings"]
    image.metadata.image_settings.hfw = deepcopy(float(ddict["grid_size"]))
    image.metadata.image_settings.resolution = deepcopy([arr.shape[0], arr.shape[1]])

    # TODO: support overwrite protection here, very annoying to overwrite

    filename = os.path.join(image.metadata.image_settings.path, f'{ddict["prev-filename"]}')
    image.save(filename)
    # for cryo need to histogram equalise
    if ddict.get("cryo", False):
        from fibsem.imaging.autogamma import auto_gamma
        image = auto_gamma(image, method="autogamma")
    filename = os.path.join(image.metadata.image_settings.path, f'{ddict["prev-filename"]}-autogamma')
    image.save(filename)

    # for garbage collection
    del ddict["images"]
    del ddict["big_image"]

    return image

# TODO: add support for overlap, nrows, ncols
def tiled_image_acquisition_and_stitch(microscope: FibsemMicroscope, 
                                  image_settings: ImageSettings, 
                                  grid_size: float, 
                                  tile_size: float, 
                                  overlap: int = 0, cryo: bool = True, 
                                  parent_ui = None) -> FibsemImage:
    """Acquire a tiled image and stitch it together. Currently only supports square grids with no overlap.
    Args:
        microscope: The microscope connection.
        image_settings: The image settings.
        grid_size: The size of the entire final image.
        tile_size: The size of the tiles.
        overlap: The overlap between tiles in pixels. Currently not supported.
        cryo: Whether to use cryo mode (histogram equalisation).
        parent_ui: The parent UI for progress updates.
    Returns:
        The stitched image."""
    
    # add datetime to filename for uniqueness
    filename = image_settings.filename
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    image_settings.filename = f"{filename}-{timestamp}"

    ddict = tiled_image_acquisition(microscope=microscope, 
                                    image_settings=image_settings, 
                                    grid_size=grid_size, tile_size=tile_size, 
                                    cryo=cryo, parent_ui=parent_ui)
    image = stitch_images(images=ddict["images"], ddict=ddict, parent_ui=parent_ui)

    return image

##### REPROJECTION
# TODO: move these to fibsem.imaging.reprojection?
def calculate_reprojected_stage_position(image: FibsemImage, pos: FibsemStagePosition) -> Point:
    """Calculate the reprojected stage position on an image.
    Args:
        image: The image.
        pos: The stage position.
    Returns:
        The reprojected stage position on the image."""

    # difference between current position and image position
    delta = pos - image.metadata.microscope_state.stage_position

    # projection of the positions onto the image
    dx = delta.x
    dy = np.sqrt(delta.y**2 + delta.z**2) # TODO: correct for perspective here
    dy = dy if (delta.y<0) else -dy

    pt_delta = Point(dx, dy)
    px_delta = pt_delta._to_pixels(image.metadata.pixel_size.x)

    beam_type = image.metadata.image_settings.beam_type
    if beam_type is BeamType.ELECTRON:
        scan_rotation = image.metadata.microscope_state.electron_beam.scan_rotation
    if beam_type is BeamType.ION:
        scan_rotation = image.metadata.microscope_state.ion_beam.scan_rotation
    
    if np.isclose(scan_rotation, np.pi):
        px_delta.x *= -1.0
        px_delta.y *= -1.0

    image_centre = Point(x=image.data.shape[1]/2, y=image.data.shape[0]/2)
    point = image_centre + px_delta

    # NB: there is a small reprojection error that grows with distance from centre
    # print(f"ERROR: dy: {dy}, delta_y: {delta.y}, delta_z: {delta.z}")

    return point

def reproject_stage_positions_onto_image(
        image:FibsemImage, 
        positions: List[FibsemStagePosition], 
        bound: bool=False) -> List[Point]:
    """Reproject stage positions onto an image. Assumes image is flat to beam.
    Args:
        image: The image.
        positions: The positions.
        bound: Whether to only return points inside the image.
    Returns:
        The reprojected stage positions on the image plane."""

    # reprojection of positions onto image coordinates
    points = []
    for pos in positions:


        # hotfix (pat): demo returns None positions #240
        if image.metadata.microscope_state.stage_position.x is None:
            image.metadata.microscope_state.stage_position.x = 0
        if image.metadata.microscope_state.stage_position.y is None:
            image.metadata.microscope_state.stage_position.y = 0
        if image.metadata.microscope_state.stage_position.z is None:
            image.metadata.microscope_state.stage_position.z = 0
        if image.metadata.microscope_state.stage_position.r is None:
            image.metadata.microscope_state.stage_position.r = 0
        if image.metadata.microscope_state.stage_position.t is None:
            image.metadata.microscope_state.stage_position.t = 0      
                
        # automate logic for transforming positions
        # assume only two valid positions are when stage is flat to either beam...  
        # r needs to be 180 degrees different
        # currently only one way: Flat to Ion -> Flat to Electron
        dr = abs(np.rad2deg(image.metadata.microscope_state.stage_position.r - pos.r))
        if np.isclose(dr, 180, atol=2):     
            pos = _transform_position(pos)

        pt = calculate_reprojected_stage_position(image, pos)
        pt.name = pos.name
        
        if bound and not is_inside_image_bounds([pt.y, pt.x], image.data.shape):
            continue
        
        points.append(pt)
    
    return points

def plot_stage_positions_on_image(
        image: FibsemImage, 
        positions: List[FibsemStagePosition], 
        show: bool = False, 
        bound: bool= True) -> plt.Figure:
    """Plot stage positions reprojected on an image as matplotlib figure. Assumes image is flat to beam.
    Args:
        image: The image.
        positions: The positions.
        show: Whether to show the plot.
        bound: Whether to only plot points inside the image.
    Returns:
        The matplotlib figure."""

    # reproject stage positions onto image 
    points = reproject_stage_positions_onto_image(image=image, positions=positions)

    # construct matplotlib figure
    fig = plt.figure(figsize=(15, 15))
    plt.imshow(image.data, cmap="gray")

    for i, pt in enumerate(points):

        # if points outside image, don't plot
        if bound and not is_inside_image_bounds([pt.y, pt.x], image.data.shape):
            continue     

        c = POSITION_COLOURS[i%len(POSITION_COLOURS)]
        plt.plot(pt.x, pt.y, ms=20, c=c, marker="+", markeredgewidth=2, label=f"{pt.name}")
        # draw position name next to point
        plt.text(pt.x-225, pt.y-50, pt.name, fontsize=14, color=c, alpha=0.75)

    plt.axis("off")
    if show:
        plt.show()

    return fig

def convert_image_coord_to_stage_position(
    microscope: FibsemMicroscope, image: FibsemImage, coord: Tuple[float, float]
) -> FibsemStagePosition:
    """Convert a coordinate in the image to a stage position. Assume image is flat to beam.
    Args:
        microscope: The microscope connection.
        image: The image
        coord: The coordinate in the image (y,x).
    Returns:
        The stage position.
    """
    # convert image to microscope image coordinates
    point = conversions.image_to_microscope_image_coordinates(
        coord=Point(x=coord[1], y=coord[0]),
        image=image.data,
        pixelsize=image.metadata.pixel_size.x,
    )
    # project as stage position
    stage_position = microscope.project_stable_move(
        dx=point.x,
        dy=point.y,
        beam_type=image.metadata.image_settings.beam_type,
        base_position=image.metadata.microscope_state.stage_position,
    )

    return stage_position

def convert_image_coordinates_to_stage_positions(
    microscope: FibsemMicroscope, image: FibsemImage, coords: List[Tuple[float, float]]
) -> List[FibsemStagePosition]:
    """Convert a list of coordinates in the image to a list of stage positions. Assume image is flat to beam.
    Args:
        microscope: The microscope connection.
        image: The image
        coords: The coordinates in the image (y,x).
    Returns:
        The stage positions."""

    stage_positions = []
    for i, coord in enumerate(coords):
        stage_position = convert_image_coord_to_stage_position(
            microscope=microscope, image=image, coord=coord
        )
        stage_position.name = f"Position {i:02d}"
        stage_positions.append(stage_position)
    return stage_positions

##### THERMO ONLY

X_OFFSET = -0.0005127403888932854 
Y_OFFSET = 0.0007937916666666666

def _to_specimen_coordinate_system(pos: FibsemStagePosition):
    """Converts a position in the raw coordinate system to the specimen coordinate system"""

    specimen_offset = FibsemStagePosition(x=X_OFFSET, y=Y_OFFSET, z=0.0, r=0, t=0, coordinate_system="RAW")
    specimen_position = pos - specimen_offset

    return specimen_position

def _to_raw_coordinate_system(pos: FibsemStagePosition):
    """Converts a position in the raw coordinate system to the specimen coordinate system"""

    specimen_offset = FibsemStagePosition(x=X_OFFSET, y=Y_OFFSET, z=0.0, r=0, t=0, coordinate_system="RAW")
    raw_position = pos + specimen_offset

    return raw_position


def _transform_position(pos: FibsemStagePosition) -> FibsemStagePosition:
    """This function takes in a position flat to a beam, and outputs the position if stage was rotated / tilted flat to the other beam).
    Args:
        pos: The position flat to the beam.
    Returns:
        The position flat to the other beam."""

    specimen_position = _to_specimen_coordinate_system(pos)
    # print("raw      pos: ", pos)
    # print("specimen pos: ", specimen_position)

    # # inverse xy (rotate 180 degrees)
    specimen_position.x = -specimen_position.x
    specimen_position.y = -specimen_position.y

    # movement offset (calibration for compucentric rotation error)
    specimen_position.x += 50e-6
    specimen_position.y += 25e-6

    # print("rotated pos: ", specimen_position)

    # _to_raw_coordinates
    transformed_position = _to_raw_coordinate_system(specimen_position)
    transformed_position.name = pos.name

    # print("trans   pos: ", transformed_position)
    logging.info(f"Initial position {pos} was transformed to {transformed_position}")

    return transformed_position