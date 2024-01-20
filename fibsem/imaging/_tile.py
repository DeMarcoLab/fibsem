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
import time

def _tile_image_collection(microscope: FibsemMicroscope, settings: MicroscopeSettings, grid_size:float, 
    tile_size:float, overlap: float = 0.0, cryo: bool = True, parent_ui = None) -> dict: 

    # TODO: OVERLAP + STITCH
    n_rows, n_cols = int(grid_size // tile_size), int(grid_size // tile_size)
    dx, dy = settings.image.hfw, settings.image.hfw 

    dy *= -1 # need to invert y-axis

    # fixed image settings
    settings.image.autogamma = False

    logging.info(f"TILE COLLECTION: {settings.image.filename}")
    logging.info(f"Taking n_rows={n_rows}, n_cols={n_cols} ({n_rows*n_cols}) images. Grid Size = {grid_size*1e6} um, Tile Size = {tile_size*1e6} um")
    logging.info(f"dx: {dx*1e6} um, dy: {dy*1e6} um")

    # start in the middle of the grid

    start_state = microscope.get_microscope_state()
    
    # we save all intermediates into a folder with the same name as the filename, then save the stitched image into the parent folder
    prev_path = settings.image.path
    settings.image.path = os.path.join(settings.image.path, settings.image.filename)
    os.makedirs(settings.image.path, exist_ok=True)
    prev_label = settings.image.filename
    
    # BIG_IMAGE FOR DEBUGGING ONLY
    settings.image.hfw = grid_size
    settings.image.filename = "big_image"
    big_image = acquire.new_image(microscope, settings.image)
    
    # TOP LEFT CORNER START
    settings.image.hfw = tile_size
    settings.image.filename = prev_label
    settings.image.autocontrast = False # required for cryo
    start_move = grid_size / 2 - tile_size / 2
    dxg, dyg = start_move, start_move
    dyg *= -1

    microscope.stable_move(dx=-dxg, dy=-dyg, beam_type=settings.image.beam_type, static_wd=True)
    state = microscope.get_microscope_state()
    images = []

    # stitched image
    shape = settings.image.resolution
    arr = np.zeros(shape=(n_rows*shape[0], n_cols*shape[1]), dtype=np.uint8)
    _counter = 0
    _total = n_rows*n_cols
    for i in range(n_rows):

        microscope.safe_absolute_stage_movement(state.stage_position)
        
        img_row = []
        microscope.stable_move(
            dx=0,
            dy=i*dy, 
            beam_type=settings.image.beam_type, 
            static_wd=True)


        for j in range(n_cols):
            settings.image.filename = f"tile_{i}_{j}"
            microscope.stable_move(dx=dx*(j!=0),  dy=0, beam_type=settings.image.beam_type) # dont move on the first tile?

            logging.info(f"ACQUIRING IMAGE {i}, {j}")
            image = acquire.new_image(microscope, settings.image)

            # stitch image
            arr[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = image.data
            
            if parent_ui:
                _counter+=1 
                parent_ui._update_tile_collection.emit({"msg": "Tile Collected", "i": i, "j": j, 
                    "n_rows": n_rows, "n_cols": n_cols, "image": arr, "counter": _counter, "total": _total })
                time.sleep(1)

            img_row.append(image)
        images.append(img_row)

    # restore initial state
    microscope.set_microscope_state(start_state)
    settings.image.path = prev_path

    ddict = {"grid_size": grid_size, "tile_size": tile_size, "n_rows": n_rows, "n_cols": n_cols, 
            "image_settings": settings.image, 
            "dx": dx, "dy": dy, "cryo": cryo,
            "start_state": start_state, "prev-filename": prev_label, "start_move": start_move, "dxg": dxg, "dyg": dyg,
            "images": images, "big_image": big_image, "stitched_image": arr}

    return ddict


# TODO: stitch while collecting
def _stitch_images(images, ddict: dict, overlap=0, parent_ui = None) -> FibsemImage:

    # arr = np.array(images)
    # n_rows, n_cols = arr.shape[0], arr.shape[1]
    # shape = arr[0, 0].data.shape

    # arr = np.zeros(shape=(n_rows*shape[0], n_cols*shape[1]), dtype=np.uint8)

    # for i in range(n_rows):
    #     for j in range(n_cols):
    #         arr[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = images[i][j].data
            
    #         if parent_ui:
    #             parent_ui._update_tile_collection.emit({"msg": "Tile Stitched", "i": i, "j": j, 
    #                 "n_rows": n_rows, "n_cols": n_cols, "image": arr })
    
    arr = ddict["stitched_image"]

    # convert to fibsem image
    image = FibsemImage(data=arr, metadata=images[0][0].metadata)
    image.metadata.microscope_state = deepcopy(ddict["start_state"])
    image.metadata.image_settings = ddict["image_settings"]
    image.metadata.image_settings.hfw = deepcopy(float(ddict["grid_size"]))
    image.metadata.image_settings.resolution = deepcopy([arr.shape[0], arr.shape[1]])

    filename = os.path.join(image.metadata.image_settings.path, f'{ddict["prev-filename"]}')
    image.save(filename)

    # for cryo need to histogram equalise
    if ddict.get("cryo", False):
        from fibsem.imaging.autogamma import auto_gamma
        image = auto_gamma(image, method="autogamma")

    filename = os.path.join(image.metadata.image_settings.path, f'{ddict["prev-filename"]}-autogamma')
    image.save(filename)

    # save ddict as yaml
    del ddict["images"]
    del ddict["big_image"]

    ddict["image_settings"] = ddict["image_settings"].to_dict()
    ddict["start_state"] = ddict["start_state"].to_dict()
    filename = os.path.join(filename, f'{ddict["prev-filename"]}') # subdir
    utils.save_yaml(filename, ddict) 

    return image


def _tile_image_collection_stitch(microscope, settings, grid_size, tile_size, overlap=0, cryo:bool=True, parent_ui = None) -> FibsemImage:

    ddict = _tile_image_collection(microscope, settings, grid_size, tile_size, cryo=cryo, parent_ui=parent_ui)
    image = _stitch_images(ddict["images"], ddict, overlap=overlap, parent_ui=parent_ui)

    return image


def _stitch_arr(images:list[FibsemImage], dtype=np.uint8):

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
    delta = pos - image.metadata.microscope_state.stage_position

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


def _reproject_positions(image:FibsemImage, positions: list[FibsemStagePosition], _bound: bool=False):
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
        # print("dr: ", dr)
        if np.isclose(dr, 180, atol=2):     
            # print("transforming position")
            pos = _transform_position(pos)

        pt = _calculate_repojection(image, pos)
        pt.name = pos.name
        
        if _bound:
            if pt.x<0 or pt.x>image.data.shape[1] or pt.y<0 or pt.y>image.data.shape[0]:
                continue 
        
        points.append(pt)
    
    return points




def _plot_positions(image: FibsemImage, positions: list[FibsemStagePosition], show:bool=False, minimap: bool=False, 
    _clip: bool=False, _bound: bool= True) -> plt.Figure:

    points = _reproject_positions(image, positions)

    # plot on matplotlib
    fig = plt.figure(figsize=(15, 15))
    plt.imshow(image.data, cmap="gray")

    COLOURS = ["lime", "blue", "cyan", "magenta", 
        "hotpink", "yellow", "orange", "red"]
    for i, (pos, pt) in enumerate(zip(positions, points)):

        # if points outside image, don't plot
        if _bound:
            if pt.x<0 or pt.x>image.data.shape[1] or pt.y<0 or pt.y>image.data.shape[0]:
                continue          
        
        # clip points to image 
        if _clip:
            pt.x = np.clip(pt.x, 0, image.data.shape[1])
            pt.y = np.clip(pt.y, 0, image.data.shape[0])


        c =COLOURS[i%len(COLOURS)]
        plt.plot(pt.x, pt.y, ms=20, c=c, marker="+", markeredgewidth=2, filename=f"{pos.name}")
        if minimap:
            fontsize = 30
        else:
            fontsize = 14
        # draw filename next to point
        plt.text(pt.x-225, pt.y-50, pos.name, fontsize=fontsize, color=c, alpha=0.75)

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

    _new_position = microscope.project_stable_move( 
            dx=point.x, dy=point.y, 
            beam_type=image.metadata.image_settings.beam_type, 
            base_position=image.metadata.microscope_state.stage_position)

    return _new_position


def _convert_image_coords_to_positions(microscope, settings, image, coords) -> list[FibsemStagePosition]:

    positions = []
    for i, coord in enumerate(coords):
        positions.append(_convert_image_coord_to_position(microscope, settings, image, coord))
        positions[i].name = f"Position {i:02d}"
    return positions


def _minimap(minimap_image: FibsemImage, positions: list[FibsemStagePosition]):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from PIL import Image
        from PyQt5.QtGui import QImage, QPixmap
        pil_image = None

        import matplotlib.pyplot as plt
        plt.close("all")
        fig = _plot_positions(image=minimap_image, positions = positions, minimap=True)
        plt.tight_layout(pad=0)
        canvas = FigureCanvasAgg(fig)

        # Render the figure onto the canvas
        canvas.draw()

        # Get the RGBA buffer from the canvas
        buf = canvas.buffer_rgba()
        # Convert the buffer to a PIL Image
        pil_image = Image.frombuffer('RGBA', canvas.get_width_height(), buf, 'raw', 'RGBA', 0, 1)
        pil_image = pil_image.resize((300, 300), Image.ANTIALIAS)
        # Convert the PIL image to a QImage
        image_qt = QImage(pil_image.tobytes(), pil_image.width, pil_image.height, QImage.Format_RGBA8888)
        # Convert the QImage to a QPixmap 
        qpixmap = QPixmap.fromImage(image_qt)
        
        return qpixmap



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


def _transform_position(pos: FibsemStagePosition):
    """This function takes in a position flat to a beam, and outputs the position if stage was rotated / tilted flat to the other beam)"""

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

    return transformed_position

from fibsem.structures import ImageSettings
def _update_image_region(microscope: FibsemMicroscope, image_settings: ImageSettings, image: FibsemImage, position: FibsemStagePosition) -> FibsemImage:
    
    region_image = acquire.new_image(microscope, image_settings)
    rows,cols = region_image.data.shape[0], region_image.data.shape[1]

    position_point = _reproject_positions(image, [position])[0]

    ymin = max(0, int(position_point.y)-rows//2)
    xmin = max(0, int(position_point.x)-cols//2)

    ymax = min(ymin+rows, image.data.shape[0])
    xmax = min(xmin+cols, image.data.shape[1])

    width = xmax - xmin
    height = ymax - ymin

    # overwrite the image with the region image, but only inside the image dimensions
    image.data[ymin:ymax, xmin:xmax] = region_image.data[:height, :width]

    return image