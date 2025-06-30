import csv
import logging
from typing import Tuple

import numpy as np
import tifffile as tff
from ome_types import from_tiff
from ome_types.model import (
    OME,
    Pixels,
    Pixels_DimensionOrder,
    Plane,
    TiffData,
    UnitsLength,
)
from ome_types.model import (
    Image as OMEImage,
)
from PIL import Image

############# PARSER FUNCTIONS #############

def parse_coordinates(fib_coord_filename: str, fm_coord_filename: str) -> list:
    """Parse the coordinates from the old style coordinate files"""

    def parse_coordinate_file(filename: str, delimiter: str = "\t") -> list:
        coords: list = []
        with open(filename) as csv_file:
            for row in csv.reader(csv_file, delimiter=delimiter):
                coords.append([field for field in row])
        return coords

    fib_coordinates = parse_coordinate_file(fib_coord_filename)
    fm_coordinates = parse_coordinate_file(fm_coord_filename)

    fib_coordinates = np.array(fib_coordinates, dtype=np.float32)
    fm_coordinates = np.array(fm_coordinates, dtype=np.float32)

    return fib_coordinates, fm_coordinates

def parse_metadata(filename: str) -> np.ndarray:
    """parse metadata from a tfs tiff file"""
    # TODO: replace this with real parser versions eventually
    md = {}
    with tff.TiffFile(filename) as tif:
        for page in tif.pages:
            for tag in page.tags.values():
                if tag.name == "FEI_HELIOS" or tag.code == 34682: # TFS_MD tag
                    md = tag.value
    return md

def parse_ome_fibsem_image(filename: str) -> tuple[np.ndarray, float]:
    """Parse OME metadata from a FibsemImage"""

    image = tff.imread(filename)
    ome = from_tiff(filename)
    pixels_md = ome.images[0].pixels
    pixel_size = pixels_md.physical_size_x  # assume isotropic
    pixel_size_unit = pixels_md.physical_size_x_unit
    pixel_size *= _unit_map[pixel_size_unit]  # convert to SI
    return image, pixel_size

def load_openfibsem_image(filename: str) -> tuple[np.ndarray, float]:

    from fibsem.structures import FibsemImage
    image = FibsemImage.load(filename)
    image = image.data
    pixel_size = None
    pixel_size = image.metadata.pixel_size.x

    return image, pixel_size

def load_image_and_metadata(filename: str) -> tuple[np.ndarray, dict]:
    # TODO: convert to FIBSEMImage always, require the package...
    try:
        image, pixel_size = parse_ome_fibsem_image(filename)
    except Exception as e:
        logging.debug(f"Failed to load as OME Image: {e}")

        try:
            image, pixel_size = load_openfibsem_image(filename)
        except Exception as e:
            logging.debug(f"Failed to load as OpenFIBSEM Image: {e}")
            try:
                image, pixel_size = load_tfs_image(filename)
            except Exception as e:
                logging.error(f"Failed to load as TFS image: {e}")
                image = tff.imread(filename)
                pixel_size = None
                return None, None

    return image, pixel_size

def load_tfs_image(filename: str) -> Tuple[np.ndarray, float]:
    """Load a TFS image and extract the pixel size from the metadata"""
    image = tff.imread(filename)
    md = parse_metadata(filename)

    pixel_size = None
    try:
        pixel_size = md["Scan"]["PixelWidth"]
    except KeyError as e:
        logging.warning(f"Pixel size not found in metadata: {e}")
        pass

    # convert to grayscale
    if image.ndim == 3:
        image = np.asarray(Image.fromarray(image).convert("L"))

    trim_metadata: bool = False
    try:
        shape = md["Image"]["ResolutionY"], md["Image"]["ResolutionX"]
        if image.shape != shape:
            logging.info(
                f"Image shape {image.shape} does not match metadata shape {shape}, likely a metadata bar present"
            )
            trim_metadata = True
    except KeyError as e:
        logging.warning(f"Image shape not found in metadata: {e}")
        pass

    # trim the image to before the first row with all zeros
    if trim_metadata:
        try:
            # crop the image to the metadata bar
            cropped_img = image[: shape[0], : shape[1]]
            # remove the metadata bar with image processing
            trimmed_img = remove_metadata_bar(image)

            logging.info(
                f"Cropped Shape: {cropped_img.shape}, Trimmed Shape: {trimmed_img.shape}"
            )
            if cropped_img.shape != trimmed_img.shape:
                raise ValueError(
                    "Cropped image shape does not match trimmed image shape"
                )

            if image.shape != trimmed_img.shape:
                logging.info(f"Image trimmed from {image.shape} to {trimmed_img.shape}")
                image = trimmed_img
        except Exception as e:
            logging.error(f"Error trimming image: {e}")
            pass

    return image, pixel_size

def remove_metadata_bar(img: np.ndarray) -> np.ndarray:
    """Loop through the image, and check if the row is all zeros indicating the start of the metadata bar"""

    for i, row in enumerate(img):
        if not np.any(row):
            # trim the image when the first row with all zeros is found
            break
    return img[:i]


def load_and_parse_fib_image(filename: str) -> tuple[np.ndarray, float]:
    image, pixel_size = load_image_and_metadata(filename)
    
    # from pprint import pprint
    # pprint(md)

    return image, pixel_size


RGB_TO_COLOUR = {
        (255, 0, 0): "red",
        (0, 255, 0): "green",
        (0, 0, 255): "blue",
        (255, 255, 0): "yellow",
        (255, 0, 255): "magenta",
        (0, 255, 255): "cyan",
        (255, 255, 255): "gray",
        (0, 0, 0): "black"
    }
COLOUR_TO_RGB = {v: k for k, v in RGB_TO_COLOUR.items()}


def rgb_to_color_name(rgb):
    colors=  RGB_TO_COLOUR

    # Find the color with the minimum Euclidean distance
    closest_color = min(colors.keys(), key=lambda color: sum((a-b)**2 for a, b in zip(rgb, color)))

    return colors[closest_color]

_unit_map = {
    UnitsLength.NANOMETER: 1e-9,
    UnitsLength.MICROMETER: 1e-6,
    UnitsLength.MILLIMETER: 1e-3,
    UnitsLength.METER: 1,
}

def load_and_parse_fm_image(path: str) -> Tuple[np.ndarray, dict]:
    image = tff.imread(path)

    zstep, pixel_size, colours, ome = None, None, None, None
    x, y, z = None, None, None
    nc, nz, ny, nx = None, None, None, None
    exposure_times = {}
    try:
        ome = from_tiff(path)
        pixels_md = ome.images[0].pixels
        pixel_size = pixels_md.physical_size_x # assume isotropic
        zstep = pixels_md.physical_size_z

        # convert to SI (if required)
        pixel_size_unit = pixels_md.physical_size_x_unit
        zstep_unit = pixels_md.physical_size_z_unit

        pixel_size *= _unit_map[pixel_size_unit]
        zstep *= _unit_map[zstep_unit]

        colours = [channel.color.as_rgb_tuple() for channel in pixels_md.channels]

        # image dimensions
        nc = pixels_md.size_c
        nz = pixels_md.size_z
        ny = pixels_md.size_y
        nx = pixels_md.size_x

        xs, ys, zs = [], [], []
        for plane in pixels_md.planes:
            xs.append(plane.position_x) # query unit?
            ys.append(plane.position_y)
            zs.append(plane.position_z)

            exposure_times[plane.the_c] = plane.exposure_time # assume constant for all planes

        x = np.mean(xs, dtype=np.float32)
        y = np.mean(ys, dtype=np.float32)
        z = np.mean(zs, dtype=np.float32)
    except Exception as e:
        logging.debug(f"Failed to extract metadata: {e}")

    # check if shape is CZYX, matches nc, nz, ny, nx
    # this is required because tifffile does not always return the correct shape, when there is z=1
    if nc is not None:
        if image.shape != (nc, nz, ny, nx):
            logging.warning(f"Image shape {image.shape} does not match metadata shape {(nc, nz, ny, nx)}")
            # reshape to match metadata shape
            image = image.reshape((nc, nz, ny, nx))

    # convert to 4D if necessary
    if image.ndim == 3:
        image = np.expand_dims(image, axis=0) # TODO: make sure we reshape colour too

    if colours is None:
        colours = [(255, 255, 255) for _ in range(image.shape[0])]

    colours = [rgb_to_color_name(colour) for colour in colours]

    return image, {"pixel_size": pixel_size, 
                   "zstep": zstep, "colours": colours,
                   "x": x, "y": y, "z": z, "exposure_time": exposure_times,
                   "ome": ome,
                   }

def get_z_plane_positions(pos_z: float, nz: int, zstep: float) -> np.ndarray:
    """Calculate the position of the z-planes based on the central plane position, number of planes, and z-step size.
    Assumes that the central plane is at pos_z, with half planes above and half below.
    NOTE: if an even number of planes is given, the central plane is at pos_z - zstep/2.
    Args:
        pos_z (float): Central plane position.
        nz (int): Number of planes.
        zstep (float): Z-step size.
    Returns:
        np.ndarray: Array of z-plane positions.
    """
    z_positions = np.linspace(pos_z - (nz - 1) * zstep / 2,
                            pos_z + (nz - 1) * zstep / 2,
                            nz)
    assert len(z_positions) == nz
    return z_positions

def write_ome_tiff(image: np.ndarray, md: dict, filename: str) -> str:
    """Write OME-TIFF file with metadata.
    Args:
        image (np.ndarray): Image data in CZYX format.
        md (dict): Metadata dictionary containing OME and other information.
            Keys include:
                - "x": X position of the image
                - "y": Y position of the image
                - "z": Z position of the image
                - "zstep": Z step size
                - "pixel_size": Pixel size in meters (x, y)
                - "exposure_time": Exposure time for each channel
                - "ome": OME metadata object
                - "colours": List of colors for each channel
        filename (str): Output filename for the OME-TIFF file.
    Returns:
        str: Path to the saved OME-TIFF file.
    """
    # extract metadata
    nc, nz, ny, nx = image.shape  # CZYX
    pos_x, pos_y = md["x"], md["y"]
    ome: OME = md["ome"]
    zstep = md["zstep"]     # pixelsize_z
    pos_z = md["z"]         # pos_z

    # compute z-plane positions
    z_positions = get_z_plane_positions(pos_z=pos_z, nz=nz, zstep=zstep)

    channels = []
    tiff_data_blocks = []
    planes = []

    ifd = 0
    for c in range(nc):

        # Note: this might need to change
        ch = ome.images[0].pixels.channels[c]
        channels.append(ch)

        exposure_time = md["exposure_time"][c]
        for z in range(nz):

            plane = Plane(
                the_z=z,
                the_c=c,
                the_t=0,  # expand to use time dimension
                position_x=pos_x,
                position_y=pos_y,
                position_z=z_positions[z],
                position_x_unit=UnitsLength.METER,
                position_y_unit=UnitsLength.METER,
                position_z_unit=UnitsLength.METER,
                exposure_time=exposure_time,
            )
            planes.append(plane)

            tiff_data = TiffData(ifd=ifd, first_c=c, first_z=z, plane_count=1)
            tiff_data_blocks.append(tiff_data)

            ifd+= 1

    dtype = image.dtype.name
    if image.dtype.name in "float32":
        dtype = "float"
        # TODO: fix dtype for exported fib-view screenshot makes it float32? convert to uint16?

    ome_image = OMEImage(
        id=ome.images[0].id,
        name=ome.images[0].name,
        description=ome.images[0].description,
        acquisition_date=ome.images[0].acquisition_date,
        pixels=Pixels(
            id=ome.images[0].pixels.id,
            type=dtype,
            size_x=nx,
            size_y=ny,
            size_z=nz,
            size_c=nc,
            size_t=1,  # single timepoint
            dimension_order=Pixels_DimensionOrder.XYZCT,
            physical_size_x=md["pixel_size"],
            physical_size_y=md["pixel_size"],
            physical_size_z=zstep,
            physical_size_x_unit=UnitsLength.METER,
            physical_size_y_unit=UnitsLength.METER,
            physical_size_z_unit=UnitsLength.METER,
            channels=[ch for ch in ome.images[0].pixels.channels],
            planes=planes,
            tiff_data_blocks=tiff_data_blocks
        )
    )

    ome_md = OME(
        images=[ome_image],
        instruments=ome.instruments,
        structured_annotations=ome.structured_annotations,
    )

    # convert OME metadata to XML, and validate
    ome_xml = ome_md.to_xml()
    assert tff.OmeXml.validate(ome_xml), "OME XML is not valid"

    # reshape image to 5D for tifffile (CZYX -> TCZYX)
    tifffile_image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2], image.shape[3])

    with tff.TiffWriter(filename) as tif:
        tif.write(data=tifffile_image, contiguous=True)
        tif.overwrite_description(ome_xml)

    return filename