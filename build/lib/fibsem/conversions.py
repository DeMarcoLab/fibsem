# TODO
import numpy as np
from autoscript_sdb_microscope_client.structures import AdornedImage


### UTILS
def pixel_to_realspace_coordinate(coord: list, image: AdornedImage) -> list:
    """Convert pixel image coordinate to real space coordinate.

    This conversion deliberately ignores the nominal pixel size in y,
    as this can lead to inaccuracies if the sample is not flat in y.

    Parameters
    ----------
    coord : listlike, float
        In x, y format & pixel units. Origin is at the top left.

    image : AdornedImage
        Image the coordinate came from.

        # do we have a sample image somewhere?
    Returns
    -------
    realspace_coord
        xy coordinate in real space. Origin is at the image center.
        Output is in (x, y) format.
    """
    coord = np.array(coord).astype(np.float64)
    if len(image.data.shape) > 2:
        y_shape, x_shape = image.data.shape[0:2]
    else:
        y_shape, x_shape = image.data.shape

    pixelsize_x = image.metadata.binary_result.pixel_size.x
    # deliberately don't use the y pixel size, any tilt will throw this off
    coord[1] = y_shape - coord[1]  # flip y-axis for relative coordinate system
    # reset origin to center
    coord -= np.array([x_shape / 2, y_shape / 2]).astype(np.int32)
    realspace_coord = list(np.array(coord) * pixelsize_x)  # to real space
    return realspace_coord # TODO: convert to use Point struct



def get_lamella_size_in_pixels(img: AdornedImage, protocol: dict, use_trench_height: bool = False) -> tuple[int]:
    """Get the relative size of the lamella in pixels based on the hfw of the image.

    Args:
        img (AdornedImage): reference image
        settings (dict): protocol dictionary
        use_lamella_height (bool, optional): get the height of the lamella (True), or Trench. Defaults to False.

    Returns:
        tuple[int]: _description_
    """
    # get real size from protocol
    lamella_width = protocol["lamella_width"]
    lamella_height = protocol["lamella_height"]
        
    total_height = lamella_height
    if use_trench_height:
        trench_height = protocol["protocol_stages"][0]["trench_height"]
        total_height += 2 * trench_height

    # convert to px
    pixelsize = img.metadata.binary_result.pixel_size.x
    vfw = img.height * pixelsize
    hfw = img.width * pixelsize
    
    lamella_height_px = int((total_height / vfw) * img.height) 
    lamella_width_px = int((lamella_width / hfw) * img.width) 

    return (lamella_height_px, lamella_width_px)