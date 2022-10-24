import glob
import napari
import zarr
import tifffile as tff
import os


def import_images(path: str) -> zarr.Array:
    vol = tff.imread(os.path.join(path, "*.tif*"), aszarr=True) # loading folder of .tif into zarr array)
    imgs = zarr.open(vol)
    return imgs

def label_images(zarr_set: zarr.Array) -> None:
    viewer = napari.view_image(zarr_set)
    # viewer.add_labels(name="segmentation")
    napari.run()

imgs = import_images(r"C:\Users\lachl\OneDrive\Desktop\DeMarco\raw_img")
label_images(imgs)