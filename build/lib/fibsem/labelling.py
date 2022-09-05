import glob
import napari
import numpy as np
from skimage import data
import zarr
import tifffile as tff
import os
from PIL import Image, ImageGrab

def import_images(path: str) -> zarr.Array:
    vol = tff.imread(os.path.join(path, "*.tif*"), aszarr=True) # loading folder of .tif into zarr array)
    imgs = zarr.open(vol)
    return imgs

def label_images(save_dir, zarr_set: zarr.Array) -> None:

    
    # create the list of polygons
    triangle = np.array([[11, 13], [111, 113], [22, 246]])

    rectangle = np.array([[0, 2],  [0, 102], [100, 102], [100, 2]])

    i =0 

    path = r"C:\Users\lucil\OneDrive\Bureau\DeMarco_Lab\tif images training"
    filenames = sorted(glob.glob(os.path.join(path, "*.tif*")))
    # while i <= zarr_set.size:
    for img, fname in zip(zarr_set, filenames):
        print(fname)
        viewer = napari.view_image(img)
        #manually add label layer then use paint tool for segmentation
        # use different colour for different types of object. MAKE SURE TO BE CONSISTENT

        napari.run()

        # screenshot = ImageGrab.grabclipboard()
        # if screenshot is None:
        #     print("You forgot to copy image to clipboard.")

        # Saves an img with the keypoints superimposed.
        # viewer.layers.save(os.path.join(save_dir, os.path.basename(fname)))
        os.makedirs(os.path.join(save_dir, os.path.basename(fname)))
        viewer.layers["img"].save(os.path.join(save_dir, os.path.basename(fname), "image.tiff"))
        # i = i +1  b


imgs = import_images(r"C:\Users\lucil\OneDrive\Bureau\DeMarco_Lab\dm-embryo-3_20220719.104850\6a4e5fad-5484-4f20-8d9f-53d111394bb0")
label_images(r'C:\Users\lucil\OneDrive\Bureau\DeMarco_Lab\dm-embryo-3_20220719.104850\6a4e5fad-5484-4f20-8d9f-53d111394bb0\segmentation', imgs)