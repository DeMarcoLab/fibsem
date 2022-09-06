import glob
import napari
import numpy as np
import zarr
import tifffile as tff
import os
from PIL import Image
from tqdm import tqdm
import argparse
import json

def label_images(raw_dir: str, data_dir: str) -> None:
    vol = tff.imread(os.path.join(raw_dir, "*.tif*"), aszarr=True) # loading folder of .tif into zarr array)
    zarr_set = zarr.open(vol)

    filenames = sorted(glob.glob(os.path.join(raw_dir, "*.tif*")))

    for img, fname in zip(zarr_set, filenames):
        #Check to see if already labelled; if so, skip
        if os.path.basename(fname).split(".")[0] in os.listdir(data_dir):
            continue

        print(fname)
        viewer = napari.view_image(img)
        # manually add label layer then use paint tool for segmentation
        # use different colour for different types of object. MAKE SURE TO BE CONSISTENT

        napari.run()

        # To stop labelling, exit napari without creating a Labels layer.
        # NOTE: Separate from an image with no class in it, in this case create an empty Labels layer
        if len(viewer.layers) < 2:
            print("Finished labelling.")
            break

        # Saves an img with the keypoints superimposed.
        os.makedirs(os.path.join(data_dir, os.path.basename(fname).split(".")[0]))
        path = os.path.join(data_dir, os.path.basename(fname).split(".")[0])
        viewer.layers["img"].save(os.path.join(data_dir, os.path.basename(fname).split(".")[0], "image"))
        viewer.layers["Labels"].save(os.path.join(data_dir, os.path.basename(fname).split(".")[0], "label.png"))

        im = Image.open(os.path.join(path, "label.png")) 
        im.save(os.path.join(path, "label.tif"))  # or 'test.tif'
        os.remove(os.path.join(path, "label.png"))


if __name__ == "__main__":
    # NOTE: Running segmentation_config.py first allows labelling.py to remember your directories for future runs.
    if "segmentation_config.json" in os.listdir():
        with open("segmentation_config.json", 'r') as f:
            config = json.load(f)

        # command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--raw_dir",
            help="the directory containing the raw images",
            dest="raw_dir",
            action="store",
            default=config["raw_dir"],
        )
        parser.add_argument(
            "--data_dir",
            help="the directory to save the images and labels to",
            dest="data_dir",
            action="store",
            default=config["data_dir"],
        )
        parser.add_argument(
            "--zarr_dir",
            help="the directory to save the zarr dataset to",
            dest="zarr_dir",
            action="store",
            default=config["zarr_dir"],
        )
        parser.add_argument(
            "--img_size",
            help="the directory to save the images and labels to",
            dest="img_size",
            action="store",
            default=(1024,1536),
        )
        parser.add_argument(
            "--no_label",
            help="the directory to save the zarr dataset to",
            dest="no_label",
            action="store_true",
        )
        
    args = parser.parse_args()
    raw_dir = args.raw_dir
    data_dir = args.data_dir
    zarr_dir = args.zarr_dir
    img_size = args.img_size
    no_label = args.no_label
    
    if no_label:
        #save_zarr_dataset(data_dir, zarr_dir, img_size=img_size)
        pass
    else:
        label_images(raw_dir, data_dir)
        #save_zarr_dataset(data_dir, zarr_dir, img_size=img_size)

