import glob
import napari
import numpy as np
import zarr
import tifffile as tff
import os
from PIL import Image
import argparse
import yaml
from validate_config import validate_config

def label_images(raw_dir: str, data_dir: str) -> None:
    vol = tff.imread(os.path.join(raw_dir, "*.tif*"), aszarr=True) # loading folder of .tif into zarr array)
    zarr_set = zarr.open(vol)

    filenames = sorted(glob.glob(os.path.join(raw_dir, "*.tif*")))

    # if not os.path.exists(os.path.join(data_dir, "images")):
    #     os.mkdir(os.path.join(data_dir, "images"))

    # if not os.path.exists(os.path.join(data_dir, "labels")):
    #     os.mkdir(os.path.join(data_dir, "labels")) 

    # TODO: can just use os.makedirs(path, exist_ok=True)  
    os.makedirs(os.path.join(data_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "labels"), exist_ok=True) 

    for img, fname in zip(zarr_set, filenames):
        #Check to see if already labelled; if so, skip
        if os.path.basename(fname).split(".")[0] in os.listdir(os.path.join(data_dir, "images")): 
            continue

        print(fname)
        # viewer = napari.view_image(img)

        viewer = napari.Viewer()
        viewer.add_image(img, name="img")
        viewer.add_labels(np.zeros_like(img), name="Labels")
        # TODO: set active tool (paintbrush)

        # manually add label layer then use paint tool for segmentation
        # use different colour for different types of object. MAKE SURE TO BE CONSISTENT

        napari.run()

        # To stop labelling, exit napari without creating a Labels layer.
        # NOTE: Separate from an image with no class in it, in this case create an empty Labels layer

        
        bname = os.path.basename(fname).split(".")[0]
        
        viewer.layers["img"].save(os.path.join(data_dir, "images", f"{bname}.tif"))
        label = viewer.layers["Labels"].data.astype(np.uint8)

        im = Image.fromarray(label) 
        im.save(os.path.join(data_dir, "labels", f"{bname}.tif"))  # or 'test.tif'


if __name__ == "__main__":
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="specify which user config file to use",
        dest="config",
        action="store",
        default=os.path.join("fibsem", "segmentation", "config.yml")
    )

    args = parser.parse_args()
    config_dir = args.config

    # NOTE: Setup your config.yml file
    with open(config_dir, 'r') as f:
        config = yaml.safe_load(f)

    print("Validating config file.")
    validate_config(config, "labelling")

    raw_dir = config['labelling']['raw_dir']
    data_dir = config['labelling']['data_dir']
    
    label_images(raw_dir, data_dir)

