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
        viewer.layers["img"].save(os.path.join(path, "image"))
        label = viewer.layers["Labels"].data
        # label = np.uint8(label)
        print(np.unique(label))
        im = Image.fromarray(label) 
        im.save(os.path.join(path, "label.tif"))  # or 'test.tif'


if __name__ == "__main__":
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="specify which user config file to use",
        dest="config",
        action="store",
        default=os.path.join("fibsem", "segmentation", "lachie_config.yml")
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

