# 
import os
import glob
import tifffile as tff

import copy
from fibsem.detection.detection import get_objects, save_json 
from tqdm import tqdm

import argparse


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to data directory")
    parser.add_argument("--labels_path", type=str, help="Path to labels directory")
    parser.add_argument("--dataset_json_path", type=str, default=None, help="Path to save dataset json")
    parser.add_argument("--min_pixels", type=int, default=100, help="Minimum number of pixels for an object to be considered")

    args = parser.parse_args()

    if args.dataset_json_path is None:
        args.dataset_json_path = os.path.join(args.data_path, "data.json")
    

    image_filenames = sorted(glob.glob(os.path.join(args.data_path, "*.tif")))
    label_filenames = sorted(glob.glob(os.path.join(args.labels_path, "*.tif")))

    filenames = list(zip(image_filenames, label_filenames)) # TDOO: we dont actually need image files for this, just the labels?
    dat = []

    progress = tqdm(filenames)
    for img_fname, label_fname in progress:
        progress.set_description(f"Processing {os.path.basename(img_fname)}")
        
        image = tff.imread(img_fname)
        mask = tff.imread(label_fname)

        # get objects
        objects = get_objects(mask, min_pixels=args.min_pixels)

        # save 
        dat.append(copy.deepcopy({"filename": os.path.basename(img_fname), 
                                "path": os.path.dirname(img_fname), 
                                "mask_filename": os.path.basename(label_fname), 
                                "mask_path": os.path.dirname(label_fname),
                                "objects": objects}))

    print(f"Saving data.json to {args.dataset_json_path}")
    save_json(dat, args.dataset_json_path)


if __name__ == "__main__":
    main()