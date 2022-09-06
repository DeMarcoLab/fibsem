import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_dir",
        help="the directory containing the raw images",
        dest="raw_dir",
        action="store",
    )
    parser.add_argument(
        "--data_dir",
        help="the directory to save the images and labels to",
        dest="data_dir",
        action="store",
    )
    parser.add_argument(
        "--zarr_dir",
        help="the directory to save the zarr dataset to",
        dest="zarr_dir",
        action="store",
    )

    args = parser.parse_args()
    raw_dir = args.raw_dir
    data_dir = args.data_dir
    zarr_dir = args.zarr_dir

    segmentation_config = {
        "raw_dir": raw_dir,
        "data_dir": data_dir,
        "zarr_dir": zarr_dir
    }

    # Saves a JSON file that git will ignore to allow each user to use local directories
    with open("segmentation_config.json", 'w') as f:
        json.dump(segmentation_config, f)

