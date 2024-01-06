# 
import argparse
import os

from fibsem.detection.detection import generate_segmentation_objects

def main():

    parser = argparse.ArgumentParser("Generate segmentation objects from segmentation labels")
    parser.add_argument("--data_path", type=str, help="Path to data directory")
    parser.add_argument("--labels_path", type=str, help="Path to labels directory")
    parser.add_argument("--dataset_json_path", type=str, default=None, help="Path to save dataset json")
    parser.add_argument("--min_pixels", type=int, default=100, help="Minimum number of pixels for an object to be considered")

    args = parser.parse_args()

    if args.dataset_json_path is None:
        args.dataset_json_path = os.path.join(args.data_path, "data.json")
    
    generate_segmentation_objects(
        data_path=args.data_path, 
        labels_path=args.labels_path, 
        dataset_json_path=args.dataset_json_path, 
        min_pixels=args.min_pixels,
        save=True
    )


if __name__ == "__main__":
    main()