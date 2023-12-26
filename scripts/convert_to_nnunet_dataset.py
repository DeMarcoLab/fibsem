import argparse
from fibsem.segmentation._nnunet import convert_to_nnunet_dataset

# Convert a fibsem dataset to nnunet format

def main():
    parser = argparse.ArgumentParser(description="Convert a dataset to nnunet format")
    parser.add_argument("--data_path", type=str, help="path to data")
    parser.add_argument("--label_path", type=str, help="path to labels")
    parser.add_argument("--nnunet_data_path", type=str, help="path to nnunet data")
    parser.add_argument(
        "--label_map", type=str, help="label map", required=False, default=None
    )
    parser.add_argument(
        "--filetype", type=str, help="filetype", required=False, default=".tif"
    )

    args = parser.parse_args()

    if args.label_map is not None:
        print(args.label_map)
        # open text file, read each line, and add to list of labels, remove newline character
        with open(args.label_map, "r") as f:
            labels = [line.rstrip("\n") for line in f]
        args.label_map = labels

    # print(args.data_path)
    # print(args.label_path)
    # print(args.nnunet_data_path)
    # print(args.label_map)
    # print(args.filetype)

    # return
    convert_to_nnunet_dataset(
        data_path=args.data_path,
        label_path=args.label_path,
        nnunet_data_path=args.nnunet_data_path,
        label_map=args.label_map,
        filetype=args.filetype,
    )


if __name__ == "__main__":
    main()
