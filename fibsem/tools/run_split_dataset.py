import os
import glob
import pandas as pd
from pathlib import Path
import argparse

def _split_dataset(DATA_PATH: Path, OUTPUT_PATH: Path):

    TRAIN_PATH = os.path.join(OUTPUT_PATH, "train" )
    TEST_PATH = os.path.join(OUTPUT_PATH, "test")

    # make dirs
    os.makedirs(os.path.join(TRAIN_PATH, "labels"), exist_ok=True)
    os.makedirs(os.path.join(TEST_PATH, "labels"), exist_ok=True)

    df = pd.read_csv(os.path.join(DATA_PATH, "data.csv"))
    df["path"] = df["image"].apply(lambda x: os.path.join(DATA_PATH, f"{x}.tif"))
    df["mask_path"] = df["image"].apply(lambda x: os.path.join(DATA_PATH, "mask", f"{x}.tif"))

    print(f"total: {len(df)} images")

    # corrected: did the user overwrite the model prediction
    # False: the model prediction was correct -> test set
    # True: the model prediction was incorrect -> train set

    # split the data into train and test
    df_train = df[df["corrected"] == True]
    df_test = df[df["corrected"] == False]

    print(f"train: {len(df_train)} images")
    print(f"test: {len(df_test)} images")

    response = input(f"Move data to {OUTPUT_PATH}? [y/n]")

    if response == "y":
        # move the images and masks to the correct folder
        for path, mask_path in zip(df_train["path"].unique(), df_train["mask_path"].unique()):
            os.rename(path, os.path.join(TRAIN_PATH, os.path.basename(path)))
            os.rename(mask_path, os.path.join(TRAIN_PATH, "labels", os.path.basename(mask_path)))

        for path, mask_path in zip(df_test["path"].unique(), df_test["mask_path"].unique()):
            os.rename(path, os.path.join(TEST_PATH, os.path.basename(path)))
            os.rename(mask_path, os.path.join(TEST_PATH, "labels", os.path.basename(mask_path)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, dest="data_path", action="store", help="Path to the data folder")
    parser.add_argument("--output_path", type=str, dest="output_path", action="store", help="Path to the output folder")

    args = parser.parse_args()

    _split_dataset(args.data_path, args.output_path)


if __name__ == "__main__":
    main()