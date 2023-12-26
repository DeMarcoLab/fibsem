import glob
import json
import os
import torch

import argparse


def export_model_checkpoint(
    path: str,
    checkpoint_path: str = None,
    checkpoint_name: str = "model_checkpoint.pth",
) -> None:
    """Save NNUNet model checkpoint as a single .pth file
    args:
        path: path to the nnunet model directory

    """
    # nnunet model directory structure for ensemble:
    # model
    #  dataset.json
    #  plans.json
    #  fold_n:
    #   checkpoint_best.pth
    #   checkpoint_final.pth

    # we want to convert it to a single .pth file with the following structure:
    # model_checkpoint.pth
    # dataset: dataset.json
    # plans: plans.json
    # fold_n:
    #  best:  checkpoint_best.pth
    #  final: checkpoint_final.pth

    # this makes it more portable and easier to load

    def load_json(path: str):
        with open(path, "r") as f:
            return json.load(f)

    # confirm that the path is a nnunet model directory
    if not os.path.isdir(path):
        raise ValueError(f"{path} is not a directory")
    if not os.path.exists(os.path.join(path, "dataset.json")):
        raise ValueError(f"{path} does not contain a dataset.json file")
    if not os.path.exists(os.path.join(path, "plans.json")):
        raise ValueError(f"{path} does not contain a plans.json file")

    print(f"Exporting model checkpoint from {path}...")

    MODEL_CHECKPOINT = {}

    # paths
    DATASET_JSON_PATH = os.path.join(path, "dataset.json")
    PLAN_JSON_PATH = os.path.join(path, "plans.json")

    # load the dataset and plans
    print("Loading dataset and plans configurations...")
    MODEL_CHECKPOINT["dataset"] = load_json(DATASET_JSON_PATH)
    MODEL_CHECKPOINT["plans"] = load_json(PLAN_JSON_PATH)

    # load the folds
    MODEL_CHECKPOINT["folds"] = {}

    # get all the fold directories,
    FOLD_DIRS = sorted(glob.glob(os.path.join(path, "fold_*")))
    print(f"Found {len(FOLD_DIRS)} folds...")
    for fold_dir in FOLD_DIRS:
        fold_name = os.path.basename(fold_dir)
        print(f"Processing fold {fold_name}...")

        # load the best/ final checkpoint
        BEST_CHECKPOINT_PATH = os.path.join(fold_dir, "checkpoint_best.pth")
        FINAL_CHECKPOINT_PATH = os.path.join(fold_dir, "checkpoint_final.pth")

        MODEL_CHECKPOINT["folds"][fold_name] = {
            "best": torch.load(BEST_CHECKPOINT_PATH, map_location=torch.device("cpu")),
            "final": torch.load(
                FINAL_CHECKPOINT_PATH, map_location=torch.device("cpu")
            ),
        }

    # save as single torch checkpoint
    if checkpoint_path is None:
        checkpoint_path = os.path.join(path, checkpoint_name)
    torch.save(MODEL_CHECKPOINT, checkpoint_path)
    print(f"Saved model checkpoint to {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Export nnunet model checkpoint as a single .pth file"
    )
    parser.add_argument("--path", type=str, help="path to nnunet model directory")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="path to save the checkpoint",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        help="name of the checkpoint",
        required=False,
        default="model_checkpoint.pth",
    )

    args = parser.parse_args()

    export_model_checkpoint(
        path=args.path,
        checkpoint_path=args.checkpoint_path,
        checkpoint_name=args.checkpoint_name,
    )


if __name__ == "__main__":
    main()
