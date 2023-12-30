import glob
import json
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

import glob
import PIL.Image
from tqdm import tqdm
import os
import shutil

import json


def load_model(path: str) -> nnUNetPredictor:
    # instantiate the nnUNetPredictor
    model = nnUNetPredictor(
        tile_step_size=0.5,
        perform_everything_on_gpu=True,
        device=torch.device("cuda", 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )

    model = load_from_checkpoint(model, path)

    return model


def load_image(fname) -> np.ndarray:
    return SimpleITKIO().read_images([fname])[0]


def inference(model: nnUNetPredictor, image: np.ndarray):
    # todo: dont' return scores by defaults
    IMAGE_PROPERTIES = {
        "sitk_stuff": {
            "spacing": (1.0, 1.0),
            "origin": (0.0, 0.0),
            "direction": (1.0, 0.0, 0.0, 1.0),
        },
        "spacing": [999.0, 1.0, 1.0],
    }

    mask, scores = model.predict_single_npy_array(
        image, IMAGE_PROPERTIES, None, None, True
    )

    # mask = class map
    # scores = probability map for each class

    return mask, scores


def load_from_checkpoint(
    model: nnUNetPredictor, checkpoint_path: str
) -> nnUNetPredictor:
    """Load model from single checkpoint"""
    # TODO: this is a copy of nnunetv2.inference.predict_from_raw_data.load_from_checkpoint
    # should be depreciated once the change is upstreamed...

    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.inference.predict_from_raw_data import determine_num_input_channels
    import nnunetv2
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
    from nnunetv2.utilities.label_handling.label_handling import (
        determine_num_input_channels,
    )
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from torch._dynamo import OptimizedModule

    # check if nnunetPredictor has method called load_from_checkpoint
    # TODO: remove once the change is upstreamed
    method_deprecated = hasattr(model, "load_from_checkpoint")
    if method_deprecated:
        print(f"this method is deprecated: {method_deprecated}")
        print("please update nnunetv2 to the latest version, and update...")

    # load the full checkpoint
    model_checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    # load the dataset and plans
    dataset_json = model_checkpoint["dataset"]
    plans_json = model_checkpoint["plans"]
    plans_manager = PlansManager(plans_json)

    # load the model parameters
    parameters = []
    checkpoint_name = "final" # always use final checkpoint for now
    for i, k in enumerate(sorted(model_checkpoint['folds'])):

        checkpoint = model_checkpoint['folds'][k][checkpoint_name]
        
        if i == 0: # use first fold to get trainer and configuration name
            trainer_name = checkpoint["trainer_name"]
            configuration_name = checkpoint["init_args"]["configuration"]
            inference_allowed_mirroring_axes = (
                checkpoint["inference_allowed_mirroring_axes"]
                if "inference_allowed_mirroring_axes" in checkpoint.keys()
                else None
            )

        parameters.append(checkpoint["network_weights"])

    configuration_manager = plans_manager.get_configuration(configuration_name)

    # restore network
    num_input_channels = determine_num_input_channels(
        plans_manager, configuration_manager, dataset_json
    )
    trainer_class = recursive_find_python_class(
        os.path.join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
        trainer_name,
        "nnunetv2.training.nnUNetTrainer",
    )
    network = trainer_class.build_network_architecture(
        plans_manager,
        dataset_json,
        configuration_manager,
        num_input_channels,
        enable_deep_supervision=False,
    )
    model.plans_manager = plans_manager
    model.configuration_manager = configuration_manager
    model.list_of_parameters = parameters
    model.network = network
    model.dataset_json = dataset_json
    model.trainer_name = trainer_name
    model.allowed_mirroring_axes = inference_allowed_mirroring_axes
    model.label_manager = plans_manager.get_label_manager(dataset_json)
    if (
        ("nnUNet_compile" in os.environ.keys())
        and (os.environ["nnUNet_compile"].lower() in ("true", "1", "t"))
        and not isinstance(model.network, OptimizedModule)
    ):
        print("Using torch.compile")
        model.network = torch.compile(model.network)

    return model


def export_model_checkpoint(path: str, checkpoint_path: str = None) -> None:
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

    MODEL_CHECKPOINT = {}

    # paths
    DATASET_JSON_PATH = os.path.join(path, "model", "dataset.json")
    PLAN_JSON_PATH = os.path.join(path, "model", "plans.json")

    # load the dataset and plans
    MODEL_CHECKPOINT["dataset"] = load_json(DATASET_JSON_PATH)
    MODEL_CHECKPOINT["plans"] = load_json(PLAN_JSON_PATH)

    # load the folds
    MODEL_CHECKPOINT["folds"] = {}

    # get all the fold directories,
    FOLD_DIRS = sorted(glob.glob(os.path.join(path, "model", "fold_*")))
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
        checkpoint_path = os.path.join(path, "model_checkpoint.pth")
    torch.save(MODEL_CHECKPOINT, checkpoint_path)
    print(f"Saved model checkpoint to {checkpoint_path}")


def convert_to_nnunet_dataset(
    data_path, label_path, nnunet_data_path, label_map=None, filetype=".tif"
):
    """
    Convert a directory of images and labels to an nnunet dataset.
    """

    DATA_PATH = data_path
    LABEL_PATH = label_path
    LABEL_MAP = label_map
    NNUNET_DATA_PATH = nnunet_data_path
    FILETYPE = filetype

    # check if the NNUNET_DATA_PATH has a prefix of DATASET

    # check if is directory
    if os.path.exists(NNUNET_DATA_PATH) and not os.path.isdir(NNUNET_DATA_PATH):
        NNUNET_DATA_PATH = os.path.dirname(NNUNET_DATA_PATH)

    folder_basename = os.path.basename(NNUNET_DATA_PATH)

    if not folder_basename.startswith("Dataset"):
        # get the dirname of NNUNET_DATA_PATH
        dirname = os.path.dirname(NNUNET_DATA_PATH)
        # get all the folders in dirname
        n_folders = len(glob.glob(os.path.join(dirname, "Dataset*")))
        # get the next dataset number
        next_dataset_number = n_folders + 1

        # rename the folder
        new_nnunet_path = os.path.join(
            dirname, f"Dataset{next_dataset_number:03d}_{folder_basename}"
        )

        print(
            f"NNUnet requires the dataset folder to begin with DatasetXXX, where XXX is the number of the dataset."
        )
        ret = input(f"Rename {NNUNET_DATA_PATH} to {new_nnunet_path}? (y/n)")

        if ret.lower() == "y":
            NNUNET_DATA_PATH = new_nnunet_path

    NNUNET_IMAGES_PATH = os.path.join(NNUNET_DATA_PATH, "imagesTr")
    NNUNET_LABELS_PATH = os.path.join(NNUNET_DATA_PATH, "labelsTr")

    GLOB_PATTERN = f"*{FILETYPE}"
    SUPPORTED_PATTERNS = ["*.tif", "*.png"]

    if GLOB_PATTERN not in SUPPORTED_PATTERNS:
        raise ValueError(
            f"glob pattern {GLOB_PATTERN} not supported. Supported filetypes are: {SUPPORTED_PATTERNS}"
        )

    os.makedirs(NNUNET_IMAGES_PATH, exist_ok=True)
    os.makedirs(NNUNET_LABELS_PATH, exist_ok=True)

    paths = zip([DATA_PATH, LABEL_PATH], [NNUNET_IMAGES_PATH, NNUNET_LABELS_PATH])

    for path, out_path in paths:
        print(f"Copying files from {path}...")
        filenames = glob.glob(os.path.join(path, f"{GLOB_PATTERN}"))

        ret = input(f"Copy {len(filenames)} files to {out_path}? (y/n)")

        if ret == "y":
            for fname in tqdm(filenames):
                # copy file to new directory
                basename = os.path.basename(fname)

                # if this is the images, we need to prepend _0000 before the fileextension
                # because nnunet expects the images to be named like this??
                if "imagesTr" in out_path:
                    # check if it has _0000 already
                    if "_0000" not in basename:
                        basename = basename.replace(FILETYPE, f"_0000{FILETYPE}")

                new_fname = os.path.join(out_path, basename)
                shutil.copy(fname, new_fname)

    ret = input(f"Write dataset json to {NNUNET_DATA_PATH}? (y/n)")

    if ret.lower() == "y":
        # NOTE: could also use nnunetv2.dataset_conversion.generate_dataset_json.generate_dataset_json
        # but we have to get most of the info manually anyway...

        # load labels, get number of unique classes
        filenames = glob.glob(os.path.join(NNUNET_LABELS_PATH, f"{GLOB_PATTERN}"))
        dataset_json = {}
        dataset_json["channel_names"] = {"0": "map"}  # only support one channel for now
        dataset_json["numTraining"] = len(filenames)
        dataset_json["file_ending"] = FILETYPE

        # get unique labels
        unique_labels = set()
        for fname in tqdm(filenames):
            im = np.asarray(PIL.Image.open(fname))
            unique_labels.update(np.unique(im))

        # check if number of unique labels matches number of labels in LABEL_MAP
        if LABEL_MAP is not None:
            matched_labels = (len(unique_labels) == len(LABEL_MAP),)
            if not matched_labels:
                print(
                    f"WARNING: Number of unique labels ({len(unique_labels)}) does not match number of labels in LABEL_MAP ({len(LABEL_MAP)})"
                )
        else:
            print(f"WARNING: No label map provided. Using default label map.")
            LABEL_MAP = [f"label_{i:02d}" for i in range(len(unique_labels))]

        dataset_json["labels"] = {label: i for i, label in enumerate(LABEL_MAP)}

        DATASET_PATH = os.path.join(NNUNET_DATA_PATH, "dataset.json")
        print(f"Writing dataset json to: {DATASET_PATH}")

        with open(DATASET_PATH, "w") as f:
            json.dump(dataset_json, f, indent=4)

    # FINISHED
    n_images = len(glob.glob(os.path.join(NNUNET_IMAGES_PATH, f"{GLOB_PATTERN}")))
    n_labels = len(glob.glob(os.path.join(NNUNET_LABELS_PATH, f"{GLOB_PATTERN}")))

    print(f"-" * 50)
    print(f"Summary: ")
    print(f"{n_images} images in {NNUNET_IMAGES_PATH}")
    print(f"{n_labels} labels in {NNUNET_LABELS_PATH}")
    print(f"Number of unique labels: {len(unique_labels)}")
    print(f"Label Map used {LABEL_MAP}")
    print(f"FileType: {FILETYPE}")
    print(f"Dataset json written to: {DATASET_PATH}")

    # get last 3 chars, convert to int
    nnUnetRaw = os.path.dirname(os.path.dirname(NNUNET_DATA_PATH))
    DATASET_ID = int(os.path.basename(NNUNET_DATA_PATH)[7:10])  # DatasetXXX always

    print("-" * 50)
    print("To train nnUNet with this dataset:")
    print(f"Dataset ID: {DATASET_ID}")
    print(f"Dataset Path: {NNUNET_DATA_PATH}")

    print(f"\n1. Set the following environment variables:")
    print(f"nnUNet_raw={nnUnetRaw}")

    print(f"Currently, the environment variables are set to:")
    print(f"nnUNet_raw: {os.environ.get('nnUNet_raw')}")
    print(f"nnUNet_preprocessed: {os.environ.get('nnUNet_preprocessed')}")
    print(f"nnUNet_preprocessed: {os.environ.get('nnUNet_preprocessed')}")

    # pre-process the dataset
    print(f"\n2. Pre-Process the Dataset: ")
    print(f"nnUNetv2_plan_and_preprocess -d {DATASET_ID} --verify_dataset_integrity")

    # train the dataset
    print(f"\n3. Train the Model on the Dataset {DATASET_ID}: ")
    print(f"nnUNet_train {DATASET_ID} 2d all")

    print(f"Post Training: ")
    print(f"\nOnce training is complete, you can run inference on the dataset:")
    print(f"See doc/nnunet_inference.md for more details.")
    print(f"-" * 50)
