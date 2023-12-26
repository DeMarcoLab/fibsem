import glob
import json
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


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


def inference(model, image: np.ndarray):
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
    method_deprecated = hasattr(model, "load_from_checkpoint")
    print(f"this method is deprecated: {method_deprecated}")
    if method_deprecated:
        print("please update nnunetv2 to the latest version, and update...")

    # load the full checkpoint
    model_checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    # load the dataset and plans
    dataset_json = model_checkpoint["dataset"]
    plans_json = model_checkpoint["plans"]
    plans_manager = PlansManager(plans_json)

    # load the model parameters
    parameters = []
    checkpoint_name = "final"  # always use final checkpoint for now
    for k in sorted(model_checkpoint["folds"]):
        checkpoint = model_checkpoint["folds"][k][checkpoint_name]

        if k == "fold_0":
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
