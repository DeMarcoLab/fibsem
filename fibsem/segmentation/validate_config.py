import os

def validate_config(config, func):
    if func == "labelling":
        if "raw_dir" not in config[func]:
            raise ValueError("raw_dir is missing. Should point to path containing unlabelled images.")
        else:
            path = config[func]["raw_dir"]
            if not os.path.exists(path):
                raise ValueError(f"{path} directory does not exist. (raw_dir)")
        if "data_dir" not in config[func]:
            raise ValueError("data_dir is missing. Should point to path where labelled images will be saved.")
        else:
            path = config[func]["data_dir"]
            if not os.path.exists(path):
                raise ValueError(f"{path} directory does not exist. (data_dir)")
        print("\nConfig file validated. \n")
        return

    elif func == "train":
        if "data_dir" not in config[func]:
            raise ValueError("data_dir is missing. Should point to path containing labelled images.")
        else:
            path = config[func]["data_dir"]
            if not os.path.exists(path):
                raise ValueError(f"{path} directory does not exist. (data_dir)")
        if "save_dir" not in config[func]:
            raise ValueError("save_dir is missing. Should point to save the model.")
        else:
            path = config[func]["save_dir"]
            if not os.path.exists(path):
                raise ValueError(f"{path} directory does not exist. (save_dir)")
        if "wandb" not in config[func]:
            raise ValueError("wandb is missing. Used to enable/disable wandb logging in training loop. Should be a boolean value.")
        else:
            val = config[func]["wandb"]
            if type(val) != bool:
                raise TypeError(f"{val} is not a boolean (True/False). (wandb)")
        if "cuda" not in config[func]:
            raise ValueError("cuda is missing. used to enable/disable cuda training. Should be a boolean value.")
        else:
            val = config[func]["cuda"]
            if type(val) != bool:
                raise TypeError(f"{val} is not a boolean (True/False). (cuda)")
        if "checkpoint" not in config[func]:
            raise ValueError("checkpoint is missing. Either a path leading to the desired saved model, or None value.")
        else:
            path = config[func]["checkpoint"]
            if path == type(str):
                if not os.path.exists(path):
                    raise ValueError(f"{path} directory does not exist. (checkpoint)")
            elif path != None:
                raise ValueError(f"{path} directory does not exist. (checkpoint)")
        if "encoder" not in config[func]:
            raise ValueError("encoder is missing. Used to specify which model architecture to use. Default is resnet18.")
        else:
            val = config[func]["encoder"]
            if type(val) != str:
                raise TypeError(f"{val} must be a string. (encoder)")
            elif val not in unet_encoders:
                raise ValueError(f"{val} not a valid encoder. Check readme for full list. (encoder)")
        if "epochs" not in config[func]:
            raise ValueError("epochs is missing. Integer value used to determine number of epochs model trains for.")
        else:
            val = config[func]["epochs"]
            if type(val) != int or val <= 0:
                raise TypeError(f"{val} is not a positive integer. (epochs)")  
        if "batch_size" not in config[func]:
            raise ValueError("batch_size is missing. Integer value used to determine batch size of dataset.")
        else:
            val = config[func]["batch_size"]
            if type(val) != int or val <= 0:
                raise TypeError(f"{val} is not a positive integer. (batch_size)")    
        if "num_classes" not in config[func]:
            raise ValueError("num_classes is missing. Integer value used to determine number of classes model classifies.")
        else:
            val = config[func]["num_classes"]
            if type(val) != int or val <= 0:
                raise TypeError(f"{val} is not a positive integer. (num_classes)")  
        if "optimizer" not in config[func]:
            raise ValueError("optimizer is missing. String value indicating whether Adam or SGD should be used.")
        else:
            val = config[func]["optimizer"]
            if type(val) == str:
                val = str.lower(val)
                if val != "adam" and val != "sgd":
                    raise ValueError(f"Optimizer must be either adam or sgd, not {val}.")
            else:
                raise TypeError(f"{val} is not a string. (optimizer)")  
        if "learning_rate" not in config[func]:
            raise ValueError("learning_rate is missing. Float value indicating the learning rate of the model.")
        else:
            val = config[func]["learning_rate"]
            if type(val) == float:
                if val <= 0:
                    raise ValueError(f"{val} must be a positive float value (learning_rate).")
            else:
                raise TypeError(f"{val} is not a float. (learning_rate)")  
        if "wandb_project" not in config[func]:
            raise ValueError("wandb_project is missing. String indicating the wandb project title for login.")
        else:
            val = config[func]["wandb_project"]
            if type(val) != str:
                raise TypeError(f"{val} is not a string. (wandb_project)")
        if "wandb_entity" not in config[func]:
            raise ValueError("wandb_entity is missing. String indicating the wandb login credentials.")
        else:
            val = config[func]["wandb_entity"]
            if type(val) != str:
                raise TypeError(f"{val} is not a string. (wandb_project)")
        print("\nConfig file validated.\n")
        return
    elif func == "inference":
        if "data_dir" not in config[func]:
            raise ValueError("data_dir is missing. Should point to path containing images.")
        else:
            path = config[func]["data_dir"]
            if not os.path.exists(path):
                raise ValueError(f"{path} directory does not exist. (data_dir)")
        if "model_dir" not in config[func]:
            raise ValueError("model_dir is missing. Should point to where the model is saved.")
        else:
            path = config[func]["model_dir"]
            if not os.path.exists(path):
                raise ValueError(f"{path} directory does not exist. (model_dir)")
        if "output_dir" not in config[func]:
            raise ValueError("output_dir is missing. Should point to where the outputs are saved.")
        else:
            path = config[func]["output_dir"]
            if not os.path.exists(path):
                raise ValueError(f"{path} directory does not exist. (output_dir)")
        if "wandb" not in config[func]:
            raise ValueError("wandb is missing. Used to enable/disable wandb logging in training loop. Should be a boolean value.")
        else:
            val = config[func]["wandb"]
            if type(val) != bool:
                raise TypeError(f"{val} is not a boolean (True/False). (wandb)")
        if "cuda" not in config[func]:
            raise ValueError("cuda is missing. used to enable/disable cuda training. Should be a boolean value.")
        else:
            val = config[func]["cuda"]
            if type(val) != bool:
                raise TypeError(f"{val} is not a boolean (True/False). (cuda)")
        if "encoder" not in config[func]:
            raise ValueError("encoder is missing. Used to specify which model architecture to use. Default is resnet18.")
        else:
            val = config[func]["encoder"]
            if type(val) != str:
                raise TypeError(f"{val} must be a string. (encoder)")
            elif val not in unet_encoders:
                raise ValueError(f"{val} not a valid encoder. Check readme for full list. (encoder)") 
        if "num_classes" not in config[func]:
            raise ValueError("num_classes is missing. Integer value used to determine number of classes model classifies.")
        else:
            val = config[func]["num_classes"]
            if type(val) != int or val <= 0:
                raise TypeError(f"{val} is not a positive integer. (num_classes)")   
        if "wandb_project" not in config[func]:
            raise ValueError("wandb_project is missing. String indicating the wandb project title for login.")
        else:
            val = config[func]["wandb_project"]
            if type(val) != str:
                raise TypeError(f"{val} is not a string. (wandb_project)")
        if "wandb_entity" not in config[func]:
            raise ValueError("wandb_entity is missing. String indicating the wandb login credentials.")
        else:
            val = config[func]["wandb_entity"]
            if type(val) != str:
                raise TypeError(f"{val} is not a string. (wandb_project)")
        print("\nConfig file validated.\n")
        return
        
# All UNet encoders that work with Imagenet weights
unet_encoders = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x16d",
    "resnext101_32x32d",
    "resnext101_32x48d",
    "dpn68",
    "dpn98",
    "dpn131",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",
    "senet154",
    "se_resnet50",
    "se_resnet101",
    "se_resnet152",
    "se_resnext50_32x4d",
    "se_resnext101_32x4d",
    "densenet121",
    "densenet169",
    "densenet201",
    "densenet161",
    "efficientnet-b0",
    "efficientnet-b1",
    "efficientnet-b2",
    "efficientnet-b3",
    "efficientnet-b4",
    "efficientnet-b5",
    "efficientnet-b6",
    "efficientnet-b7",
    "mobilenet_v2",
    "efficientnet-b0",
    "xception"
]
        