from multiprocessing.sharedctypes import Value
import yaml
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

    if func == "train":
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
        if "debug" not in config[func]:
            raise ValueError("debug is missing. Used to enable/disable debugging in training loop. Should be a boolean value.")
        else:
            val = config[func]["debug"]
            if type(val) != bool:
                raise TypeError(f"{val} is not a boolean (True/False). (debug)")
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
        print("\nConfig file validated.\n")
        return
        
        