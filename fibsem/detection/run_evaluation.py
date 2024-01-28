import argparse
import os

import pandas as pd
import yaml

from fibsem.detection import evaluation

def main(config: dict):

    os.makedirs(config["save_path"], exist_ok=True)

    if config["run_eval"]:
        df_eval = evaluation.run_evaluation_v2(path=config["data_path"],
                                             image_path = config["images_path"], 
                                                checkpoints=config["checkpoints"], 
                                                plot=config["show_det_plot"],
                                                save=config["save_det_plot"], 
                                                save_path=config["save_path"])
    else:
        df_eval = pd.read_csv(os.path.join(config["save_path"], "eval.csv"))

    if config["plot_eval"]:
        category_orders = {"checkpoint": df_eval["checkpoint"].unique().tolist(), 
                    "feature": sorted(df_eval["feature"].unique().tolist())}
        evaluation.plot_evaluation_data(df=df_eval, 
                                        category_orders=category_orders,
                                        thresholds=config["thresholds"], 
                                        show=config["show_eval_plot"], 
                                        save=config["save_eval_plot"],
                                        save_path=config["save_path"])



if __name__ == "__main__":
# command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="the directory containing the config file to use",
        dest="config",
        action="store",
        default=os.path.join(os.path.join(os.path.dirname(__file__), "config.yml")),
    )
    args = parser.parse_args()
    config_dir = args.config

    # NOTE: Setup your config.yml file
    with open(config_dir, "r") as f:
        config = yaml.safe_load(f)

    main(config=config)