
import glob
from fibsem.segmentation.model import SegmentationModel
import tifffile as tf
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

from fibsem.detection.detection import NeedleTip, LamellaCentre, LamellaLeftEdge,LamellaRightEdge, ImageCentre, LandingPost, detect_features
from fibsem.segmentation.model import load_model
import os
from tqdm import tqdm
import time

# setting up folders for evaluation
label_folders = ["label_00","label_01","label_02","label_03"]

# setting up model for inference
checkpoint_path = r"C:\Users\rkan0039\Documents\detection_training\models"
checkpoint_name = "model4.pt"
checkpoint = os.path.join(checkpoint_path,checkpoint_name)
encoder = "resnet34"
model = load_model(checkpoint,encoder=encoder)

save_images = False

# Main loop that goes through each folder and evaluates the model

for label_folder in tqdm(label_folders,desc=f'Evaluating folders'):

    # load evaluation folder 
    main_folder = r'C:\Users\rkan0039\Documents\detection_training\new_eval'
    date_time_test_conducted = time.strftime("%Y_%m_%d__%H_%M_%S")
    report_folder_name = f"report_{label_folder}"
    eval_folder = os.path.join(main_folder, label_folder)

    # loading ground truth data
    GT_data_file = os.path.join(eval_folder, "data.csv")
    GT_data = pd.read_csv(GT_data_file)
    fnames = GT_data["label"]
    fnames_list = fnames.to_list()

    # creating output report folder
    report_folder_path = os.path.join(eval_folder, report_folder_name)
    if not os.path.exists(report_folder_path):
        os.makedirs(report_folder_path)

    filenames = sorted(glob.glob(os.path.join(eval_folder,'*.tif')))
    

    detection_types = {
        "NeedleTip": NeedleTip(),
        "LamellaCentre": LamellaCentre(),
        "LamellaLeftEdge":LamellaLeftEdge(),
        "LamellaRightEdge":LamellaRightEdge(),
        "ImageCentre": ImageCentre(),
        "LandingPost": LandingPost(),

    }

    i=0
    data_list = []

    Evaluation_ID = [
    f"Report Conducted: {date_time_test_conducted}",
    f"Model: {checkpoint_name}",
    f'Encoder: {encoder}'
    ]

    # main loop that goes through each image and evaluates the model
    for  filename in tqdm(filenames,desc="Evaluating Images"):

        # takes the file name and removes the file path and .tif
        pic = filename.split('\\')[-1][:-4]

        # not all images have ground truth data so this checks if the image has ground truth data
        if pic in fnames_list:
            
            # loading ground truth data
            label_row = GT_data.loc[GT_data["label"]==pic]
            label = label_row["label"].values[0]
            p1_type = str(label_row["p1.type"].values[0])
            p2_type = str(label_row["p2.type"].values[0])
            p1_x = label_row["p1.x"].values[0]
            p1_y = label_row["p1.y"].values[0]
            p2_x = label_row["p2.x"].values[0]
            p2_y = label_row["p2.y"].values[0]


            from copy import deepcopy
            dat = {"label":label,"p1.type":p1_type,"p2.type":p2_type,"p1.x":p1_x,"p1.y":p1_y,"p2.x":p2_x,"p2.y":p2_y}

            # reading image and running inference
            img = tf.imread(filename)
            res_x = img.shape[1]
            res_y = img.shape[0]
            mask = model.inference(img)
            # segmenting image
            features = [detection_types[p1_type], detection_types[p2_type]]
            det = detect_features(img, model, features=features, pixelsize=10e-9)
            f1 = det.features[0]
            f2 = det.features[1]

            # converting pixel coordinates to distance along axes, same as GT
            convert_p1_x = f1.px.x/res_x
            convert_p1_y = f1.px.y/res_y
            convert_p2_x = f2.px.x/res_x
            convert_p2_y = f2.px.y/res_y

            dat["ML_p1_x"] = convert_p1_x
            dat["ML_p1_y"] = convert_p1_y
            dat["ML_p2_x"] = convert_p2_x
            dat["ML_p2_y"] = convert_p2_y

            dat["p1.x_offset"] = convert_p1_x - p1_x
            dat["p1.y_offset"] = convert_p1_y - p1_y
            dat["p2.x_offset"] = convert_p2_x - p2_x
            dat["p2.y_offset"] = convert_p2_y - p2_y

            dat["p1.euc_dist"] = np.sqrt((p1_x - convert_p1_x)**2 + (p1_y - convert_p1_y)**2)
            dat["p2.euc_dist"] = np.sqrt((p2_x - convert_p2_x)**2 + (p2_y - convert_p2_y)**2)

            dat["datetime"] = date_time_test_conducted
            dat["model"] = checkpoint_name
            dat["encoder"] = encoder

            data_list.append(deepcopy(dat))           

            # creating figures for review
            # each feature is a seperate colour
            # GT is marked with a + and ML is marked with an x
            
            if save_images:
                fig,ax = plt.subplots(1, 2, figsize=(12, 7))

                ax[0].imshow(det.image, cmap="gray")
                ax[1].imshow(det.mask)

                ax[1].plot(
                    f1.px.x,
                    f1.px.y,
                    color="blue",
                    marker="x",
                    ms=20,
                    label=f"{f1.name} ML"
                )

                ax[1].plot(
                    f2.px.x,
                    f2.px.y,
                    color="white",
                    marker="x",
                    ms=20,
                    label=f"{f2.name} ML"
                )


                ax[0].plot(
                    int(p1_x*res_x),
                    int(p1_y*res_y),
                    color="blue",
                    marker="+",
                    ms=20,
                    label=f"{f1.name} GT",
                )
                
                ax[0].plot(
                    f1.px.x,
                    f1.px.y,
                    color="blue",
                    marker="x",
                    ms=20,
                    label=f"{f1.name} ML",
                )

                ax[0].plot(
                    int(p2_x*res_x),
                    int(p2_y*res_y),
                    color="red",
                    marker="+",
                    ms=20,
                    label=f"{f2.name} GT",
                )

                ax[0].plot(
                    f2.px.x,
                    f2.px.y,
                    color="red",
                    marker="x",
                    ms=20,
                    label=f"{f2.name} ML",
                )

                ax[0].legend()
                ax[1].legend()
                plt.title(f"{label}")
                
            
                report_fig_name = f"Report_{label}.png"

                fig.savefig(os.path.join(report_folder_path,report_fig_name))

                plt.close(fig)

    eval_name = f"{date_time_test_conducted}_eval.csv"

    df = pd.DataFrame(data_list)
    df.to_csv(os.path.join(eval_folder,eval_name))

