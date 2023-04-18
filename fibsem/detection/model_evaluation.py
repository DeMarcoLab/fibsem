
import glob
from fibsem.segmentation.model import SegmentationModel
import tifffile as tf
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd

from fibsem.detection.detection import NeedleTip, LamellaCentre, LamellaLeftEdge,LamellaRightEdge, ImageCentre, LandingPost, locate_shift_between_features_v2
from fibsem.segmentation.model import load_model
import os
from tqdm import tqdm
import time

# setting up folders for evaluation
label_folders = ["label_00"]

# setting up model for inference
checkpoint_path = r"C:\Users\rkan0039\Documents\detection_training\models"
checkpoint_name = "model4.pt"
checkpoint = os.path.join(checkpoint_path,checkpoint_name)
encoder = "resnet34"
model = load_model(checkpoint,encoder=encoder)

# Main loop that goes through each folder and evaluates the model

for label_folder in tqdm(label_folders,desc=f'Evaluating folders'):

    # load evaluation folder 
    main_folder = r'C:\Users\rkan0039\Documents\detection_training\new_eval'
    date_time_test_conducted = time.strftime("%Y-%m-%d--%H-%M-%S")
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
    
    # ML_ evaulations
    # pre-creating lists to put ML data into for csv output
    ML_p1_x = []
    ML_p1_y = []

    ML_p2_x = []
    ML_p2_y = []

    p1_x_offset = []
    p1_y_offset = []
    p2_x_offset = []
    p2_y_offset = []
    p1_euc_dist = []
    p2_euc_dist = []

    labels_list = []
    p1_x_list =[]
    p1_y_list = []
    p2_x_list = []
    p2_y_list = []
    p1_type_list = []
    p2_type_list = []

    detection_types = {
        "NeedleTip": NeedleTip(),
        "LamellaCentre": LamellaCentre(),
        "LamellaLeftEdge":LamellaLeftEdge(),
        "LamellaRightEdge":LamellaRightEdge(),
        "ImageCentre": ImageCentre(),
        "LandingPost": LandingPost(),

    }

    i=0

    # main loop that goes through each image and evaluates the model
    for filename in tqdm(filenames,desc="Evaluating Images"):

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

            # appending data to lists
            labels_list.append(label)
            p1_x_list.append(p1_x)
            p1_y_list.append(p1_y)
            p2_x_list.append(p2_x)
            p2_y_list.append(p2_y)
            p1_type_list.append(p1_type)
            p2_type_list.append(p2_type)

            # reading image and running inference
            img = tf.imread(filename)
            res_x = img.shape[1]
            res_y = img.shape[0]
            mask = model.inference(img)
            # segmenting image
            features = [detection_types[p1_type], detection_types[p2_type]]
            det = locate_shift_between_features_v2(img, model, features=features, pixelsize=10e-9)
            f1 = det.features[0]
            f2 = det.features[1]

            # converting pixel coordinates to distance along axes, same as GT
            convert_p1_x = f1.feature_px.x/res_x
            convert_p1_y = f1.feature_px.y/res_y
            convert_p2_x = f2.feature_px.x/res_x
            convert_p2_y = f2.feature_px.y/res_y

            # Appending ML data to lists
            ML_p1_x.append(convert_p1_x)
            ML_p1_y.append(convert_p1_y)
            ML_p2_x.append(convert_p2_x)
            ML_p2_y.append(convert_p2_y)

            # calculating offset and euclidean distance
            # GT value is subtracted from ML value, meaning + offset is when ML predicition is to the left of the GT when viewed on the image (x axis) and above the GT when viewed on the image (y axis)
            p1_x_offset.append(p1_x - convert_p1_x)
            p1_y_offset.append(p1_y - convert_p1_y)
            p2_x_offset.append(p2_x - convert_p2_x)
            p2_y_offset.append(p2_y - convert_p2_y)

            p1_euc_dist.append(np.sqrt((p1_x - convert_p1_x)**2 + (p1_y - convert_p1_y)**2))
            p2_euc_dist.append(np.sqrt((p2_x - convert_p2_x)**2 + (p2_y - convert_p2_y)**2))



            # creating figures for review
            # each feature is a seperate colour
            # GT is marked with a + and ML is marked with an x
            
            fig,ax = plt.subplots(1, 2, figsize=(12, 7))

            ax[0].imshow(det.image, cmap="gray")
            ax[1].imshow(det.mask)

            ax[1].plot(
                f1.feature_px.x,
                f1.feature_px.y,
                color="blue",
                marker="x",
                ms=20,
                label=f"{f1.name} ML"
            )

            ax[1].plot(
                f2.feature_px.x,
                f2.feature_px.y,
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
                f1.feature_px.x,
                f1.feature_px.y,
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
                f2.feature_px.x,
                f2.feature_px.y,
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


    Evaluation_ID = [
    f"Report Conducted: {date_time_test_conducted}",
    f"Model: {checkpoint_name}",
    f'Encoder: {encoder}'
    ]
            

    data = {
        "label":labels_list,
        "p1.type":p1_type_list,
        "p1.x":p1_x_list,
        "p1.y":p1_y_list,
        "p2.type":p2_type_list,
        "p2.x":p2_x_list,
        "p2.y":p2_y_list,
        "ML_p1.x":ML_p1_x,
        "ML_p1.y":ML_p1_y,
        "ML_p2.x":ML_p2_x,
        "ML_p2.y":ML_p2_y,
        "p1.x_offset":p1_x_offset,
        "p1.y_offset":p1_y_offset,
        "p2.x_offset":p2_x_offset,
        "p2.y_offset":p2_y_offset,
        "p1_euc_dist":p1_euc_dist,
        "p2_euc_dist":p2_euc_dist,
        }

    df = pd.DataFrame(data)
    eval_info = pd.Series(Evaluation_ID)
    df["Evaluation Info"] = eval_info

    report_name = f"Report_{label_folder}.csv"

    df.to_csv(os.path.join(eval_folder,report_name))


