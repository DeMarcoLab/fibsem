
import glob
from fibsem.segmentation.model import SegmentationModel
import tifffile as tf
import matplotlib.pyplot as plt
from random import shuffle
import numpy as np 
import pandas as pd
# from autoscript_sdb_microscope_client.structures import AdornedImage

from fibsem.structures import Point
from fibsem.imaging import masks
from fibsem.detection.detection import NeedleTip, LamellaCentre, LamellaLeftEdge,LamellaRightEdge, ImageCentre, locate_shift_between_features_v2
import skimage

from pathlib import Path
from dataclasses import dataclass
from fibsem import conversions
from pprint import pprint
from fibsem.segmentation.model import load_model

from fibsem import config


### load evaluation folder 

eval_folder = r'C:\Users\Rohit\Documents\UNI\DEMARCO\new_eval\label_00\{}'

GT_data_file = r'C:\Users\Rohit\Documents\UNI\DEMARCO\new_eval\label_00\data.csv'

GT_data = pd.read_csv(GT_data_file)

fnames = GT_data["label"]
fnames_list = fnames.to_list()


# b = GT_data.loc[GT_data["label"]==a]
# print(b["p1.x"])
filenames = sorted(glob.glob(eval_folder.format('*.tif')))
checkpoint = r"C:\Users\Rohit\Documents\UNI\DEMARCO\seg_model_fibsem\model4.pt"
model = load_model(checkpoint,encoder="resnet34")

## ML_ evaulations

ML_p1_x = []
ML_p1_y = []

ML_p2_x = []
ML_p2_y = []

accuracy_p1 = []
accuracy_p2 = []

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

}
i=0

for filename in filenames:

    pic = filename.split('\\')[-1][:-4]
    print(f"{i}/{len(filenames)}")
    i+=1

    if pic in fnames_list:

        label_row = GT_data.loc[GT_data["label"]==pic]

        label = label_row["label"].values[0]
        p1_type = str(label_row["p1.type"].values[0])
        p2_type = str(label_row["p2.type"].values[0])
        p1_x = label_row["p1.x"].values[0]
        p1_y = label_row["p1.y"].values[0]
        p2_x = label_row["p2.x"].values[0]
        p2_y = label_row["p2.y"].values[0]

        labels_list.append(label)
        p1_x_list.append(p1_x)
        p1_y_list.append(p1_y)
        p2_x_list.append(p2_x)
        p2_y_list.append(p2_y)
        p1_type_list.append(p1_type)
        p2_type_list.append(p2_type)

        img = tf.imread(filename)

        res_x = img.shape[1]
        res_y = img.shape[0]

        mask = model.inference(img)

        features = [detection_types[p1_type], detection_types[p2_type]]

        det = locate_shift_between_features_v2(img, model, features=features, pixelsize=10e-9)

        #plot_det_result_EVAL(det,save=True,save_path=save_path)
        f1 = det.features[0]
        f2 = det.features[1]

        convert_p1_x = f1.feature_px.x/res_x
        convert_p1_y = f1.feature_px.y/res_y
        convert_p2_x = f2.feature_px.x/res_x
        convert_p2_y = f2.feature_px.y/res_y

        ML_p1_x.append(convert_p1_x)
        ML_p1_y.append(convert_p1_y)

        ML_p2_x.append(convert_p2_x)
        ML_p2_y.append(convert_p2_y)

        plt.imshow(det.image, cmap="gray")

        plt.plot(
            int(p1_x*res_x),
            int(p1_y*res_y),
            color="blue",
            marker="+",
            ms=20,
            label=f"{f1.name} GT",
        )
        plt.plot(
            int(p2_x*res_x),
            int(p2_y*res_y),
            color="blue",
            marker="x",
            ms=20,
            label=f"{f2.name} GT",
        )


        plt.plot(
            f1.feature_px.x,
            f1.feature_px.y,
            color="red",
            marker="+",
            ms=20,
            label=f"{f1.name} ML",
        )
        plt.plot(
            f2.feature_px.x,
            f2.feature_px.y,
            color="red",
            marker="x",
            ms=20,
            label=f"{f2.name} ML",
        )

        plt.legend()
        plt.show()
       
        # plt.savefig(r'C:\Users\Rohit\Documents\UNI\DEMARCO\new_eval\label_00\mask.tiff')

        if i == 1:
            break


        

data = {"label":labels_list,"p1.type":p1_type_list,"p1.x":p1_x_list,"p1.y":p1_y_list,"p2.type":p2_type_list,"p2.x":p2_x_list,"p2.y":p2_y_list,"ML_p1.x":ML_p1_x,"ML_p1.y":ML_p1_y,"ML_p2.x":ML_p2_x,"ML_p2.y":ML_p2_y}

df = pd.DataFrame(data)

df.to_csv(r"C:\Users\Rohit\Documents\UNI\DEMARCO\new_eval\label_00\ML_output.csv")



# p1_type = result_row["p1.type"]
# p2_type = result_row["p2.type"]

# print(f'label: {f1} p1_type: {p1_type}, p2_type: {p2_type}')

# labels = []
# p1_type = []
# p2_type = []
# p1_x = []
# p1_y = []
# p2_x = []
# p2_y = []


# assert len(filenames) > 0

# # model


# for i, fname in enumerate(filenames):

#     img = tf.imread(fname)

#     # inference
#     mask = model.inference(img)

#     # detect features
#     features = [NeedleTip(), LamellaCentre()]

#     det = locate_shift_between_features_v2(img, model, features=features, pixelsize=10e-9)
#     label = name = fname.split('\\')[-1][:-4]
#     labels.append(label)

#     #plot_det_result_EVAL(det,save=True,save_path=save_path)
#     f1 = det.features[0]
#     f2 = det.features[1]

#     p1_type.append(f1.name)
#     p2_type.append(f2.name)
#     p1_x.append(f1.feature_px.x)
#     p1_y.append(f1.feature_px.y)
#     p2_x.append(f2.feature_px.x)
#     p2_y.append(f2.feature_px.y)

#     print(f"image {(i+1)}/{len(filenames)} Feature: {f1.name}:  x: {f1.feature_px.x}, y: {f1.feature_px.y}  Feature: {f2.name}:  x: {f2.feature_px.x}, y: {f2.feature_px.y}")
#     data = {"label":labels,"p1_type":p1_type,"p1_x":p1_x,"p1_y":p1_y,"p2_type":p2_type,"p2_x":p2_x,"p2_y":p2_y}

#     if i == 0:
#         break


# df = pd.DataFrame(data)

# df.to_csv(r"C:\Users\Rohit\Documents\UNI\DEMARCO\new_eval\label_00\ML_output.csv")