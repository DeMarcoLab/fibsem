import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import glob

Needle_tip_dist = []
Lamella_centre_dist = []
Lamella_left_edge_dist = []
Lamella_right_edge_dist = []
Image_centre_dist = []
Landing_post_dist = []

detection_features = {
    "NeedleTip": Needle_tip_dist,
    "LamellaCentre": Lamella_centre_dist,
    "LamellaLeftEdge": Lamella_left_edge_dist,
    "LamellaRightEdge": Lamella_right_edge_dist,
    "ImageCentre": Image_centre_dist,
    "LandingPost": Landing_post_dist,
}

euclid_dist_values = {
    "NeedleTip": [],
    "LamellaCentre": [],
    "LamellaLeftEdge": [],
    "LamellaRightEdge": [],
    "ImageCentre": [],
    "LandingPost": [],
}

labels = ["label_00","label_01","label_02","label_03"]

for label in labels:

    label_folder = label
    main_folder = r'C:\Users\rkan0039\Documents\detection_training\new_eval'
    report_folder_name = f"report_{label_folder}"
    eval_folder = os.path.join(main_folder, label_folder)
    report_folder_name = "*_eval.csv"
    report_folder_path = sorted(glob.glob(os.path.join(eval_folder, report_folder_name)))

    report = pd.read_csv(report_folder_path[-1])

    

    for feature in detection_features:

        mask1 = report["p1.type"] == feature

        output_x = report.loc[mask1, "p1.x_offset"].to_list()

        output_y = report.loc[mask1, "p1.y_offset"].to_list()

        output = [[x,y] for x, y in zip(output_x, output_y)]

        e_dist_ouput = report.loc[mask1, "p1.euc_dist"].to_list()

        euclid_dist_values[feature].extend(e_dist_ouput)

        detection_features[feature].extend(output)




        mask2 = report["p2.type"] == feature

        output_x = report.loc[mask2, "p2.x_offset"].to_list()

        output_y = report.loc[mask2, "p2.y_offset"].to_list()

        output = [[x,y] for x, y in zip(output_x, output_y)]

        detection_features[feature].extend(output)

        e_dist_ouput = report.loc[mask2, "p2.euc_dist"].to_list()

        euclid_dist_values[feature].extend(e_dist_ouput)




fig, ax = plt.subplots(1,2, figsize=(20,10))

for feature_name, pts in detection_features.items():
    x = [pt[0] for pt in pts]
    y = [pt[1] for pt in pts]
    ax[0].scatter(x,y, label=feature_name) 


f_names = []
e_values = []

for feature_name, x in euclid_dist_values.items():
    f_names.append(feature_name)
    avgs = np.array(x)
    avgs = np.average(avgs)
    e_values.append(avgs)


ax[0].set_title("Detection Summary")
ax[0].set_xlabel("x offset")
ax[0].set_ylabel("y offset")
ax[0].legend()
ax[0].axhline(y=0, color='black', linewidth=0.5)
ax[0].axvline(x=0, color='black', linewidth=0.5)

ax[1].set_title("Average Error by Euclidean Distance")
ax[1].bar(f_names, e_values)


plt.show()
fig.savefig("detection_summary.png")
# st.pyplot(plt)
