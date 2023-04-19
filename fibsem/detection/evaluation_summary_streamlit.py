import os
import streamlit as st
import numpy as np
import pandas as pd

label_folder = "label_00"
main_folder = r'C:\Users\rkan0039\Documents\detection_training\new_eval'
report_folder_name = f"report_{label_folder}"
eval_folder = os.path.join(main_folder, label_folder)
report_folder_name = report_name = f"Report_{label_folder}.csv"
report_folder_path = os.path.join(eval_folder, report_folder_name)

report = pd.read_csv(report_folder_path)



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

labels = ["label_00","label_01","label_02","label_03"]

for label in labels:

    label_folder = label
    main_folder = r'C:\Users\rkan0039\Documents\detection_training\new_eval'
    report_folder_name = f"report_{label_folder}"
    eval_folder = os.path.join(main_folder, label_folder)
    report_folder_name = report_name = f"Report_{label_folder}.csv"
    report_folder_path = os.path.join(eval_folder, report_folder_name)

    report = pd.read_csv(report_folder_path)


    for feature in detection_features:

        mask1 = report["p1.type"] == feature

        output = report.loc[mask1, "p1_euc_dist"]

        output = output.to_list()

        detection_features[feature].extend(output)

        mask2 = report["p2.type"] == feature

        output = report.loc[mask2, "p2_euc_dist"]

        output = output.to_list()

        detection_features[feature].extend(output)


s1 = pd.Series(detection_features["NeedleTip"], name="NeedleTip")
s2 = pd.Series(detection_features["LamellaCentre"], name="LamellaCentre")
s3 = pd.Series(detection_features["LamellaLeftEdge"], name="LamellaLeftEdge")
s4 = pd.Series(detection_features["LamellaRightEdge"], name="LamellaRightEdge")
s5 = pd.Series(detection_features["ImageCentre"], name="ImageCentre")
s6 = pd.Series(detection_features["LandingPost"], name="LandingPost")

data = pd.concat([s1, s2, s3, s4, s5, s6], axis=1)


print(data)
    

# chart_data = report

# st.line_chart(chart_data,y=["p1_euc_dist","p2_euc_dist"])