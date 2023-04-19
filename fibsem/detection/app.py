import streamlit as st
import pandas as pd
import numpy as np
import os
import glob

import plotly.express as px


st.title("Fibsem Detection")

# fname = st.text_input("Enter file name", r"C:\Users\rkan0039\Documents\detection_training\new_eval\label_00\report_label_00\eval.csv")
# df = pd.read_csv(fname)



labels = ["label_00","label_01","label_02","label_03"]

combined_dfs = []

for label in labels:

    label_folder = label
    main_folder = r'C:\Users\rkan0039\Documents\detection_training\new_eval'
    report_folder_name = f"report_{label_folder}"
    eval_folder = os.path.join(main_folder, label_folder)
    report_folder_name = "*_eval.csv"
    report_folder_path = sorted(glob.glob(os.path.join(eval_folder, report_folder_name)))

    report = pd.read_csv(report_folder_path[-1])


    combined_dfs.append(report)

main_df = pd.concat(combined_dfs)


st.write(main_df)


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


summary_list  = []

data = {}

from copy import deepcopy

for feature in detection_features:

        mask1 = main_df["p1.type"] == feature

        output_x = main_df.loc[mask1, "p1.x_offset"].to_list()

        output_y = main_df.loc[mask1, "p1.y_offset"].to_list()

        output_euc = main_df.loc[mask1, "p1.euc_dist"].to_list()

        for i in range(len(output_x)):

            data["Feature_Type"] = feature

            data["x_offset"] = output_x[i]

            data["y_offset"] = output_y[i]

            data["euc_dist"] = output_euc[i]

            summary_list.append(deepcopy(data))





        mask2 = main_df["p2.type"] == feature

        output_x = main_df.loc[mask2, "p2.x_offset"].to_list()

        output_y = main_df.loc[mask2, "p2.y_offset"].to_list()
        
        output_euc = main_df.loc[mask2, "p2.euc_dist"].to_list()


        for i in range(len(output_x)):

            data["Feature_Type"] = feature

            data["x_offset"] = output_x[i]

            data["y_offset"] = output_y[i]

            data["euc_dist"] = output_euc[i]

            summary_list.append(deepcopy(data))


feature_list_df = pd.DataFrame(summary_list)

st.write(feature_list_df)

# groupby p1.type and plot average euc_dist
df_group = feature_list_df.groupby("Feature_Type").mean().reset_index()

fig = px.bar(df_group, x="Feature_Type", y="euc_dist")
st.plotly_chart(fig)


img_fname = r"C:\Users\rkan0039\Documents\detection_training\new_eval\label_03\2023-03-27.11-44-43AM_label.tif"
import tifffile as tff
img = tff.imread(img_fname)
st.image(img)


fig = px.scatter(feature_list_df, x="x_offset", y="y_offset", color="Feature_Type", hover_data=["euc_dist"])
fig.update_layout(
    xaxis=dict(showline=True, linewidth=2, linecolor='white', mirror=True, zeroline=True, zerolinecolor='gray', zerolinewidth=2),
    yaxis=dict(showline=True, linewidth=2, linecolor='white', mirror=True, zeroline=True, zerolinecolor='gray', zerolinewidth=2)
)

st.plotly_chart(fig)