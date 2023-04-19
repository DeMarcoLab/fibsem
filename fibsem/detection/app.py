import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px


st.title("Fibsem Detection")

fname = st.text_input("Enter file name", r"C:\Users\rkan0039\Documents\detection_training\new_eval\label_00\report_label_00\eval.csv")
df = pd.read_csv(fname)

st.write(df)


# groupby p1.type and plot average euc_dist
df_group = df.groupby("p1.type").mean().reset_index()

fig = px.bar(df_group, x="p1.type", y="p1_euc_dist")
st.plotly_chart(fig)


img_fname = r"C:\Users\rkan0039\Documents\detection_training\new_eval\label_03\2023-03-27.11-44-43AM_label.tif"
import tifffile as tff
img = tff.imread(img_fname)
st.image(img)


fig = px.scatter(df, x="p1.x_offset", y="p1.y_offset", color="p1.type", hover_data=["p1_euc_dist"])
st.plotly_chart(fig)