import streamlit as st
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob


from fibsem.structures import FibsemImage

from fibsem.detection import detection

import os
from copy import deepcopy
from pathlib import Path
import numpy as np

from fibsem.detection import detection, evaluation
from fibsem.detection import utils as det_utils
from fibsem.detection.detection import DetectedFeatures
from fibsem.segmentation import model as fibsem_model
from fibsem.segmentation.model import load_model
from fibsem.structures import (
    BeamType,
    FibsemImage,
    Point,
)
import logging
import plotly.express as px

st.set_page_config(layout="wide")
st.title("FIBSEM ML Evaluation")

DATA_PATH = "/home/patrick/github/fibsem/fibsem/log/data/"
sample_names = ["dm-embryo", "yeast", "celegans"]

# read csv
SAMPLE_NAME = sample_names[2] 
SAMPLE_PATH = os.path.join(DATA_PATH, SAMPLE_NAME)

SAMPLE_PATH = st.text_input("Data Path", SAMPLE_PATH)

CHECKPOINT_PATH = os.path.join(os.path.dirname(fibsem_model.__file__), "models", "model4.pt")
# CHECKPOINT_PATH = "/home/patrick/github/fibsem/fibsem/segmentation/models/dm-embryo/30_04_2023_18_42_40_n20_model.pt"
# CHECKPOINT_PATH = "/home/patrick/github/fibsem/fibsem/segmentation/models/celegans/30_04_2023_19_01_11_n20_model.pt"

cols = st.columns(3)
CHECKPOINT_PATH = cols[0].text_input("Checkpoint Path", CHECKPOINT_PATH)
ENCODER = cols[1].selectbox("Encoder", ["resnet18", "resnet34", "resnet50", "resnet101"], index=1)
NUM_CLASSES = cols[2].number_input("Number of Classes", min_value=1, max_value=10, value=3, step=1)

button = st.button("Run Evaluation")

if button:
    df = evaluation.run_eval(SAMPLE_PATH, checkpoint=CHECKPOINT_PATH, encoder=ENCODER, num_classes=NUM_CLASSES, show=False)

df = pd.read_csv(os.path.join(SAMPLE_PATH, "eval.csv"))

# group by feature, calculate mean distance, plot
# sort by distance
df = df.sort_values(by="distance", ascending=False)
st.dataframe(df, use_container_width=True)

# sort by feature
df = df.sort_values(by="feature")

df_group = df.groupby("feature").mean()
df_group = df_group.sort_values(by="feature")

fig = px.box(df, x="feature", y="distance", points="all", color="feature", title="Distance from Ground Truth to Predicted")



cols = st.columns(2)
cols[0].plotly_chart(fig)

# plot dx, dy, color by feature
fig = px.scatter(df, x="dx", y="dy", color="feature", title="Distance from Ground Truth to Predicted")
cols[1].plotly_chart(fig)





# Model Evaluation
st.header("Model Evaluation")

cols = st.columns(3)
FEATURES = cols[0].multiselect("Features", df["feature"].unique(), default=df["feature"].unique())

# filter df 
df = df[df["feature"].isin(FEATURES)]

FNAME = cols[1].selectbox("Image", df["fname"].unique()) 

@st.cache_data
def model_eval(SAMPLE_PATH, CHECKPOINT_PATH, ENCODER, NUM_CLASSES, FNAME):
    image = FibsemImage.load(os.path.join(SAMPLE_PATH, f"{FNAME}.tif"))
    model = load_model(checkpoint=CHECKPOINT_PATH, encoder=ENCODER, nc=NUM_CLASSES)

    df_gt = pd.read_csv(os.path.join(SAMPLE_PATH, "data.csv"))
    gt = df_gt[df_gt["image"] == FNAME]
    features = gt["feature"].to_list()
    features = [detection.get_feature(feature) for feature in features]
    gt_det = detection._det_from_df(gt, SAMPLE_PATH, FNAME)

    det = detection.detect_features(image.data, model, features, pixelsize=25e-9)

    return gt_det, det

gt_det, det = model_eval(SAMPLE_PATH, CHECKPOINT_PATH, ENCODER, NUM_CLASSES, FNAME)
fig = detection.plot_detections([gt_det, det], titles=["Ground Truth", "Prediction"])
st.pyplot(fig)
