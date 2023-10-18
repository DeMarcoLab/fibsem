import streamlit as st


from autolamella import config as cfg
import os

import plotly.io as pio
import pandas as pd

import plotly.express as px

st.set_page_config(layout="wide")

pio.templates.default = "plotly_white"

conn = st.experimental_connection("fibsem.db", type="sql", url = "sqlite:///fibsem.db")

# select project
df = conn.query("SELECT * from projects")

PROJECT_NAMES = df["name"].values.tolist()
PROJECT_NAME = st.sidebar.selectbox("Select project", PROJECT_NAMES)

PROJECT_ID = df[df["name"] == PROJECT_NAME]["id"].values[0]

# select experiment from project
df = conn.query(f"SELECT * FROM experiments WHERE project_id = {PROJECT_ID}")

EXPERIMENT_NAMES = df["name"].values.tolist()
EXPERIMENT_NAME = st.sidebar. selectbox("Select experiment", EXPERIMENT_NAMES)
EXPERIMENT_ID = df[df["name"] == EXPERIMENT_NAME]["id"].values[0]

# show data from experiment
# df = conn.query(f"SELECT * FROM experiments WHERE project_id = {PROJECT_ID} AND name = '{EXPERIMENT_NAME}'")


# show data from experiment

st.header(f"Experiment: {EXPERIMENT_NAME}")


# HISTORY
df_history = conn.query(f"SELECT e.name, e.date, h.petname, h.start, h.end, h.duration, h.stage FROM history h JOIN experiments e ON e.id = h.experiment_id WHERE h.experiment_id = {EXPERIMENT_ID} ")


# Duration
cols = st.columns(2)
fig_duration = px.bar(df_history, x="petname", y="duration", color="stage", barmode="group", hover_data=df_history.columns, title="Lamella Duration by Stage")
cols[0].plotly_chart(fig_duration, use_container_width=True)

# Duration
fig_duration = px.bar(df_history, x="stage", y="duration", color="petname", barmode="group", hover_data=df_history.columns, title="Stage Duration by Lamella")
cols[1].plotly_chart(fig_duration, use_container_width=True)

# STEPS
df_steps = conn.query(f"SELECT * FROM steps WHERE experiment_id = {EXPERIMENT_ID}")

# convert timestamp to datetime, aus timezone 
df_steps.timestamp = pd.to_datetime(df_steps.timestamp)

# convert timestamp to australian timezone
df_steps.timestamp = df_steps.timestamp.dt.tz_convert("Australia/Sydney")

fig_timeline = px.scatter(df_steps, x="step_n", y="timestamp", color="stage", symbol="petname",
    title="AutoLamella Timeline", 
    hover_name="stage", hover_data=df_steps.columns)
    # size = "duration", size_max=20)
st.plotly_chart(fig_timeline, use_container_width=True)



# DETECTIONS
df_det = conn.query(f"SELECT * FROM detections WHERE experiment_id = {EXPERIMENT_ID}")


df_group = df_det.groupby(["feature", "is_correct"]).count().reset_index() 
df_group = df_group.pivot(index="feature", columns="is_correct", values="petname")

# if no false, add false column
if "False" not in df_group.columns:
    df_group["False"] = 0
if "True" not in df_group.columns:
    df_group["True"] = 0

# fill missing values with zero
df_group.fillna(0, inplace=True)

df_group["total"] = df_group["True"] + df_group["False"]
df_group["percent_correct"] = df_group["True"] / df_group["total"]
df_group["percent_correct"] = df_group["percent_correct"]#.round(2)
# df_group = df_group.sort_values(by="percent_correct", ascending=False)
df_group.reset_index(inplace=True)

cols = st.columns(2)
# plot
fig_acc = px.bar(df_group, x="feature", y="percent_correct", color="feature", title="ML Accuracy", hover_data=df_group.columns)
cols[0].plotly_chart(fig_acc, use_container_width=True)

# precision
fig_det = px.scatter(df_det, x="dpx_x", y="dpx_y", color="stage", symbol="feature",  hover_data=df_det.columns, title="ML Error Size")
cols[1].plotly_chart(fig_det, use_container_width=True)

# INTERACTIONS

df = conn.query(f"""SELECT e.name, e.date, i.timestamp, i.petname, i.stage, i.step, i.type, i.subtype, i.dm_x, i.dm_y, i.beam_type FROM interactions i JOIN experiments e ON e.id = i.experiment_id""")


# plot
fig_interactions = px.scatter(df, x="dm_x", y="dm_y", color="type", symbol="subtype", hover_data=df.columns, title="Interactions")
st.plotly_chart(fig_interactions, use_container_width=True)



# SPLIT INTO RUNS / AGGREGATE DATA



st.markdown("---")
st.header("Comparative Data")

tab_ml, tab_duration = st.tabs(["Machine Learning", "Duration"])

with tab_ml:
    st.subheader("Machine Learning")

    df = conn.query(f"SELECT e.name, e.date, d.timestamp, d.petname, d.stage, d.step, d.feature, d.is_correct, d.dpx_x, d.dpx_y FROM detections d JOIN experiments e ON e.id = d.experiment_id")
    

    # drop if Autoliftout in e.name
    df = df[~df["name"].str.contains("AUTOLIFTOUT")]

    UNIQUE_FEATURES_IN_EXPERIMENT = df[df["name"] == EXPERIMENT_NAME]["feature"].unique()

    # filter df
    df = df[df["feature"].isin(UNIQUE_FEATURES_IN_EXPERIMENT)]

    df_group = df.groupby(["name", "date", "feature", "is_correct"]).count().reset_index()

    df_group = df_group.pivot(index=["name", "date", "feature"], columns="is_correct", values="petname")

    # if no false, add false column
    if "False" not in df_group.columns:
        df_group["False"] = 0
    if "True" not in df_group.columns:
        df_group["True"] = 0

    # fill missing values with zero
    df_group.fillna(0, inplace=True)

    df_group["total"] = df_group["True"] + df_group["False"]
    df_group["percent_correct"] = df_group["True"] / df_group["total"]
    df_group["percent_correct"] = df_group["percent_correct"]#.round(2)
    df_group = df_group.sort_values(by="date", ascending=False)

    df_group.reset_index(inplace=True)

    # plot
    # sort by date
    df_group.sort_values(by="date", ascending=True, inplace=True)
    fig_acc = px.bar(df_group, x="name", y="percent_correct", color="feature", barmode="group", title="ML Accuracy", hover_data=df_group.columns)
    st.plotly_chart(fig_acc, use_container_width=True)

    # plot line chart
    fig_acc = px.line(df_group, x="name", y="percent_correct", color="feature", title="ML Accuracy", hover_data=df_group.columns)

    # set y limits at 0 - 1
    fig_acc.update_yaxes(range=[0, 1])
    st.plotly_chart(fig_acc, use_container_width=True)

    # group by name, petname, feature, is_correct
    df_group = df.groupby(["name", "date", "petname", "stage", "feature", "is_correct"]).count().reset_index()
    df_group = df_group.pivot(index=["name", "date", "petname", "stage", "feature"], columns="is_correct", values="timestamp")

    # if no false, add false column
    if "False" not in df_group.columns:
        df_group["False"] = 0
    if "True" not in df_group.columns:
        df_group["True"] = 0

    # fill missing values with zero
    df_group.fillna(0, inplace=True)

    df_group["total"] = df_group["True"] + df_group["False"]
    df_group["percent_correct"] = df_group["True"] / df_group["total"]
    df_group["percent_correct"] = df_group["percent_correct"]#.round(2)
    df_group = df_group.sort_values(by="date", ascending=False)

    df_group.reset_index(inplace=True)

    st.dataframe(df_group, use_container_width=True)

    # sort by date, petname
    df_group.sort_values(by=["date", "petname"], ascending=True, inplace=True)

    for STAGE_NAME in df_group["stage"].unique():
        df_group_stage_filt = df_group[df_group["stage"] == STAGE_NAME]
        fig_ml = px.bar(df_group_stage_filt, x="petname", y="percent_correct", color="feature", 
                        barmode="group", facet_row="name", title=f"ML Accuracy Per Lamella Per Experiment (Stage: {STAGE_NAME})", hover_data=df_group.columns, 
                        height=max(100*len(df_group_stage_filt["name"].unique()), 300))
        
        # set y limits 0 -1
        fig_ml.update_yaxes(range=[0, 1])
        st.plotly_chart(fig_ml, use_container_width=True)

    # select all detections
    df = conn.query(f"SELECT e.name, e.date, d.timestamp, d.petname, d.stage, d.step, d.feature, d.is_correct, d.dpx_x, d.dpx_y FROM detections d JOIN experiments e ON e.id = d.experiment_id")
    df = df[~df["name"].str.contains("AUTOLIFTOUT")]

    # filter df
    df = df[df["feature"].isin(UNIQUE_FEATURES_IN_EXPERIMENT)]

    # calculate average accuracy
    df_group = df.groupby(["name", "date", "is_correct"]).count().reset_index()
    df_group = df_group.pivot(index=["name", "date"], columns="is_correct", values="petname")


    # if no false, add false column
    if "False" not in df_group.columns:
        df_group["False"] = 0
    if "True" not in df_group.columns:
        df_group["True"] = 0

    # fill missing values with zero
    df_group.fillna(0, inplace=True)

    df_group["total"] = df_group["True"] + df_group["False"]
    df_group["percent_correct"] = df_group["True"] / df_group["total"]
    df_group["percent_correct"] = df_group["percent_correct"]#.round(2)
    df_group = df_group.sort_values(by="date", ascending=False)

    df_group.reset_index(inplace=True)

    N_TRUE = df_group["True"].sum()
    N_FALSE = df_group["False"].sum()
    N_TOTAL = N_TRUE + N_FALSE
    ACCURACY = N_TRUE / N_TOTAL

    SELECTED_ACCURACY = df_group[df_group["name"] == EXPERIMENT_NAME]["percent_correct"].values[0]

    # UNIQUE_FEATURES_IN_EXPERIMENT = df[df["name"] == EXPERIMENT_NAME]["feature"].unique()

    cols = st.columns(len(UNIQUE_FEATURES_IN_EXPERIMENT)+1)
    cols[0].metric("ML Accuracy", f"{SELECTED_ACCURACY*100:.2f}%", delta=f"{(SELECTED_ACCURACY - ACCURACY)*100:.2f}%")

    for i, FEATURE_NAME in enumerate(UNIQUE_FEATURES_IN_EXPERIMENT, 1):
        df_det_filt = df[df["feature"] == FEATURE_NAME]

        # calculate average accuracy
        df_group = df_det_filt.groupby(["name", "date", "is_correct"]).count().reset_index()
        df_group = df_group.pivot(index=["name", "date"], columns="is_correct", values="petname")


        # if no false, add false column
        if "False" not in df_group.columns:
            df_group["False"] = 0
        if "True" not in df_group.columns:
            df_group["True"] = 0

        # fill missing values with zero
        df_group.fillna(0, inplace=True)

        df_group["total"] = df_group["True"] + df_group["False"]
        df_group["percent_correct"] = df_group["True"] / df_group["total"]
        df_group["percent_correct"] = df_group["percent_correct"]#.round(2)
        df_group = df_group.sort_values(by="date", ascending=False)

        df_group.reset_index(inplace=True)

        N_TRUE = df_group["True"].sum()
        N_FALSE = df_group["False"].sum()
        N_TOTAL = N_TRUE + N_FALSE
        ACCURACY = N_TRUE / N_TOTAL

        SELECTED_ACCURACY = df_group[df_group["name"] == EXPERIMENT_NAME]["percent_correct"].values[0]

        cols[i].metric(f"{FEATURE_NAME}", f"{SELECTED_ACCURACY*100:.2f}%", delta=f"{(SELECTED_ACCURACY - ACCURACY)*100:.2f}%")


# st.markdown("---")
with tab_duration:
    # history
    st.subheader("Stage Duration")
    df_history = conn.query(f"SELECT e.name, e.date, e.id, h.petname, h.start, h.end, h.duration, h.stage FROM history h JOIN experiments e ON e.id = h.experiment_id")

    UNIQUE_STAGES_IN_EXPERIMENT =  df_history[df_history["name"]==EXPERIMENT_NAME]["stage"].unique()

    SETUP_STAGES = ["SetupTrench", "ReadyTrench", "SetupLamella", "Finished", "Setup"]

    # filter out setup stages from unique stages
    UNIQUE_STAGES_IN_EXPERIMENT = [x for x in UNIQUE_STAGES_IN_EXPERIMENT if x not in SETUP_STAGES]

    # filter out other stages
    df_history = df_history[df_history["stage"].isin(UNIQUE_STAGES_IN_EXPERIMENT)]
    # df_history = df_history[~df_history["stage"].isin(SETUP_STAGES)] # FILTER OUT SETUP

    # calculate average duration # fill missing values with zero
    df_group = df_history.groupby(["name", "date", "stage"]).mean().reset_index()
    df_group.fillna(0, inplace=True)
    df_group.reset_index(inplace=True)

    # drop stages with setup, ready
    df_group = df_group[~df_group["stage"].isin(["SetupTrench", "ReadyTrench", "SetupLamella", "Finished", "Setup"])]
    df_group = df_group.sort_values(by="date", ascending=True)

    df_group["duration_mins"] = df_group["duration"] /60
    df_group["duration_hrs"] = df_group["duration"] /60/60

    # plot
    fig_duration = px.bar(df_group, x="name", y="duration_mins", color="stage", barmode="group", title="Duration", hover_data=df_group.columns)
    st.plotly_chart(fig_duration, use_container_width=True)

    fig_duration = px.line(df_group, x="name", y="duration_mins", color="stage", title="Stage Duration", hover_data=df_group.columns)

    # set y limits at 0 - 1
    # fig_acc.update_yaxes(range=[0, 1])
    st.plotly_chart(fig_duration, use_container_width=True)


    # calculate average duration per lamella
    df_history_filter = df_history[~df_history["stage"].isin(["SetupTrench", "ReadyTrench", "SetupLamella", "Finished", "Setup"])]



    st.subheader("Stage Duration")


    # OVERALL DURATION
    df_group = df_history_filter.groupby(["name", "date"]).mean().reset_index()
    df_group = df_group.sort_values(by="duration", ascending=False)

    df_group["duration_hrs"] = df_group["duration"] /60/60  
    df_group["duration_mins"] = df_group["duration"] /60
    # drop id column
    df_group.drop(columns="id", inplace=True)

    SELECTED_DURATION = df_group[df_group["name"] == EXPERIMENT_NAME]["duration"].values[0]
    AVERAGE_DURATION = df_group['duration'].mean()
    DURATION_DELTA = (SELECTED_DURATION - AVERAGE_DURATION) / AVERAGE_DURATION * 100


    cols = st.columns(len(UNIQUE_STAGES_IN_EXPERIMENT)+1)
    cols[0].metric(f"Average", f"{SELECTED_DURATION/60:.2f} mins", 
        delta=f"{DURATION_DELTA:.2f}%", delta_color="inverse")

    st.dataframe(df_group, use_container_width=True)

    # PER STAGE DURATION
    for i, STAGE_NAME in enumerate(UNIQUE_STAGES_IN_EXPERIMENT, 1):
        df_history_filter2 = df_history_filter[df_history_filter["stage"] == STAGE_NAME]

        df_group = df_history_filter2.groupby(["name", "date"]).mean().reset_index()
        df_group = df_group.sort_values(by="duration", ascending=False)


        # show metric
        try:
            SELECTED_DURATION = df_group[df_group["name"] == EXPERIMENT_NAME]["duration"].values[0]
            AVERAGE_DURATION = df_group['duration'].mean()
            DURATION_DELTA = (SELECTED_DURATION - AVERAGE_DURATION) / AVERAGE_DURATION * 100

            cols[i].metric(f"{STAGE_NAME}", f"{SELECTED_DURATION/60:.2f}min", 
                delta=f"{DURATION_DELTA:.2f}%", delta_color="inverse")
        except Exception as e:
            st.write(e)
            cols[i].empty()