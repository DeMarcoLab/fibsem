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
st.write("PROEJCTS")
st.data_editor(df)

PROJECT_NAMES = df["name"].values.tolist()
PROJECT_NAME = st.sidebar.selectbox("Select project", PROJECT_NAMES)

PROJECT_ID = df[df["name"] == PROJECT_NAME]["id"].values[0]

# select experiment from project
df = conn.query(f"SELECT * FROM experiments WHERE project_id = {PROJECT_ID}")

EXPERIMENT_NAMES = df["name"].values.tolist()
EXPERIMENT_NAME = st.sidebar. selectbox("Select experiment", EXPERIMENT_NAMES)
EXPERIMENT_ID = df[df["name"] == EXPERIMENT_NAME]["id"].values[0]

# show data from experiment
df = conn.query(f"SELECT * FROM experiments WHERE project_id = {PROJECT_ID} AND name = '{EXPERIMENT_NAME}'")
st.dataframe(df)

# show data from experiment


# HISTORY
df_history = conn.query(f"SELECT e.name, e.date, h.petname, h.start, h.end, h.duration, h.stage FROM history h JOIN experiments e ON e.id = h.experiment_id WHERE h.experiment_id = {EXPERIMENT_ID} ")

st.dataframe(df)
# Duration
cols = st.columns(2)
fig_duration = px.bar(df_history, x="petname", y="duration", color="stage", barmode="group", hover_data=df_history.columns, title="Lamella Duration by Stage")
cols[0].plotly_chart(fig_duration, use_container_width=True)

# Duration
fig_duration = px.bar(df_history, x="stage", y="duration", color="petname", barmode="group", hover_data=df_history.columns, title="Stage Duration by Lamella")
cols[1].plotly_chart(fig_duration, use_container_width=True)

# STEPS
df_steps = conn.query(f"SELECT * FROM steps WHERE experiment_id = {EXPERIMENT_ID}")

st.dataframe(df_steps)

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

st.dataframe(df_det)


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
df_group["percent_correct"] = df_group["percent_correct"].round(2)
# df_group = df_group.sort_values(by="percent_correct", ascending=False)
df_group.reset_index(inplace=True)

cols = st.columns(2)
# plot
fig_acc = px.bar(df_group, x="feature", y="percent_correct", color="feature", title="ML Accuracy", hover_data=df_group.columns)
cols[0].plotly_chart(fig_acc, use_container_width=True)

# precision
fig_det = px.scatter(df_det, x="dpx_x", y="dpx_y", color="stage", symbol="feature",  hover_data=df_det.columns, title="ML Error Size")
cols[1].plotly_chart(fig_det, use_container_width=True)


st.subheader("AGGREGATE")


df = conn.query(f"SELECT e.name, e.date, d.timestamp, d.petname, d.stage, d.step, d.feature, d.is_correct, d.dpx_x, d.dpx_y FROM detections d JOIN experiments e ON e.id = d.experiment_id")

st.dataframe(df)



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
df_group["percent_correct"] = df_group["percent_correct"].round(2)
df_group = df_group.sort_values(by="date", ascending=False)

df_group.reset_index(inplace=True)

st.dataframe(df_group)

# plot
fig_acc = px.bar(df_group, x="date", y="percent_correct", color="feature", barmode="group", title="ML Accuracy", hover_data=df_group.columns)
st.plotly_chart(fig_acc, use_container_width=True)




# INTERACTIONS

df = conn.query(f"""SELECT e.name, e.date, i.timestamp, i.petname, i.stage, i.step, i.type, i.subtype, i.dm_x, i.dm_y, i.beam_type FROM interactions i JOIN experiments e ON e.id = i.experiment_id""")


st.dataframe(df)


# plot
fig_interactions = px.scatter(df, x="dm_x", y="dm_y", color="type", symbol="subtype", hover_data=df.columns, title="Interactions")
st.plotly_chart(fig_interactions, use_container_width=True)



# SPLIT INTO RUNS / AGGREGATE DATA



st.markdown("---")
st.header("COMPARISON TO AGGREGATES")
# select all detections
df = conn.query(f"SELECT e.name, e.date, d.timestamp, d.petname, d.stage, d.step, d.feature, d.is_correct, d.dpx_x, d.dpx_y FROM detections d JOIN experiments e ON e.id = d.experiment_id")

st.dataframe(df)

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
df_group["percent_correct"] = df_group["percent_correct"].round(2)
df_group = df_group.sort_values(by="date", ascending=False)

df_group.reset_index(inplace=True)

N_TRUE = df_group["True"].sum()
N_FALSE = df_group["False"].sum()
N_TOTAL = N_TRUE + N_FALSE
ACCURACY = N_TRUE / N_TOTAL

SELECTED_ACCURACY = df_group[df_group["name"] == EXPERIMENT_NAME]["percent_correct"].values[0]

st.markdown(f"**Selected Accuracy:** {SELECTED_ACCURACY:.2f}")
st.markdown(f"**Overall Accuracy:** {ACCURACY:.2f}")
st.metric("ML Accuracy", f"{SELECTED_ACCURACY:.2f}", delta=f"{SELECTED_ACCURACY - ACCURACY:.2f}")



# history
df_history = conn.query(f"SELECT e.name, e.date, e.id, h.petname, h.start, h.end, h.duration, h.stage FROM history h JOIN experiments e ON e.id = h.experiment_id")


st.dataframe(df_history)

# calculate average duration 
df_group = df_history.groupby(["name", "date", "stage"]).mean().reset_index()
# df_group = df_group.pivot(index=["name", "date"], columns="stage", values="duration")

# fill missing values with zero
df_group.fillna(0, inplace=True)



df_group.reset_index(inplace=True)

st.dataframe(df_group)

# SELECTED_DURATION = df_group[df_group["name"] == EXPERIMENT_NAME]["duration"].values[0]

# st.markdown(f"**Selected Duration:** {SELECTED_DURATION:.2f}")

# st.metric("Duration", f"{SELECTED_DURATION:.2f}", delta=f"{SELECTED_DURATION - df_group['duration'].mean():.2f}")




# drop stages with setup, ready
df_group = df_group[~df_group["stage"].isin(["SetupTrench", "ReadyTrench", "SetupLamella"])]
st.dataframe(df_group)


df_group = df_group.sort_values(by="date", ascending=True)
# plot
fig_duration = px.bar(df_group, x="name", y="duration", color="stage", barmode="group", title="Duration", hover_data=df_group.columns)
st.plotly_chart(fig_duration, use_container_width=True)

# calculate average duration per lamella
df_history_filter = df_history[~df_history["stage"].isin(["SetupTrench", "ReadyTrench", "SetupLamella"])]


STAGE_NAME = st.selectbox("Stage", df_history_filter["stage"].unique())
for STAGE_NAME in df_history_filter["stage"].unique():
    df_history_filter2 = df_history_filter[df_history_filter["stage"] == STAGE_NAME]
    # st.dataframe(df_history_filter)

    df_group = df_history_filter2.groupby(["name", "date"]).mean().reset_index()
    df_group = df_group.sort_values(by="duration", ascending=False)

    # st.dataframe(df_group)
    # st.write(STAGE_NAME)
    # show metric
    SELECTED_DURATION = df_group[df_group["name"] == EXPERIMENT_NAME]["duration"].values[0]
    AVERAGE_DURATION = df_group['duration'].mean()
    DURATION_DELTA = (SELECTED_DURATION - AVERAGE_DURATION)
    # calc delta as %
    DURATION_DELTA = DURATION_DELTA / AVERAGE_DURATION * 100

    st.subheader(f"Duration {STAGE_NAME}")
    st.markdown(f"**Selected Duration:** {SELECTED_DURATION:.2f}")
    st.metric("Duration", f"{SELECTED_DURATION/60:.2f}min", 
        delta=f"{DURATION_DELTA:.2f}%", delta_color="inverse")