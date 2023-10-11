import streamlit as st

from fibsem.tools import _parser

import plotly.express as px


st.set_page_config(layout="wide", page_title="OpenFIBSEM Telemetry")
st.title("OpenFIBSEM Telemetry")


DEFAULT_PATH = "data2.csv"

PATH = st.text_input("Path to telemetry data", DEFAULT_PATH)

@st.cache_data
def _load_data(PATH):
    return _parser._parse_health_monitor_data(PATH) 

df_data = _load_data(PATH)



tabs = st.tabs(["Raw Data", "Filter Data", "Beam On/Off"])

with tabs[0]:
    # st.write(df_data)
    st.write("### RAW DATA ###")

with tabs[1]:

    # FILTER BY SUBSYSTEM
    st.write("### FILTER BY SUBSYSTEM ###")

    subsystems = df_data.columns.str.split(".").str[0].unique().tolist()
    subsystem = st.selectbox("Select Subsystem", subsystems)
    # also keep datetime column
    subsystem_cols =  ["datetime"] + [col for col in df_data.columns if subsystem in col]


    # TODO: filter once only for all


    df_subsystem = df_data[subsystem_cols]
    # drop rows with all NaN values excluding datetime column
    # drop columns with all NaN values excluding datetime column
    df_subsystem = df_subsystem.dropna(axis=0, how="all", subset=df_subsystem.columns[1:])
    df_subsystem = df_subsystem.dropna(axis=1, how="all")

    # filter by component
    components = df_subsystem.columns.str.split(".").str[1].unique().tolist()

    component_list = st.multiselect("Select Components", ["ALL"] + components, default=["ALL"])
    if "ALL" not in component_list:
        component_cols = ["datetime"]
        for comp in component_list:
            component_cols += [col for col in df_subsystem.columns if comp in col ]
        df_subsystem = df_subsystem[component_cols]


    # filter by parameter
    parameters = df_subsystem.columns.str.split(".").str[2].unique().tolist()
    parameter_list = st.multiselect("Select Parameters", ["ALL"] + parameters, default=["ALL"])
    if "ALL" not in parameter_list:
        parameter_cols = ["datetime"]
        for param in parameter_list:
            parameter_cols += [col for col in df_subsystem.columns if param in col ]
        df_subsystem = df_subsystem[parameter_cols]


    df_subsystem = df_subsystem.dropna(axis=0, how="all", subset=df_subsystem.columns[1:])
    df_subsystem = df_subsystem.dropna(axis=1, how="all")

    if len(df_subsystem) > 0:

        st.write(subsystem, component_list, parameter_list)
        st.write(f"{len(df_subsystem)} rows, {len(df_subsystem.columns)-1} columns ({subsystem})")
        st.write(df_subsystem)

        fig = px.line(df_subsystem, x="datetime", y=df_subsystem.columns[1:], title=f"Health Monitor Data - {subsystem}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No data to display")


with tabs[2]:

    #### Beam Is On, Is Blanked ####

    fig_cols = st.columns(2)


    # select columns with beam on and beam blanked
    for filter_str in ["Beam Is On", "Is Blanked"]:

        cols = ["datetime"] + [col for col in df_data.columns if filter_str in col] # or "Beam Is On" in col 
        st.write(cols)
        df_beam = df_data[cols]

        df_beam = df_beam.dropna(axis=0, how="all", subset=df_beam.columns[1:])
        df_beam = df_beam.dropna(axis=1, how="all")

        # fillna with the previous row values
        df_beam = df_beam.fillna(method="ffill")
        # calculate duration
        df_beam["duration"] = df_beam["datetime"].diff().dt.total_seconds()
        df_beam = df_beam.dropna(axis=0, how="any", subset=df_beam.columns[1:])
        # st.dataframe(df_beam)


        # group by beam on and beam blanked
        fib_columns = [col for col in df_beam.columns if "FIB" in col]
        sem_columns = [col for col in df_beam.columns if "SEM" in col]
        # st.write(fib_columns, sem_columns)

        df_groupby_fib = df_beam.groupby(fib_columns).agg({"duration": "sum"}).reset_index()
        df_groupby_sem = df_beam.groupby(sem_columns).agg({"duration": "sum"}).reset_index()

        fig_cols[0].write(df_groupby_sem)
        fig_cols[1].write(df_groupby_fib)

        # convert duration to hours
        df_groupby_fib["duration"] = df_groupby_fib["duration"] / 3600
        df_groupby_sem["duration"] = df_groupby_sem["duration"] / 3600


        # plot as piecharts

        fig_fib = px.pie(df_groupby_fib, values="duration", title=f"FIB Beam {filter_str}", names=fib_columns[0], hover_data=df_groupby_fib.columns)
        fig_sem = px.pie(df_groupby_sem, values="duration", title=f"SEM Beam {filter_str}", names=sem_columns[0], hover_data=df_groupby_sem.columns)

        # show names on figure
        fig_fib.update_traces(textposition='inside', textinfo='percent+label')
        fig_sem.update_traces(textposition='inside', textinfo='percent+label')
        

        fig_cols[0].plotly_chart(fig_sem, use_container_width=True)
        fig_cols[1].plotly_chart(fig_fib, use_container_width=True)



