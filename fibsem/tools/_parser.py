
import pandas as pd
import numpy as np

def _parse_health_monitor_data(path: str) -> pd.DataFrame:
    # PATH = "health-monitor/data2.csv"

    df = pd.read_csv(path, skiprows=5)

    # display(df)


    # header = system
    # row 0 = subsystem
    # row 1 = component
    # row 2 = parameter
    # row 3 = unit
    # row 4 = type

    # col 0 = date
    # col 1 = timestamp

    df_headers = df.iloc[0:5, 2:]
    df_data = df.iloc[6:, :]

    systems = df_headers.columns.values 
    subsystems = df_headers.iloc[0, :].values
    components = df_headers.iloc[1, :].values
    parameters = df_headers.iloc[2, :].values
    units = df_headers.iloc[3, :].values
    types = df_headers.iloc[4, :].values

    type_map = {
        "Int": np.uint8,
        "Float": np.float32,
        "String": str,
        "Boolean": bool,
        "DateTime": np.datetime64,
    }

    new_columns = ["Date", "Time"]
    new_columns_type = ["datetime64", "datetime64"]
    for subsystem, component, parameter, unit, type in zip(subsystems, components, parameters, units, types):
        new_columns.append(f"{subsystem}.{component}.{parameter} ({unit})")
        new_columns_type.append(type_map[type])

    df_data.columns = new_columns

    # replace all values of "NaN" with np.nan
    df_data = df_data.replace("NaN", np.nan)

    # drop columns with all NaN values
    # df_data = df_data.dropna(axis=0, how="all")
    # df_data = df_data.astype(dict(zip(new_columns, new_columns_type)))

    # combine date and time columns
    df_data["datetime"] = pd.to_datetime(df_data["Date"] + " " + df_data["Time"])

    # set timezone to Aus/Sydney for datetime column
    # df_data["datetime"] = df_data["datetime"].dt.tz_localize("UTC").dt.tz_convert("Australia/Sydney")

    # drop Date and Time columns
    df_data = df_data.drop(columns=["Date", "Time"])
    

    # print duplicate columns
    # drop duplicate columns
    df_data = df_data.loc[:,~df_data.columns.duplicated()]
    print(df_data.columns[df_data.columns.duplicated()])

    
    return df_data
