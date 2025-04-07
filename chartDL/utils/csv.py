import json
import pandas as pd
import os
from pathlib import Path


def import_ohlcv_from_csv(
    file_path: str,  # path of the csv file
    header: bool = True,  # indicate if the file has a header or not
    datetime_format: str = "%Y-%m-%d %H:%M:%S",  # format for datetime parsing
    column_names: list = None,  # custom column names
    delimiter: str = ",",  # delimiter for the CSV file
):
    """
    Import OHLCV (Open, High, Low, Close, Volume) data from a CSV file.
    Data must be structured as [Date, Time, Open, High, Low, Close, Volume] or
    [DateTime, Open, High, Low, Close, Volume].

    Args:
        file_path (str): Path to the file containing OHLCV data.
        header (bool): True if the file has a header row; False otherwise.
        delimiter (str): Delimiter used in the file (default is comma).
        datetime_format (str): The format to parse the datetime from the file.
        column_names (list): List of column names if the file does not have a header.
                             Must include "Date", "Time", "Open", "High", "Low", "Close", "Volume".

    Returns:
        pd.DataFrame: DataFrame containing the parsed OHLCV data with a TimeStamp column.
        with format [TimeStamp, Open, High, Low, Close, Volume].
    """

    # Read the CSV file
    if header:
        df = pd.read_csv(file_path, delimiter=delimiter)
    else:
        if column_names is None:
            raise ValueError("Column names must be provided if the header is False.")
        df = pd.read_csv(
            file_path, delimiter=delimiter, header=None, names=column_names
        )

    # Check for DateTime or Date and Time columns and modify the datetime column
    if "DateTime" in df.columns:
        df["TimeStamp"] = pd.to_datetime(df["DateTime"], format=datetime_format)
        df = df.drop(columns=["DateTime"])
    elif "Date" in df.columns and "Time" in df.columns:
        # Ensure both Date and Time are present before combining
        df["TimeStamp"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            format=datetime_format,
        )
        df = df.drop(columns=["Date", "Time"])
    else:
        raise ValueError(
            "Data must contain either 'DateTime' or both 'Date' and 'Time' columns."
        )

    # Bring the TimeStamp column to the first position
    new_order = ["TimeStamp"] + [col for col in df.columns if col != "TimeStamp"]
    return df[new_order]


def export_ohlcv_to_csv(
    file_path: str,  # Path of the CSV file to save data
    df: pd.DataFrame,  # Market data in pandas DataFrame
    header: bool = True,  # Indicate whether to include a header in the CSV
    date_format: str = "%Y-%m-%d",  # Format for the Date column
    time_format: str = "%H:%M:%S",  # Format for the Time column
    single_datetime: bool = False,  # indicate that just one column for data and time
):
    """
    Export OHLCV data from a pandas DataFrame to a CSV file in the conventional format.
    The DataFrame must have the format of [TimeStamp, Open, High, Low, Close, Volume].

    Args:
        file_path (str): Path to save the CSV file.
        df (pd.DataFrame): DataFrame containing OHLCV data with a 'TimeStamp' column.
        header (bool): True to include header in the CSV; False otherwise.
        date_format (str): Format to use for the 'Date' column. Default is '%Y-%m-%d'.
        time_format (str): Format to use for the 'Time' column. Default is '%H:%M:%S'.
        single_datetime (bool): flag for representing single column for datatime.
    """

    # Check if the 'TimeStamp' column exists in the DataFrame
    if "TimeStamp" not in df.columns:
        raise ValueError("DataFrame must contain a 'TimeStamp' column.")

    # make copy of dataframe to prevent from changing orginal one
    df = df.copy()

    # Create 'Date' and 'Time' columns from 'TimeStamp'
    if single_datetime:
        df["DateTime"] = df["TimeStamp"].dt.strftime(f"{date_format} {time_format}")
        dt_cols = ["DateTime"]  # just first columns
    else:
        df["Date"] = df["TimeStamp"].dt.strftime(date_format)  # Format the Date
        df["Time"] = df["TimeStamp"].dt.strftime(time_format)  # Format the Time
        dt_cols = ["Date", "Time"]  # First two columns

    # Drop the 'TimeStamp' column as it will be replaced by Date and Time
    df = df.drop(columns=["TimeStamp"])

    # Reorder columns: Date, Time, and then remaining OHLCV columns
    remaining_cols = [
        col for col in df.columns if col not in dt_cols
    ]  # Remaining columns
    new_order = dt_cols + remaining_cols  # New column order
    df = df[new_order]  # Reorder the DataFrame

    # Save the DataFrame to a CSV file
    df.to_csv(file_path, header=header, index=False)  # Write to CSV


def get_all_csv_files(directory_path: str):
    """
    Gather all CSV file paths in a specified directory and its subdirectories.

    Args:
        directory_path (str): Path to the directory to search for CSV files.

    Returns:
        list: A list of file paths for all CSV files found in the directory.
    """

    csv_files = []  # Initialize an empty list to store CSV file paths
    # Walk through the directory and its subdirectories
    for root, _, files in os.walk(directory_path):
        for file in files:
            # Check if the file has a .csv extension
            if file.lower().endswith(".csv"):  # Use .lower() to handle case sensitivity
                csv_files.append(
                    os.path.join(root, file)
                )  # Append the full path to the list

    return csv_files  # Return the list of CSV file paths


def merge_monthly_and_yearly_csv(
    monthly_path: str,  # Path to directory of monthly CSV data
    yearly_path: str,  # Path to directory of yearly CSV data
    **kwargs,  # Import function specifications for import_ohlcv_from_csv
):
    """
    Merge monthly and yearly separated data into a single DataFrame.

    Args:
        monthly_path (str): Path to the directory containing monthly CSV files.
        yearly_path (str): Path to the directory containing yearly CSV files.
        **kwargs: Additional keyword arguments to pass to the import_ohlcv_from_csv function.

    Returns:
        pd.DataFrame: A single DataFrame containing merged monthly and yearly data, sorted by TimeStamp.
    """

    # Gather all CSV files from both monthly and yearly directories
    csv_files = get_all_csv_files(monthly_path) + get_all_csv_files(yearly_path)

    # Initialize a list to store DataFrames
    dfs = []

    # Import all CSV files and append to the list
    for file_path in csv_files:
        # Import data using the specified import function and additional keyword arguments
        df = import_ohlcv_from_csv(file_path, **kwargs)
        dfs.append(df)

    # Concatenate all DataFrames vertically
    df_merged = pd.concat(dfs, axis=0, ignore_index=True)

    # Sort the merged DataFrame by the TimeStamp column
    df_merged = df_merged.sort_values(by="TimeStamp", ignore_index=True)

    return df_merged  # Return the merged DataFrame


def export_multi_view_data(path: str, results: list[dict]):
    """
    Saves multi-timeframe view data for both original and live synchronized data.

    Args:
        path (str): Directory where data will be saved.
        results (list[dict]): List of dictionaries for each timeframe containing:
            - 'timeframe' (str): Timeframe of the data (e.g., '1h', '4h').
            - 'origin' (pd.DataFrame): Original DataFrame data for the timeframe.
            - 'live' (pd.DataFrame): Live-synchronized DataFrame data for the timeframe.
    """

    # Create directories for 'origin' and 'live' data if they do not exist
    save_path = Path(path)
    origin_save_path = save_path / "origin"
    live_save_path = save_path / "live"
    origin_save_path.mkdir(parents=True, exist_ok=True)
    live_save_path.mkdir(parents=True, exist_ok=True)

    # Define date and time formats for export
    date_format = "%Y-%m-%d"
    time_format = "%H:%M:%S"

    # Prepare information for saving paths
    informations = {
        "timeframes": [res["timeframe"] for res in results],
        "data_paths": [],  # Store paths for each timeframe's data files
    }

    # Save each timeframe's data as CSV files in the appropriate directories
    for res in results:
        # Save original data to the 'origin' directory
        origin_file_path = origin_save_path / f"origin-{res['timeframe']}.csv"
        export_ohlcv_to_csv(
            origin_file_path,
            res["origin"],
            header=True,
            date_format=date_format,
            time_format=time_format,
        )

        # Save live data to the 'live' directory
        live_file_path = live_save_path / f"live-{res['timeframe']}.csv"
        export_ohlcv_to_csv(
            live_file_path,
            res["live"],
            header=True,
            date_format=date_format,
            time_format=time_format,
        )

        # Update the information dictionary with paths for JSON export
        informations["data_paths"].append(
            {
                "timeframe": res["timeframe"],
                "origin_path": str(origin_file_path),
                "live_path": str(live_file_path),
            }
        )

    # Save information as a JSON file in the main data directory
    info_file_path = save_path / "data_information.json"
    with open(info_file_path, "w") as json_file:
        json.dump(informations, json_file, indent=4)

    print(
        f"Data successfully exported to {save_path}. JSON summary saved at {info_file_path}"
    )


def import_multi_view_data(path: str):
    """
    Imports multi-timeframe view data from saved CSV files and JSON summary file.

    Args:
        path (str): Directory where the multi-timeframe data is saved.

    Returns:
        list[dict]: List of dictionaries for each timeframe containing:
            - 'timeframe' (str): Timeframe of the data (e.g., '1h', '4h').
            - 'origin' (pd.DataFrame): Original DataFrame data for the timeframe.
            - 'live' (pd.DataFrame): Live-synchronized DataFrame data for the timeframe.
    """

    # Define paths for save directories and JSON information file
    save_path = Path(path)
    origin_save_path = save_path / "origin"
    live_save_path = save_path / "live"
    info_file_path = save_path / "data_information.json"

    # Check existence of directories and the JSON information file
    if not save_path.exists():
        raise FileNotFoundError(f"The specified path '{path}' does not exist.")
    if not info_file_path.exists():
        raise FileNotFoundError(
            f"JSON information file 'data_information.json' not found in {path}."
        )
    if not origin_save_path.exists() or not live_save_path.exists():
        raise FileNotFoundError(
            "Expected 'origin' and 'live' folders are missing in the specified path."
        )

    # Load JSON information
    with open(info_file_path, "r") as json_file:
        information = json.load(json_file)

    # List to hold imported data
    imported_data = []

    # Import data for each timeframe using paths from JSON information
    for timeframe_info in information["data_paths"]:
        timeframe = timeframe_info["timeframe"]

        # Load origin data
        origin_file_path = Path(timeframe_info["origin_path"])
        if not origin_file_path.exists():
            raise FileNotFoundError(
                f"Origin data file '{origin_file_path}' for timeframe '{timeframe}' not found."
            )
        origin_data = import_ohlcv_from_csv(
            origin_file_path, header=True, datetime_format="%Y-%m-%d %H:%M:%S"
        )

        # Load live data
        live_file_path = Path(timeframe_info["live_path"])
        if not live_file_path.exists():
            raise FileNotFoundError(
                f"Live data file '{live_file_path}' for timeframe '{timeframe}' not found."
            )
        live_data = import_ohlcv_from_csv(
            live_file_path, header=True, datetime_format="%Y-%m-%d %H:%M:%S"
        )

        # Append to imported data list
        imported_data.append(
            {"timeframe": timeframe, "origin": origin_data, "live": live_data}
        )

    return imported_data
