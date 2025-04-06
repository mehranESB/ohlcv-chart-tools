import pandas as pd


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
