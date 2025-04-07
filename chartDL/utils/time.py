from datetime import datetime, timedelta


# Dictionary for converting timeframes to timedelta objects for future use
time_delta = {
    "1m": timedelta(minutes=1),  # 1-minute interval
    "5m": timedelta(minutes=5),  # 5-minute interval
    "15m": timedelta(minutes=15),  # 15-minute interval
    "30m": timedelta(minutes=30),  # 30-minute interval
    "1h": timedelta(hours=1),  # 1-hour interval
    "4h": timedelta(hours=4),  # 4-hour interval
    "1d": timedelta(days=1),  # 1-day interval
}


def check_valid_timeframe(timeframe: str):
    """
    Checks if the provided timeframe is valid by looking it up in the `time_delta` dictionary.

    Args:
        timeframe (str): The timeframe string (e.g., '1m', '5m', '1h', etc.) to check.

    Returns:
        bool: Returns True if the timeframe is valid, False otherwise.
    """
    # Check if the timeframe exists in the predefined time_delta dictionary
    return timeframe in time_delta


def is_new_bar(time, prev_time, time_frame: str):
    """
    Check if the incoming data represents a new bar based on the timeframe.

    Args:
        time (datetime): Current data timestamp.
        prev_time (datetime): Previous opening time for the last bar.
        time_frame (str): The time interval of the data (e.g., "1m", "5m", "1h").

    Returns:
        bool: True if it's a new bar; False otherwise.
    """

    return (time - prev_time) >= get_time_delta(time_frame)


def get_time_delta(timeframe: str):
    """
    Returns the corresponding timedelta for the given timeframe string.

    This function looks up the provided timeframe in the predefined dictionary `time_delta`
    and returns the corresponding `timedelta` object. If the timeframe is not found, it raises a ValueError.

    Args:
        timeframe (str): A string representing the timeframe (e.g., '1m', '5m', '1h', etc.).

    Returns:
        timedelta: The corresponding timedelta object for the given timeframe.

    Raises:
        ValueError: If the provided timeframe is not supported (i.e., not in the `time_delta` dictionary).
    """

    # Check if the provided timeframe is in the predefined time_delta dictionary
    if timeframe in time_delta:
        return time_delta[timeframe]

    # Raise an error if the timeframe is unsupported
    raise ValueError(
        f"Unsupported timeframe: '{timeframe}'. Valid timeframes are {', '.join(time_delta.keys())}."
    )


def round_open_time(time, time_frame: str):
    """
    Rounds a datetime to the nearest open time based on the specified timeframe.

    Args:
        time (datetime): Current data timestamp.
        time_frame (str): Timeframe to round to (e.g., "1m", "5m", "1h").

    Returns:
        datetime: Rounded datetime that represents the opening time for the timeframe.
    """
    year, month, day, hour, minute = (
        time.year,
        time.month,
        time.day,
        time.hour,
        time.minute,
    )

    if time_frame == "1m":
        return datetime(year, month, day, hour, minute)
    if time_frame == "5m":
        return datetime(year, month, day, hour, minute - (minute % 5))
    if time_frame == "15m":
        return datetime(year, month, day, hour, minute - (minute % 15))
    if time_frame == "30m":
        return datetime(year, month, day, hour, minute - (minute % 30))
    if time_frame == "1h":
        return datetime(year, month, day, hour)
    if time_frame == "4h":
        return datetime(year, month, day, hour - (hour % 4))
    if time_frame == "1d":
        return datetime(year, month, day)
    # Fallback for unsupported timeframes
    raise ValueError(f"Unsupported time_frame: {time_frame}")


def get_X_delta(df):
    """
    This function calculates the most common difference (mode of differences)
    between consecutive values in the 'X' column of a DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame that must contain a column named 'X'.

    Returns:
        delta_x: The most common difference (mode) between consecutive rows in the 'X' column.

    Raises:
        IndexError: If the DataFrame does not contain a column named 'X'.
    """

    # Check if DataFrame has 'X' column
    if "X" not in df.columns:
        raise IndexError("The DataFrame must contain a column named 'X'.")

    # Calculate differences in the 'X' column
    x_diff = df["X"].diff()

    # Check if the differences are empty or all NaN
    if x_diff.isna().all():
        raise ValueError(
            "The 'X' column does not have enough data to compute differences."
        )

    # Calculate the mode of the differences
    delta_x = x_diff.mode()

    # Check if mode is empty (edge case: all rows in 'X' column are identical or NaN)
    if delta_x.empty:
        raise ValueError(
            "Could not determine a mode for the differences in the 'X' column."
        )

    # Return the first mode (in case of multiple modes)
    return delta_x[0]


def get_timeframe(df):
    """
    This function calculates the most common time delta between consecutive rows in the 'TimeStamp' column of the dataframe.
    It uses the difference between timestamps to infer the time frame (e.g., '1m', '5m', '1h', etc.).

    Args:
        df (pd.DataFrame): The dataframe containing market data with a 'TimeStamp' column.

    Returns:
        str: The inferred time frame (e.g., '1m', '5m', '1h', etc.)

    Raises:
        ValueError: If an unsupported time frame is detected or 'TimeStamp' column is missing/invalid.
    """

    # Check if there are any invalid or NaT timestamps after conversion
    if df["TimeStamp"].isnull().any():
        raise ValueError(
            "There are invalid or missing timestamps in the 'TimeStamp' column."
        )

    # Calculate the time differences between consecutive timestamps
    time_diff = df["TimeStamp"].diff()

    # Get the most common time difference (mode)
    most_common_diff = time_diff.mode()[0]

    # Initialize time_frame to None
    time_frame = None

    # Compare the most common time difference with the predefined time deltas
    for key, value in time_delta.items():
        if value == most_common_diff:
            time_frame = key
            break

    # If no match is found in the predefined time deltas, raise an error
    if time_frame is None:
        raise ValueError(
            f"Unsupported time frame detected: {most_common_diff}. Please check the data for inconsistencies."
        )

    # Return the inferred time frame (e.g., '1m', '5m', '1h', etc.)
    return time_frame
