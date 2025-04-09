from .utils import csv as csv_utils
from .utils import time as time_utils
import numpy as np
from datetime import datetime
import pandas as pd
from typing import Union

import pandas as pd


import pandas as pd


def create_sample_df(dt, data, columns, timeframe):
    """
    Creates a pandas DataFrame from the given data, adds a TimeStamp column, and stores the timeframe as metadata.

    Parameters:
    - dt: A list or pandas Series of datetime values (timestamps).
    - data: A 2D array or list containing the OHLCV data (e.g., Open, High, Low, Close, Volume).
    - columns: A list of column names (e.g., ["Open", "High", "Low", "Close", "Volume"]).
    - timeframe: A string representing the timeframe (e.g., "1h", "15min").

    Returns:
    - A pandas DataFrame with TimeStamp as a column and the given data, and with timeframe stored in the metadata.
    """
    # Create DataFrame from data with the given column names
    df = pd.DataFrame(data.copy(), columns=columns)

    # Add TimeStamp as a column
    df["TimeStamp"] = dt.copy()

    # Optionally, move TimeStamp to the first position (just for better readability)
    df = df[["TimeStamp"] + [col for col in df.columns if col != "TimeStamp"]]

    # Attach timeframe as metadata
    df.attrs["timeframe"] = timeframe

    return df


class MultiDataset:

    def __init__(
        self, path: str, seq_len: int, from_to: list[str] = None, nIgnore_last: int = 25
    ):
        """
        Initializes the MultiDataset object, loading multi-timeframe data and preparing
        it for sequence-based model training within a specified date range.

        Args:
            path (str): Path to the directory containing 'origin' and 'live' data folders.
            seq_len (int): Sequence length for each data sample returned by the dataset.
            from_to (list[str], optional): List of two strings specifying the start and end datetime
                                           for data filtering, in "%Y.%m.%d %H:%M" format.
            nIgnore_last (int): Number of bars to ignore at the end of the dataset.

        Attributes:
            market_data (list[dict]): List of dictionaries containing data for each timeframe, sorted by timeframe.
            dt (np.ndarray): Timestamps of the data, synchronized across all timeframes.
            seq_len (int): Sequence length for the model's training input.
            path (str): Path to the data directory.
            index_range (tuple[int, int]): Valid start and end index for data retrieval based on the date range.
        """

        # Import and sort market data by timeframe in ascending order
        market_data = csv_utils.import_multi_view_data(path)
        market_data = sorted(
            market_data, key=lambda data: time_utils.time_delta[data["timeframe"]]
        )

        # Extract timestamps from 'live' data as time array of dataset
        self.dt = np.array(market_data[0]["live"]["TimeStamp"])

        # Set dataset properties
        self.market_data = self.convert_to_numpy(
            market_data
        )  # converted to numpy for efficiency
        self.seq_len = seq_len
        self.path = path
        self.nIgnore_last = nIgnore_last

        # Set the valid index range based on the from_to time range
        self.set_valid_range(from_to)

    def convert_to_numpy(self, market_data: list):
        """
        Converts the market data DataFrames into NumPy arrays to improve performance
        in data-intensive operations.

        Args:
            market_data (list): A list of dictionaries where each dictionary contains
                                market data in pandas DataFrame format.

        Returns:
            list: A list of dictionaries with data in NumPy array format for improved performance.
        """
        numpy_market_data = []
        for data in market_data:
            # Extract and convert the 'origin' and 'live' DataFrames to NumPy
            origin_dt = data["origin"]["TimeStamp"].to_numpy()  # Convert to NumPy array
            live_cursor = data["live"]["Cursor"].to_numpy()  # Convert to NumPy array

            # Drop columns that won't be used in the NumPy arrays
            origin_data = data["origin"].drop(columns=["TimeStamp"], inplace=False)
            live_data = data["live"].drop(
                columns=["Cursor", "TimeStamp"], inplace=False
            )

            # Append converted data to the new list
            numpy_market_data.append(
                {
                    "timeframe": data["timeframe"],
                    "origin_dt": origin_dt,  # Store timestamp as a NumPy array
                    "origin": origin_data.to_numpy(),  # Convert 'origin' DataFrame to NumPy array
                    "live_cursor": live_cursor,  # Store cursor as a NumPy array
                    "live": live_data.to_numpy(),  # Convert 'live' DataFrame to NumPy array
                    "columns": origin_data.columns.tolist(),  # Preserve column names for reference
                }
            )

        return numpy_market_data

    def __len__(self):
        """
        Returns:
            int: The valid length of the dataset based on the accepted range
                 from self.index_range, accounting for the start and end indices.
        """
        # Calculate the length as the difference between the end and start index,
        # adding 1 because the range is inclusive.
        return self.index_range[1] - self.index_range[0] + 1

    def __getitem__(self, index: int):
        """
        Retrieves a sequence of market data up to the specified index in the lowest timeframe.

        Args:
            index (int): Index of the data to retrieve, should be between 0 and len(self) - 1.

        Returns:
            list[Data]: A list of Data instances containing the sequence data for each timeframe,
                        up to the specified index in the lowest timeframe.

        Raises:
            ValueError: If the index is out of the valid range (0 to len(self) - 1).
        """

        # Validate and adjust index within the internal range
        if index < 0 or index >= len(self):
            raise ValueError(f"Index must be in range 0 to {len(self) - 1}.")
        index += self.index_range[0]  # Align with dataset's internal range

        # Retrieve market data for each timeframe as Data instances
        data_instances = []
        for i, timeframe_data in enumerate(self.market_data):
            if i == 0:  # Lowest timeframe
                # Retrieve sequence of data in the lowest timeframe
                dt = timeframe_data["origin_dt"][index - self.seq_len : index]
                data = timeframe_data["origin"][index - self.seq_len : index]

            else:  # Higher timeframes
                # Get cursor to align higher timeframe data with the current index
                cursor = timeframe_data["live_cursor"][index - 1]

                # Retrieve sequence up to the cursor and replace last row with live data
                dt = timeframe_data["origin_dt"][cursor - self.seq_len + 1 : cursor + 1]
                data = timeframe_data["origin"][
                    cursor - self.seq_len + 1 : cursor + 1
                ].copy()

                # Update the last row with current live data to prevent future leakage
                data[-1] = timeframe_data["live"][index - 1]

            # Append the Data instance for this timeframe
            data_instances.append(
                create_sample_df(
                    dt, data, timeframe_data["columns"], timeframe_data["timeframe"]
                )
            )

        return data_instances

    def set_valid_range(self, from_to: list[str] = None):
        """
        Sets the index range (start, end) for data retrieval based on the specified datetime range.

        Args:
            from_to (list[str], optional): List containing start and end datetime as strings
                                           in "%Y.%m.%d %H:%M" format. If None, uses the full data range.
        """
        # 1. Convert 'from_to' times to datetime objects, defaulting to the full range if not specified
        if from_to is None:
            from_time, to_time = self.dt[0], self.dt[-1]
        else:
            try:
                from_time = np.datetime64(
                    datetime.strptime(from_to[0], "%Y.%m.%d %H:%M")
                )
                to_time = np.datetime64(datetime.strptime(from_to[1], "%Y.%m.%d %H:%M"))

            except ValueError:
                raise ValueError("from_to times must be in the format '%Y.%m.%d %H:%M'")

        # 2. Calculate the first valid index in the largest timeframe without NaN values in 'live' data
        # Use NumPy methods to identify NaN rows in 'live' data and find the first valid row
        live_data = self.market_data[-1][
            "live"
        ]  # Access the largest timeframe live data
        mask = np.isnan(live_data).any(axis=1)
        first_valid_index = np.argmax(~mask)  # Get the first index without NaN values

        # 3. Calculate the minimum start index to allow a complete sequence in the highest timeframe
        cursor = self.market_data[-1]["live_cursor"]  # Use cursor data for indexing
        # Calculate cursor index for valid sequences
        cursor_index = cursor[first_valid_index] + self.seq_len - 1
        start_index = np.argmax(cursor == cursor_index) + 1

        # 4. Determine the last valid index, accounting for ignored trailing data points
        end_index = len(self.dt) - self.nIgnore_last

        # 5. Adjust start and end indices to match the specified date range
        # Create a boolean mask for timestamps within the desired range
        valid_mask = (self.dt >= from_time) & (self.dt <= to_time)
        range_indices = np.nonzero(valid_mask)[
            0
        ]  # Get indices where the condition holds
        # Update start and end indices based on the valid mask range
        if range_indices.size > 0:
            start_index = max(range_indices[0], start_index)
            end_index = min(range_indices[-1], end_index)

        # 6. Assign the computed range to the dataset's index_range attribute
        self.index_range = (start_index, end_index)
        self.from_to = from_to


class SingleDataset:

    def __init__(
        self,
        market_data: pd.DataFrame,
        seq_len: int,
        from_to: list[str] = None,
        nIgnore_last: int = 25,
    ):
        """
        Initializes the SingleDataset object for working with single-timeframe data.

        Args:
            market_data (pd.DataFrame): DataFrame containing the market data. Must include a "TimeStamp" column.
            seq_len (int): Sequence length for model training.
            from_to (list[str], optional): Start and end date for filtering, in "%Y.%m.%d %H:%M" format.
            nIgnore_last (int): Number of bars to ignore at the end of the dataset.

        Attributes:
            dt (np.ndarray): Array of timestamps from the "TimeStamp" column.
            market_data (np.ndarray): Converted market data for efficient array-based operations.
            seq_len (int): Sequence length for data samples.
            nIgnore_last (int): Number of trailing points ignored to avoid incomplete sequences.
            timeframe (str): Timeframe inferred from the market data.
            columns (list[str]): List of column names from the market data.
        """

        # get the time frame of data
        timeframe = time_utils.get_timeframe(market_data)

        # Extract timestamps and drop the "TimeStamp" column
        self.dt = np.array(market_data["TimeStamp"])
        market_data = market_data.drop("TimeStamp", axis=1)

        # Set dataset properties
        self.market_data = market_data.to_numpy()  # Converted to NumPy for efficiency
        self.seq_len = seq_len
        self.nIgnore_last = nIgnore_last
        self.timeframe = timeframe
        self.columns = market_data.columns.tolist()

        # Set the valid index range based on the date range
        self.set_valid_range(from_to)

    def __len__(self):
        """
        Returns the number of valid data points in the dataset.

        Returns:
            int: The length of the valid dataset range.
        """
        return self.index_range[1] - self.index_range[0] + 1

    def __getitem__(self, index: int):
        """
        Retrieves a sequence of data for a given index.

        Args:
            index (int): Index of the data point to retrieve. Must be within the valid range.

        Returns:
            Data: An instance of the `Data` class containing the sequence data.

        Raises:
            ValueError: If the index is out of bounds.
        """
        # Ensure the index is within valid bounds
        if index < 0 or index >= len(self):
            raise ValueError(f"Index must be in range 0 to {len(self) - 1}.")
        index += self.index_range[0]  # Align with internal data range

        # Extract the sequence of timestamps and corresponding data
        dt = self.dt[index - self.seq_len : index]
        data = self.market_data[index - self.seq_len : index]

        # Return the data wrapped in a Data class instance
        return create_sample_df(dt, data, self.columns, self.timeframe)

    def set_valid_range(self, from_to: list[str] = None):
        """
        Sets the index range for valid data retrieval based on the specified date range.

        Args:
            from_to (list[str], optional): List containing start and end datetime strings
                                           in "%Y.%m.%d %H:%M" format. Defaults to the entire data range.
        """
        # Default to the full range if no `from_to` is provided
        if from_to is None:
            from_time, to_time = self.dt[0], self.dt[-1]
        else:
            try:
                from_time = np.datetime64(
                    datetime.strptime(from_to[0], "%Y.%m.%d %H:%M")
                )
                to_time = np.datetime64(datetime.strptime(from_to[1], "%Y.%m.%d %H:%M"))
            except ValueError:
                raise ValueError("from_to times must be in the format '%Y.%m.%d %H:%M'")

        # Find the first valid index without NaN values
        mask = np.isnan(self.market_data).any(axis=1)
        first_valid_index = np.argmax(~mask) + self.seq_len

        # Determine the last valid index, accounting for trailing points to ignore
        end_index = len(self.dt) - self.nIgnore_last

        # Adjust start and end indices to the date range specified in `from_to`
        valid_mask = (self.dt >= from_time) & (self.dt <= to_time)
        range_indices = np.nonzero(valid_mask)[0]
        if range_indices.size > 0:
            first_valid_index = max(range_indices[0], first_valid_index)
            end_index = min(range_indices[-1], end_index)

        # Assign the range to the class attributes
        self.index_range = (first_valid_index, end_index)
        self.from_to = from_to
