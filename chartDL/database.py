import numpy as np
import pandas as pd
from datetime import datetime
from .utils import time as time_utils


class DataFeeder:

    def __init__(self, df: pd.DataFrame, from_to: list[str] = None):
        """
        Initializes a DataFeeder instance to retrieve OHLCV data row-by-row.

        Args:
            df (pd.DataFrame): DataFrame containing OHLCV data with 'TimeStamp' as datetime column.
            from_to (list[str], optional): List with two strings specifying start and end datetime
                                           for data filtering, in "%Y.%m.%d %H:%M" format.
        """
        # Filter data by time range if specified
        if from_to is not None:
            start_time, end_time = [
                datetime.strptime(t, "%Y.%m.%d %H:%M") for t in from_to
            ]
            mask = (df["TimeStamp"] >= start_time) & (df["TimeStamp"] <= end_time)
            df = df[mask]

        self.df = df.reset_index(drop=True)  # Reset index for clean iteration

        # Extract datetime as an array of Python datetime objects
        self.dt = df["TimeStamp"].dt.to_pydatetime()

        # Convert OHLCV data to a numpy float32 array
        columns = ["Open", "High", "Low", "Close", "Volume"]
        self.data = df[columns].to_numpy(dtype=np.float32)

        # Initialize cursor
        self.cursor = 0

    def __len__(self):
        """Return the number of data rows available in the DataFeeder."""
        return self.data.shape[0]

    def reset(self):
        """Reset the cursor to the beginning of the data."""
        self.cursor = 0

    def has_data(self):
        """Return True if there is more data to retrieve, False otherwise."""
        return self.cursor < len(self)

    def retrieve(self):
        """
        Retrieve the data at the current cursor position, including timestamp and OHLCV values.

        Returns:
            tuple: (time, data), where 'time' is a datetime object and 'data' is a numpy array of OHLCV values.
        """
        if not self.has_data():
            raise IndexError("No more data to retrieve.")

        time = self.dt[self.cursor]
        data = self.data[self.cursor]
        self.cursor += 1
        return time, data


class Database:

    def __init__(self, time_frame: str = "1m", initial_len: int = 10000):
        """
        Initializes a Database object for storing market data (OHLCV) and additional data
        such as indicators and charting information.

        Args:
            time_frame (str): Timeframe for data processing (e.g., "1m", "5m").
            initial_len (int): Initial length of the data container arrays.
                               Allows for memory allocation upfront to improve efficiency.
        """

        # time frame for storing data
        if not time_utils.check_valid_timeframe(time_frame):
            raise ValueError(f"Unsupported timeframe: '{time_frame}'.")
        self.time_frame = time_frame

        # Initialize time array with Unix epoch time as default values
        self.dt = np.full(
            initial_len, datetime(1970, 1, 1), dtype="O"
        )  # 'O' for datetime objects

        # Initialize OHLCV data as a numpy array of zeros
        self.data_columns = ["Open", "High", "Low", "Close", "Volume"]
        self.data = np.zeros((initial_len, len(self.data_columns)), dtype=np.float32)

        # Placeholder lists for charts and indicators
        self.charts = []  # List to store charting objects
        self.indicators = []  # List to store indicator objects

        # Attributes to track the current OHLCV data and timestamp
        self.current_opening_time = datetime(1970, 1, 1)  # Default current opening time
        self.current_ohlcv = np.zeros(
            len(self.data_columns), dtype=np.float32
        )  # Default OHLCV values

        # Internal state attributes
        self.cursor = -1  # Index to track the last updated position in the data array
        self.is_new_bar = False  # Flag to indicate if a new bar of data has been added

    def update_on_OHLCV(self, time, ohlcv):
        """
        Updates the database with incoming OHLCV data and manages new bar creation.

        Args:
            time (datetime): The opening time for the incoming OHLCV data.
            ohlcv (array-like): OHLCV data as a list or array with shape (5,).
        """
        # Check if the incoming data signifies a new bar
        self.is_new_bar = time_utils.is_new_bar(
            time, self.current_opening_time, self.time_frame
        )

        if self.is_new_bar:
            # Round to the nearest opening time based on timeframe
            self.current_opening_time = time_utils.round_open_time(
                time, self.time_frame
            )
            self.current_ohlcv = np.array(ohlcv, dtype=np.float32)
            # Advance cursor to the next row
            self.cursor += 1
            # Update datetime array for the new bar
            self.dt[self.cursor] = self.current_opening_time
        else:
            # Update current OHLCV values for the same bar
            self.current_ohlcv[1] = max(self.current_ohlcv[1], ohlcv[1])  # Update High
            self.current_ohlcv[2] = min(self.current_ohlcv[2], ohlcv[2])  # Update Low
            self.current_ohlcv[3] = ohlcv[3]  # Update Close
            self.current_ohlcv[4] += ohlcv[4]  # Accumulate Volume

        # Save updated OHLCV data to the current row in the data array
        self.data[self.cursor, :5] = self.current_ohlcv

        # compute indicators
        self.compute_indicators()

        # update attached charts
        self.update_charts()

    def export_data(self, dtype: str = "array"):
        """
        Exports the database data from the start up to the current cursor position in the specified format.

        Args:
            dtype (str): The export format, either "df" (returns a DataFrame) or "array" (returns numpy arrays).

        Returns:
            If dtype is "array":
                Tuple of (dt, data), where dt is an array of timestamps, and data is the OHLCV data array up to the cursor.
            If dtype is "df":
                Pandas DataFrame containing 'TimeStamp' and OHLCV data up to the cursor position.
        """
        # Slice time and data arrays up to the current cursor position
        dt = self.dt[: self.cursor + 1]
        data = self.data[: self.cursor + 1]

        if dtype == "array":
            # Return raw numpy arrays for time and OHLCV data
            return dt, data
        elif dtype == "df":
            # Convert data to a DataFrame
            df = pd.DataFrame(data, columns=self.data_columns)
            df["TimeStamp"] = dt  # Add TimeStamp column

            # Reorder columns to have TimeStamp first
            return df[["TimeStamp"] + self.data_columns]
        else:
            raise ValueError(f"Unsupported dtype: {dtype}. Choose 'df' or 'array'.")

    def add_indicator(self, indicator):
        """
        Adds an indicator to the database's indicators list and modifies the data container
        to store the indicator's computed values.

        Args:
            indicator: An indicator object to add, which should have `nColumns` for the number of
                       data columns it uses and `column_name()` method to provide column names.
        """
        # Get the current shape of the data array
        nRows, nCols = self.data.shape

        # Retrieve new column names from the indicator
        new_columns = indicator.column_name()
        num_new_columns = len(new_columns)

        # Define index range in data array for the new indicator's columns
        # This helps identify where the indicator data is stored in the expanded data array
        indicator.idx_range = (nCols, nCols + num_new_columns - 1)

        # Expand data container to include columns for indicator data
        # New columns are initialized to zero
        new_data = np.zeros((nRows, num_new_columns), dtype=np.float32)
        self.data = np.hstack((self.data, new_data))

        # Append new column names to the data_columns attribute
        self.data_columns += new_columns

        # Add the indicator object to the indicators list for future reference
        self.indicators.append(indicator)

    def compute_indicators(self):
        """
        Computes the values of all attached indicators on the data up to the current cursor position.
        Updates the indicator values in the database's data container.

        Each indicator's computed values are stored in pre-allocated columns within `self.data`,
        based on the `idx_range` attribute of each indicator.
        """
        # Return if no indicators are attached to avoid unnecessary computation
        if not self.indicators:
            return

        # Export data up to the current cursor position in "array" format
        dt, data = self.export_data("array")

        # Compute and store values for each indicator
        for indicator in self.indicators:
            # Compute indicator values
            data_i = indicator.compute(dt, data)

            # Retrieve index range for the indicator's columns in the data array
            idx_range = indicator.idx_range

            # Update the data container with computed indicator values
            self.data[self.cursor, idx_range[0] : idx_range[1] + 1] = data_i

    def cols_info(self):
        """
        Returns a dictionary with information about each column in the data.
        Provides the default mode (e.g., "chart", "log") for standard OHLCV columns
        and any custom modes defined for indicator columns.

        Returns:
            dict: A dictionary where keys are column names and values are their modes.
        """
        # Base data columns information (OHLCV)
        info = {
            "TimeStamp": "time",
            "Open": "chart",
            "High": "chart",
            "Low": "chart",
            "Close": "chart",
            "Volume": "log",
        }

        # Append indicator columns info, if any
        if self.indicators:
            for indicator in self.indicators:
                # Update info dictionary with column names and modes from each indicator
                info.update({name: indicator.mode for name in indicator.column_name()})

        return info

    def update_charts(self):
        """
        Updates all attached chart objects with the latest data up to the current cursor position.

        This method will export the time and data arrays up to the current data point,
        and then pass this data to each attached chart for updating.

        Returns:
            None
        """
        # Check if there are any charts attached; if not, exit early
        if not self.charts:
            return

        # Retrieve all time and data values up to the current cursor
        dt, data = self.export_data("array")

        # Loop through each chart in the charts list and update with current data
        for chart in self.charts:
            # Call the chart's update method with the latest data
            chart.update(dt, data)
