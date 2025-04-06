import numpy as np
import pandas as pd
from datetime import datetime


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
