from . import database as db


def convert_timeframe(source_df, to_time: str = "1h"):
    """
    Converts OHLCV market data from a lower timeframe to a higher timeframe. The function
    processes the source dataframe, aggregates data into the specified higher timeframe,
    and returns a new dataframe containing the converted OHLCV data.

    Args:
        source_df (pd.DataFrame): Source dataframe containing OHLCV data at a lower time frame.
        to_time (str): Target time frame for conversion (e.g., "1h", "1d", etc.).

    Returns:
        pd.DataFrame: A dataframe containing OHLCV data in the new, higher timeframe.
    """

    # Validate the target timeframe to ensure it's a valid format (this might depend on your implementation)
    valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
    if to_time not in valid_timeframes:
        raise ValueError(
            f"Invalid target timeframe: {to_time}. Supported timeframes are: {valid_timeframes}"
        )

    # Create data feeder from the source DataFrame (assuming DataFeeder is a custom class to handle the data)
    data_feeder = db.DataFeeder(source_df)

    # Create a database for storing higher timeframe data (assuming Database is a custom class)
    init_len = len(data_feeder)
    database = db.Database(to_time, initial_len=init_len)

    # Conversion loop: iterate through the data feeder and update the database
    while data_feeder.has_data():
        # Retrieve the next time and data (OHLCV) from the data feeder
        time, data = data_feeder.retrieve()

        # Update the database with the retrieved data for the current time
        database.update_on_OHLCV(time, data)

    # Once all data has been processed, export the data as a Pandas DataFrame
    # You can specify "csv" or "array" for other formats if needed, but we'll return a DataFrame here
    df_converted = database.export_data("df")

    # Return the converted DataFrame containing the data in the new higher timeframe
    return df_converted
