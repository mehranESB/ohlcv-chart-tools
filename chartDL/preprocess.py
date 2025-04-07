from . import database as db
from .utils.time import get_timeframe
from .utils.csv import export_multi_view_data


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


def multi_timeframe_view(
    source_df, higher_timeframes: list[str], indicators: list = [], save_to: str = None
):
    """
    Processes OHLCV data for multiple timeframes from a source DataFrame, computes indicators,
    synchronizes higher timeframe data, and optionally saves results.

    Args:
        source_df (pd.DataFrame): DataFrame with OHLCV data for the base (target) timeframe.
        higher_timeframes (list of str): List of additional higher timeframes to generate views for.
        indicators (list, optional): List of indicator objects to compute on each database.
        save_to (str, optional): Directory path to save processed data (if provided).

    Returns:
        tuple: List of DataFrames with synchronized data for each timeframe and list of timeframes.
    """

    # Infer the target timeframe from source_df and add unique higher timeframes
    target_timeframe = get_timeframe(source_df)
    time_frames = [target_timeframe] + list(set(higher_timeframes) - {target_timeframe})

    # Initialize data feeder and databases
    data_feeder = db.DataFeeder(source_df)
    init_len = len(data_feeder)
    view_databases = [
        db.Database(timeframe, initial_len=init_len) for timeframe in time_frames
    ]

    # Attach any indicators to each database
    if indicators:
        for database in view_databases:
            for indicator in indicators:
                database.add_indicator(indicator)

    # Set up LiveBackups for synchronized higher timeframe data
    target_db = view_databases[0]
    higher_dbs = view_databases[1:]
    live_backup = db.LiveBackups(target_db, higher_dbs)

    # Process each row of data from the feeder and update each database
    while data_feeder.has_data():
        time, data = data_feeder.retrieve()

        # Update each database with new data
        for database in view_databases:
            database.update_on_OHLCV(time, data)

        # Synchronize higher timeframe data with the target database
        live_backup.update()

    # Export synchronized data as DataFrames
    exported_dataframes, time_frames = live_backup.export_data("df")

    # Optionally save data if 'save_to' is provided
    if save_to is not None:
        results = []
        for database, live_df, timeframe in zip(
            view_databases, exported_dataframes, time_frames
        ):
            results.append(
                {
                    "timeframe": timeframe,
                    "origin": database.export_data("df"),  # Original timeframe data
                    "live": live_df,  # Live synchronized data
                }
            )
        export_multi_view_data(save_to, results)

    return exported_dataframes, time_frames
