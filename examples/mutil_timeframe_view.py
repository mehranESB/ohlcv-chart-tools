from pathlib import Path
import chartDL.utils.csv as utils
import chartDL.preprocess as pp
import chartDL.indicator as indc

# Define the path to the source CSV file containing OHLCV data at a 1-hour frequency
source_file_path = Path("./DATA/csv/EURUSD-1h.csv")

# Import the source OHLCV data from the CSV into a DataFrame
# This function reads and formats the data, including parsing headers and date-time format
source_df = utils.import_ohlcv_from_csv(
    source_file_path, header=True, datetime_format="%Y-%m-%d %H:%M:%S"
)

# Specify higher timeframes to generate views from the base (1-hour) timeframe in source_df
# `multi_timeframe_view` uses the first entry as the target timeframe (inferred from source_df)
higher_timeframes = ["4h", "1d"]

# Initialize indicators to calculate on each timeframe's database
# Here, Simple Moving Average (SMA) and Exponential Moving Average (EMA) indicators are added
indicators = [indc.SMA(20), indc.EMA(10)]

# Define the directory where multi-timeframe data will be saved
# Separate subdirectories will be created for 'origin' and 'live' data
save_path = "./DATA/multi_view/EURUSD"

# Call the multi_timeframe_view function to generate views for each specified timeframe
# The function processes the data, applies indicators, and saves results in the specified path
mvt_dfs, time_frames = pp.multi_timeframe_view(
    source_df, higher_timeframes, indicators=indicators, save_to=save_path
)

# Display the first few rows of each DataFrame for verification
# This helps confirm the data corresponds to each specified timeframe
print("Multi-timeframe view data:")
for timeframe, df in zip(time_frames, mvt_dfs):
    print(f"\nTimeframe: {timeframe}")
    print(df.head())
