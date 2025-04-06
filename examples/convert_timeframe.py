from pathlib import Path
import chartDL.utils.csv as utils
import chartDL.preprocess as pp

# Define the path to the source CSV file (1-minute data)
source_file_path = Path("./DATA/csv/EURUSD-1m.csv")

# Import the source CSV file into a DataFrame, using the specified datetime format
# Assuming 'import_ohlcv_from_csv' is a utility function in 'utils' to read OHLCV data.
# `header=True` assumes the file has a header row, and `datetime_format` specifies the date parsing format.
source_df = utils.import_ohlcv_from_csv(
    source_file_path, header=True, datetime_format="%Y-%m-%d %H:%M:%S"
)

# Define the target timeframe for conversion (e.g., "1h" for 1-hour data).
target_timeframe = "1h"

# Perform the timeframe conversion using the `convert_timeframe` function.
# This function takes the lower timeframe `source_df` and outputs data at the higher `target_timeframe`.
df_converted = pp.convert_timeframe(source_df, to_time=target_timeframe)

# Output: Print a preview of the source data (lower timeframe) and the converted data (higher timeframe).
print(f"Source data in lower timeframe (rows={len(source_df)}):")
print(source_df.head())  # Preview the first few rows of the source data.

print(f"Converted data in higher timeframe (rows={len(df_converted)}):")
print(df_converted.head())  # Preview the first few rows of the converted data.
