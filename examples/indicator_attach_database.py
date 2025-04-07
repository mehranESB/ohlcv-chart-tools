from pathlib import Path
import chartDL.utils.csv as utils
import chartDL.database as db
import chartDL.indicator as indc

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

# Create data feeder from the source DataFrame (assuming DataFeeder is a custom class to handle the data)
data_feeder = db.DataFeeder(source_df)

# Create a database for storing higher timeframe data (assuming Database is a custom class)
init_len = len(data_feeder)
database = db.Database(target_timeframe, initial_len=init_len)

# Here, Simple Moving Average (SMA) and Exponential Moving Average (EMA) indicators are added
database.add_indicator(indc.SMA(20))
database.add_indicator(indc.EMA(10))

# Conversion loop: iterate through the data feeder and update the database
while data_feeder.has_data():
    # Retrieve the next time and data (OHLCV) from the data feeder
    time, data = data_feeder.retrieve()

    # Update the database with the retrieved data for the current time
    database.update_on_OHLCV(time, data)
df_converted = database.export_data("df")

# Output: Print a preview of the source data (lower timeframe) and the converted data (higher timeframe).
print(f"Source data in lower timeframe (rows={len(source_df)}):")
print(source_df.head())  # Preview the first few rows of the source data.

print(f"Converted data in higher timeframe (rows={len(df_converted)}):")
print(df_converted.head())  # Preview the first few rows of the converted data.
