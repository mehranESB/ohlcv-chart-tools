# This test is for importing and exporting data for CSV files
from pathlib import Path
import pandas as pd
import chartDL.utils.csv as utils  # Assuming utils contains your functions

# Write your CSV file path
csv_file_path = Path("./DATA/forex/monthly/histdata/EURUSD/DAT_MT_EURUSD_M1_202411.csv")

# Import with pandas to see the structure of the file
df_raw = pd.read_csv(csv_file_path)

# Print 5 rows of the raw imported file with pandas
print("Raw format: ")
print(df_raw.head())

# Define column names and datetime format for import
column_names = ["Date", "Time", "Open", "High", "Low", "Close", "Volume"]
datetime_format = "%Y.%m.%d %H:%M"

# Import with utils format
df_import = (
    utils.import_ohlcv_from_csv(  # Adjusted function name to match your definition
        csv_file_path,
        header=False,
        column_names=column_names,
        datetime_format=datetime_format,
    )
)

# Print imported DataFrame with utils module
print("Imported format: ")
print(df_import.head())  # Prints the head of the imported DataFrame

# Export data into CSV
export_file_path = Path("./test.csv")
utils.export_ohlcv_to_csv(
    export_file_path,
    df_import,
    header=True,
    date_format="%Y-%m-%d",
    time_format="%H:%M:%S",
)

# Read exported CSV and print head of data
df_export = pd.read_csv(export_file_path)
print("Export format: ")
print(df_export.head())
