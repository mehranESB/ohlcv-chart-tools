from pathlib import Path
import chartDL.utils.csv as utils  # Assuming utils contains the necessary functions

# Define paths to monthly and yearly data directories
monthly_path = Path("./DATA/forex/monthly/histdata/EURUSD")
yearly_path = Path("./DATA/forex/yearly/histdata/EURUSD")

# Define specifications for importing the CSV files
column_names = ["Date", "Time", "Open", "High", "Low", "Close", "Volume"]
datetime_format = "%Y.%m.%d %H:%M"
header = False

# Merge the imported yearly and monthly data into a single DataFrame
df_merged = utils.merge_monthly_and_yearly_csv(
    monthly_path,
    yearly_path,
    header=header,
    column_names=column_names,
    datetime_format=datetime_format,
)

# Print summary information about the merged data
print(
    f"Data range: {df_merged['TimeStamp'].iloc[0]} to {df_merged['TimeStamp'].iloc[-1]}"
)
print(f"Total number of rows: {len(df_merged)}")
print("Head of the merged data:")
print(df_merged.head())
