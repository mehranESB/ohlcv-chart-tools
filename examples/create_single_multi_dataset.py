from pathlib import Path
from chartDL.utils.csv import import_ohlcv_from_csv
from chartDL.dataset import MultiDataset, SingleDataset

# create single time frame dataset
file_path = Path("./DATA/csv/EURUSD-15m.csv")
df = import_ohlcv_from_csv(file_path, header=True, datetime_format="%Y-%m-%d %H:%M:%S")
dataset = SingleDataset(df, seq_len=32)

# create mutli time frame dataset
multi_folder_path = Path("./DATA/multi_view/EURUSD-1h")
multi_dataset = MultiDataset(
    multi_folder_path, seq_len=32, from_to=["2020.04.01 16:30", "2022.10.01 00:00"]
)

print(f"single dataset in path: {file_path}")
print(dataset[100])

# pick a sample and convert and print on screen
print(f"multi time frame dataset in folder path: {multi_folder_path}")
mulit_data = multi_dataset[20]
for df in mulit_data:
    print(f"timeframe: {df.attrs['timeframe']}")
    print(df.head())
