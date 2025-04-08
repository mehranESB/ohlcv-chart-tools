from torch.utils.data import Dataset, DataLoader
from chartDL.dataset import MultiDataset


class CustomDataset(Dataset):
    def __init__(self, dataset: MultiDataset):
        self.dataset = dataset  # store to use it as data source

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        multi_time_frame_data = self.dataset[index]

        retrive_data = {}
        for data in multi_time_frame_data:
            name = f"x_{data.timeframe}"
            value = data[["High", "Low"]]
            retrive_data[name] = value

        return retrive_data


# Initialize MultiDataset (use your own path and sequence length)
multi_dataset = MultiDataset("./DATA/multi_view/EURUSD-1h", seq_len=128)

# Create a DataLoader for batching and shuffling
data_loader = DataLoader(CustomDataset(multi_dataset), batch_size=64, shuffle=True)

# Iterate through batches during model training
for batch in data_loader:
    ...
