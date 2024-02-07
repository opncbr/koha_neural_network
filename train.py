from koha.koha_network import KohaNetwork
from koha.config import KohaConfig, KohaNetworkConfig

from torch.utils.data import DataLoader
import torch
import numpy as np
import os
from tqdm import tqdm

device = "mps"  #'cuda' if torch.cuda.is_available() else 'cpu'
dataset = "shakespeare"
saved_model_dir = "saved_models/koha_model"
sequence_length = 100
batch_size = 64
shuffle = True
num_workers = 0


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, block_size, train):
        self.data = np.memmap(
            os.path.join(f"data/{dataset}", "train.bin" if train else "val.bin"),
            dtype=np.uint16,
            mode="r",
        )
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = torch.from_numpy((self.data[idx : idx + self.block_size]).astype(np.int64))
        y = torch.from_numpy(
            (self.data[idx + 1 : idx + 1 + self.block_size]).astype(np.int64)
        )
        return x, y


data = TextDataset(dataset=dataset, block_size=sequence_length, train=True)
data_loader = DataLoader(
    data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
)

network_config = KohaNetworkConfig()
block_config = KohaConfig()
koha_network = KohaNetwork(network_config, block_config)


def train(model):
    model.train()
    # total_loss = 0
    for x, _ in tqdm(data_loader):
        model.koha_layer.initialize_state(batch_size)
        losses = []
        for i in range(sequence_length):
            input = x[:, i].unsqueeze(1)
            loss, out = model(input)
            losses.append(loss.item())
        print("loss", np.average(losses))


train(koha_network)

# torch.save(koha_input_layer.state_dict(), saved_model_dir)
