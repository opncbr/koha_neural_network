from koha.koha_network import KohaNetwork
from koha.config import KohaConfig

from torch.utils.data import DataLoader
import torch
import numpy as np
import os
from tqdm import tqdm
from koha.helpers import getenv
import inspect


device = "mps"  #'cuda' if torch.cuda.is_available() else 'cpu'
dataset = "shakespeare"
saved_model_dir = "saved_models/koha_model"
sequence_length = 100
batch_size = 64
shuffle = True
num_workers = 0

DEBUG = getenv("DEBUG", 0)


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


def train(model, data_loader, optimizer):
    model.train()
    # total_loss = 0
    for x, y in tqdm(data_loader):
        model.koha_layer.initialize_state(batch_size)
        losses = []
        for i in range(sequence_length):
            input = x[:, i].unsqueeze(1)
            target = y[:, i].unsqueeze(1)
            out, loss = model(input, target)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("loss", np.average(losses))


data = TextDataset(dataset=dataset, block_size=sequence_length, train=True)
data_loader = DataLoader(
    data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
)

vocab_size = 50256
koha_config = KohaConfig()
koha_network = KohaNetwork(vocab_size, koha_config)
optimizer = koha_network.configure_optimizer(koha_config)

train(koha_network, data_loader, optimizer)


# torch.save(koha_input_layer.state_dict(), saved_model_dir)
