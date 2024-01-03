from koha.config import KohaInputLayerConfig
from koha.koha_input_layer import KohaInputLayer
import torch
import numpy as np
import os
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset = "shakespeare"
sequence_length = 100
batch_size = 1

data_dir = os.path.join("data", dataset)
train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")

config = KohaInputLayerConfig()
koha_input_layer = KohaInputLayer(config)

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - sequence_length, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + sequence_length]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + sequence_length]).astype(np.int64))
            for i in ix
        ]
    )
    if device == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


total_loss = 0
for i in tqdm(range(1000)):
    x, _ = get_batch(split='train')
    x = x.squeeze()
    for token in x:
        _, loss = koha_input_layer(token.item())
        total_loss += loss.item()