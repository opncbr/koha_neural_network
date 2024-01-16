import torch
from torch.nn import Embedding
from math import sqrt
from .config import KohaNetworkConfig, KohaBlockConfig
from koha_block import KohaBlock

class KohaNetwork(torch.nn.Module):
    def __init__(self, network_config:KohaNetworkConfig, block_config: KohaBlockConfig):
        super().__init__()
        self.vocab_size = network_config.vocab_size
        self.emb_dim = network_config.emb_dim
        self.embeddings = Embedding(self.vocab_size, self.emb_dim, sparse=True)
        self.koha_blocks = [KohaBlock(block_config, True) if ind == 0  else KohaBlock(block_config, False) for ind in range (network_config.context)]
        self.reset_parameters()
    

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.embeddings.weight, a=sqrt(5))
