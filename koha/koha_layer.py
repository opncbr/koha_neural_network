import torch

from .config import KohaConfig
from .koha_module import KohaModule
from .koha_state import KohaState


class KohaLayer(torch.nn.Module):
    def __init__(self, config: KohaConfig):
        super().__init__()
        self.koha_state = KohaState(config)
        self.koha_module = KohaModule(config)

    def forward(self, x):
        X, Z, M = self.koha_state(x)
        pos_outputs, neg_outputs = self.koha_module(X, Z, M)
        self.koha_state.update_state(pos_outputs)
        return pos_outputs, neg_outputs
