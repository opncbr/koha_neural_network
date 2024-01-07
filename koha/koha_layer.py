import torch
from torch.nn.parameter import Parameter
from .config import KohaLayerConfig
from math import sqrt

class KohaLayer(torch.nn.Module):
    def __init__(self, config:KohaLayerConfig):
        super().__init__()
        self.unit_num = config.unit_num
        self.emb_dim = config.emb_dim
        self.window_size = config.window_size
        self.receptive_field = config.receptive_field
        self.signatures = Parameter(torch.empty((self.unit_num, self.emb_dim)), requires_grad=True)
        self.weights = Parameter(torch.empty((self.unit_num, self.receptive_field, self.emb_dim)), requires_grad=False)
        self.previous_winners = []
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.signatures, a=sqrt(5))
        torch.nn.init.kaiming_uniform_(self.weights, a=sqrt(5))

    