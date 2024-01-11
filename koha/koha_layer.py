import torch
from .config import KohaLayerConfig
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from math import sqrt

class KohaLayer(torch.nn.Module):
    def __init__(self, config:KohaLayerConfig):
        super().__init__()
        self.unit_num = config.unit_num
        self.emb_dim = config.emb_dim
        self.sigma = config.sigma
        self.receptive_field = config.receptive_field
        self.neg_sampling_num = config.neg_sampling_num
        self.keys = torch.nn.Linear(self.emb_dim, self.unit_num, bias=config.bias)
        self.values = Parameter(torch.empty((self.unit_num, self.emb_dim)))
        self.previous_winners = []
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.values, a=sqrt(5))
    
    def forward(self, x):
        k = self.keys(x)

        pos_distribution = F.softmax(k, dim=-1)
        with torch.no_grad():
            neg_distribution = F.softmax(-k, dim=-1)
        pos_values = pos_distribution @ self.v
        neg_values = neg_distribution @ self.v