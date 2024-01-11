import torch
from .config import KohaLayerConfig
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from math import sqrt

class State():
    def __init__(self, window_size):
        self.window_size = window_size
        self.pos_past = None
        self.neg_past = None
    
    def state_transition(self, pos, neg):

class KohaLayer(torch.nn.Module):
    def __init__(self, config:KohaLayerConfig, first_layer: bool):
        super().__init__()
        self.first_layer = first_layer
        self.unit_num = config.unit_num
        self.emb_dim = config.emb_dim
        self.receptive_field = config.receptive_field
        self.neg_sampling_num = config.neg_sampling_num
        self.keys = torch.nn.Linear(self.emb_dim, self.unit_num, bias=config.bias)
        self.values = Parameter(torch.empty((self.unit_num, self.emb_dim)))
        self.previous_winners = State(config.window_size)
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.values, a=sqrt(5))
    
    def forward(self, x):
        k = self.keys(x)

        # compute positive sample
        pos_distribution = F.softmax(k, dim=-1)
        # compute negative sample. If first layer == True, allow gradient flow (removes the need for pos/neg sampling for the embedding layer)
        if self.first_layer:
            neg_distribution = F.softmax(-k, dim=-1)
        else:
            with torch.no_grad():
                neg_distribution = F.softmax(-k, dim=-1)

        pos_values = pos_distribution @ self.v
        neg_values = neg_distribution @ self.v

        # create positive & negative samples
        
        # perform backprop

        # return layer state
        with torch.no_grad():
            y_prime = pos_distribution @ self.v
        return y_prime