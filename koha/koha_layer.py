import torch
from torch.nn.parameter import Parameter
from .config import KohaLayerConfig
from torch.nn import Embedding
from math import sqrt

class KohaLayer(torch.nn.Module):
    def __init__(self, config:KohaLayerConfig):
        super().__init__()
        self.unit_num = config.unit_num
        self.emb_dim = config.emb_dim
        self.receptive_field = config.receptive_field
        self.neg_sampling_num = config.neg_sampling_num
        self.EPS = 1e-15
        self.signatures = Embedding(self.unit_num, self.emb_dim, sparse= config.sparse)
        self.weights = Parameter(torch.empty((self.unit_num, self.receptive_field, self.emb_dim)), requires_grad=False)
        self.previous_winners = []
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.signatures, a=sqrt(5))
        torch.nn.init.kaiming_uniform_(self.weights, a=sqrt(5))

    
    def gaussian_neighborhood(num_neurons, sigma, inverse = False):
        center_neuron = num_neurons // 2
        neuron_indices = torch.arange(num_neurons)
        distance = torch.abs(neuron_indices - center_neuron)

        # smaller sigmas result in narrower neighborhood functions
        influence = torch.exp(-distance.pow(2) / (2 * torch.tensor(sigma).pow(2)))
        if inverse:
            influence_prob = 1 - influence
            influence_prob[center_neuron] = 0
            influence_prob = influence_prob / influence_prob.sum()
        else:
            influence[center_neuron] = 0
            influence_prob = influence / influence.sum()
        return influence_prob
    
    def _neghborhood_function(self, winner, sigma, inverse):
        distribution = self.gaussian_neighborhood(self.unit_num, sigma, inverse)
        remaining = self.unit_num - winner
        center_neuron = self.unit_num // 2
        rearranged_distribution = torch.cat([distribution[center_neuron:], distribution[:center_neuron]])
        return torch.cat([rearranged_distribution[- winner :], rearranged_distribution[: remaining]])

    def positive_neighborhood_function(self, winner, sigma):
        return self._neghborhood_function(winner, sigma, inverse=False)

    def negative_neighborhood_function(self, winner, sigma):
        return self._neghborhood_function(winner, sigma, inverse=True)
    
    
# have a key value pair for each synapse. perform competitive learning for each synapse. take the scores of the BMUs and perform softmax, combine them, create one combines signature.
# no hebbian update rule required. the synapse as well as the signature get udpated through gradient descent. 