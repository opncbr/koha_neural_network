import torch
from torch.nn.parameter import Parameter
from torch.nn import Embedding
from math import sqrt
import random
from .config import KohaInputLayerConfig

class KohaInputLayer(torch.nn.Module):
    # The Koha input layer performs (in the case of text) a one-to-one mapping from token to embedding. It's training is equivalent to that of word2vec
    def __init__(self, config:KohaInputLayerConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.emb_dim = config.emb_dim
        self.lr = config.lr
        self.window_size = config.window_size
        self.neg_sampling_num = config.neg_sampling_num
        self.sample = config.sample
        self.total = 0
        self.EPS = 1e-15

        #self.signatures = Parameter(torch.empty((self.vocab_size, self.emb_dim)), requires_grad=True)
        self.signatures = Embedding(self.vocab_size, self.emb_dim, sparse= config.sparse)
        self.signature_optimizer = torch.optim.SparseAdam(list(self.parameters()), lr=0.01)
        self.previous_winners = []
        #self.register_buffer("unit_occurrence", torch.ones(self.vocab_size, dtype=torch.int32))
        self.register_buffer("negative_unit_filter", torch.randint(0,self.vocab_size,(1, self.vocab_size * config.unit_filter_scale)).squeeze(0)) # used for negative sampling
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.signatures.weight, a=sqrt(5))

    def clear_previous_winners(self):
        self.previous_winners = []

    #def _subsample(self, x):
    #    rands = torch.rand(len(x))
    #    print(rands.shape)
    #    freq = self.unit_occurrence[x] / self.total
    #    print(freq.shape)
    #    token_prob = (torch.sqrt(freq / self.sample) + 1) * (self.sample / freq)
    #    print(token_prob.shape)
    #    to_be_kept = token_prob > rands
    #    sampled_tokens = x[to_be_kept]
    #    return sampled_tokens

    #def _update_occurrence(self, x):
    #    #count = torch.ones_like(x)
    #    self.unit_occurrence[x] += 1 #.scatter_add_(0, index=x, src=count)
    #    total += 1

    def _update_negative_unit_filter(self, x):
        rand_ind = random.randint(0, self.vocab_size)
        self.negative_unit_filter[rand_ind] = x

    def _get_positive_samples(self, x):
        if self.previous_winners != []:
            context, target = torch.tensor(self.previous_winners), torch.ones(len(self.previous_winners), dtype=torch.int32) * x
            self.previous_winners.append(x)
            if len(self.previous_winners) > self.window_size:
                del self.previous_winners[0]
        else:
            self.previous_winners.append(x)
            context, target = None, None
        return context, target

    def _get_negative_samples(self, x):
        target = torch.ones(self.neg_sampling_num, dtype=torch.int32) * x
        rand = torch.randint(0, self.vocab_size, (1, self.neg_sampling_num)).squeeze(0)
        context = self.negative_unit_filter[rand]
        return context, target

    def loss(self, x):
        
        # compute positive loss
        context, target = self._get_positive_samples(x)
        positive_loss = 0
        if context != None:
            contex_emb = self.signatures(context)
            target_emb = self.signatures(target)
            out = (contex_emb * target_emb).sum(dim=-1).view(-1)
            positive_loss = -torch.log(torch.sigmoid(out) + self.EPS).mean()
            
        # compute negative loss
        context, target = self._get_negative_samples(x)
        contex_emb = self.signatures(context)
        target_emb = self.signatures(target)
        out = (contex_emb * target_emb).sum(dim=-1).view(-1)
        negative_loss = -torch.log(1 - torch.sigmoid(out) + self.EPS).mean()
        
        return positive_loss + negative_loss
        

    def forward(self, x):
        # update unit frequencies
        # self._update_occurrence(x)
        # perform subsampling 
        
        # compute signature gradients
        self.signature_optimizer.zero_grad()
        loss = self.loss(x)
        loss.backward()

        # update signatures
        self.signature_optimizer.step()

        return x, loss
        