import torch
from torch.nn import Embedding
from math import sqrt
from .config import KohaInputLayerConfig
import random

class KohaInputLayer(torch.nn.Module):
    # The Koha input layer performs (in the case of text) a one-to-one mapping from token to embedding. It's training is equivalent to that of word2vec
    def __init__(self, config:KohaInputLayerConfig):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.emb_dim = config.emb_dim
        self.window_size = config.window_size
        self.neg_sampling_num = config.neg_sampling_num
        self.EPS = 1e-15
        self.neg_iter = 0
        self.signatures = Embedding(self.vocab_size, self.emb_dim, sparse= config.sparse)
        self.signature_optimizer = torch.optim.SparseAdam(list(self.parameters()), lr=config.lr)
        self.previous_winners = []
        self.register_buffer("_negative_unigram", torch.randint(0,self.vocab_size,(1, self.vocab_size * config.neg_unigram_scale)).squeeze(0)) # used for negative sampling
        self.reset_parameters()
    

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.signatures.weight, a=sqrt(5))


    def clear_previous_winners(self):
        self.previous_winners = []


    def get_neg_unigram(self):
        return self._negative_unigram[: self.neg_iter]


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
        rand = torch.randint(0, self.neg_iter + 1, (1, self.neg_sampling_num)).squeeze(0)
        context = self._negative_unigram[rand]
        return context, target
    

    def update_neg_unigram(self, x):
        if self.neg_iter < self._negative_unigram.size(0) - 1:
            self._negative_unigram[self.neg_iter] = x
            self.neg_iter += 1
        else:
            rand = random.randint(0, self._negative_unigram.size(0) - 1)
            self._negative_unigram[rand] = x


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
        print(self.neg_iter, x)
        # compute signature gradients
        self.signature_optimizer.zero_grad()
        loss = self.loss(x)
        loss.backward()

        # update negative unigram
        self.update_neg_unigram(x)

        # update signatures
        self.signature_optimizer.step()

        return x, loss
        