import torch
from .config import KohaLayerConfig
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from math import sqrt
import inspect

class State():
    def __init__(self, window_size):
        self.window_size = window_size
        self.pos_past = None
        self.neg_past = None
    
    def state_transition(self, pos, neg):

    def get_positive_samples(self):

    def get_negative_samples(self):

class KohaLayer(torch.nn.Module):
    def __init__(self, config:KohaLayerConfig, first_layer: bool):
        super().__init__()
        self.first_layer = first_layer
        self.unit_num = config.unit_num
        self.emb_dim = config.emb_dim
        self.receptive_field = config.receptive_field
        self.EPS = 1e-15
        self.keys = torch.nn.Linear(self.emb_dim, self.unit_num, bias=config.bias)
        self.values = Parameter(torch.empty((self.unit_num, self.emb_dim)))
        self.layer_optimizer = self.configure_optimizer(config)
        self.state = State(config.window_size)
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
        self.layer_optimizer.zero_grad()
        loss = self.loss(x)
        loss.backward()
        self.layer_optimizer.step()

        # return layer state
        with torch.no_grad():
            y_prime = pos_distribution @ self.v
        return y_prime
    

    def loss(self):
        positive_loss, negative_loss = 0, 0
        # compute positive loss
        context, target = self.state.get_positive_samples()
        if context != None:
            out = (context * target).sum(dim=-1).view(-1)
            positive_loss = -torch.log(torch.sigmoid(out) + self.EPS).mean()
            
        # compute negative loss
        context, target = self.state.get_negative_samples()
        if context != None:
            out = (context * target).sum(dim=-1).view(-1)
            negative_loss = -torch.log(1 - torch.sigmoid(out) + self.EPS).mean()

        return positive_loss + negative_loss
    

    def configure_optimizer(self, config: KohaLayerConfig):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': config.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and config.device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=config.learning_rate, betas= (config.beta1, config.beta2), **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer