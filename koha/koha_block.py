import torch
from .config import KohaBlockConfig
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from math import sqrt
from .helpers import getenv
import inspect

DEBUG = getenv("DEBUG", 0)


class Sampler:
    def __init__(self, config: KohaBlockConfig):
        self.config = config

    def sample_transition(self, pos: torch.Tensor, neg: torch.Tensor):
        self.pos_past[:, self.idx, :] = pos.detach()
        self.neg_past[:, self.idx, :] = neg.detach()
        self.idx = (self.idx + 1) % self.pos_past.size(1)

    def get_positive_scores(self, y):
        return (y @ self.pos_past.transpose(-1, -2)).sum(dim=-1).view(-1)

    def get_negative_scores(self, y):
        return (y @ self.neg_past.transpose(-1, -2)).sum(dim=-1).view(-1)

    def initialize_state(self, batch):
        self.idx = 0
        self.pos_past = torch.zeros(
            batch, self.config.window_size, self.config.emb_dim, requires_grad=False
        )
        self.neg_past = torch.zeros(
            batch, self.config.window_size, self.config.emb_dim, requires_grad=False
        )


class KohaBlock(torch.nn.Module):
    def __init__(self, config: KohaBlockConfig, first_layer: bool):
        super().__init__()
        self.first_layer = first_layer
        self.emb_dim = config.emb_dim
        self.head_num = config.head_num
        self.head_size = config.emb_dim // config.head_num
        assert self.emb_dim % self.head_num == 0
        self.receptive_field = config.receptive_field + 1
        self.EPS = 1e-15

        self.R_q = Parameter(
            torch.empty((self.head_num, self.head_size, self.emb_dim))
        )  # Shape (head_num, head_size, emb_dim)
        self.W_q = Parameter(
            torch.empty((self.head_num, self.head_size, self.head_size))
        )  # Shape (head_num, head_size, head_size)
        self.R_k = Parameter(
            torch.empty(
                (self.head_num, self.receptive_field, self.head_size, self.emb_dim)
            )
        )  # Shape (head_num, receptive_field, head_size, emb_dim)
        self.W_k = Parameter(
            torch.empty(
                (self.head_num, self.receptive_field, self.head_size, self.head_size)
            )
        )  # Shape (head_num, receptive_field, head_size, head_size)
        self.R_v = Parameter(
            torch.empty(
                (self.head_num, self.receptive_field, self.head_size, self.emb_dim)
            )
        )  # Shape (head_num, receptive_field, head_size, emb_dim)
        self.W_v = Parameter(
            torch.empty(
                (self.head_num, self.receptive_field, self.head_size, self.head_size)
            )
        )  # Shape (head_num, receptive_field, head_size, head_size)
        self.W_o = Parameter(
            torch.empty((self.emb_dim, self.emb_dim))
        )  # Shape (emb_dim, emb_dim)

        self.layer_optimizer = self.configure_optimizer(config)
        self.sampler = Sampler(config)
        self.apply(self._initialize_parameters)

    def _initialize_parameters(self, module):
        if isinstance(module, Parameter):
            torch.nn.init.normal_(module.data, mean=0.0, std=0.02)

    def forward_pass(self, x, z, mask, pos: bool):
        batch = x.size(0)

        Q = torch.einsum("be, hne -> bhn", x, self.R_q)
        Q = F.softmax(Q, dim=-1) if pos else F.softmax(-Q, dim=-1)
        Q = torch.einsum("bhn, hnm -> bhm", Q, self.W_q)
        # Q: Shape (batch, head_num, head_size)

        K = torch.einsum("bre, hrne -> bhrn", z, self.R_k)
        K = F.softmax(K, dim=-1)
        K = torch.einsum("bhrn, hrnm -> bhrm", K, self.W_k)
        # K: Shape (batch, head_num, receptive_field, head_size)

        V = torch.einsum("bre, hrne -> bhrn", z, self.R_v)
        V = F.softmax(V, dim=-1)
        V = torch.einsum("bhrn, hrnm -> bhrm", V, self.W_v)
        # V: Shape (batch, head_num, receptive_field, head_size)

        att = torch.einsum("bhn, bhrn -> bhr", Q, K) * (1.0 / sqrt(self.head_size))
        att = att.masked_fill(mask == False, float("-inf"))
        att = F.softmax(att, dim=-1)
        # att: Shape (batch, head_num, receptive_field)

        y = torch.einsum("bhr, bhrn -> bhn", att, V)
        y = y.reshape(batch, self.emb_dim)  # Re-assemble all head outputs side by side
        y = y @ self.W_o
        # y: Shape (batch, emb_dim)

        return y

    def forward(self, x, z, mask):
        y_pos = self.forward_pass(x, z, mask, True)
        y_neg = self.forward_pass(x, z, mask, False)

        # add positive and negative outputs to the Sampler
        self.sampler.sample_transition(y_pos, y_neg)

        # compute negative loss
        pos_out = self.sampler.get_positive_scores(y_pos)
        neg_out = self.sampler.get_negative_scores(y_pos)

        positive_loss = -torch.log(torch.sigmoid(pos_out) + self.EPS).mean()
        negative_loss = -torch.log(1 - torch.sigmoid(neg_out) + self.EPS).mean()
        loss = positive_loss + negative_loss
        # return positive output
        return loss, y_pos.detach()

    def configure_optimizer(self, config: KohaBlockConfig):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if DEBUG > 0:
            print(
                f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
            )
            print(
                f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
            )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and config.device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            **extra_args,
        )
        if DEBUG > 0:
            print(f"using fused AdamW: {use_fused}")

        return optimizer
