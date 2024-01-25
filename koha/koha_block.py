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
        self.idx = 0
        self.pos_past = torch.empty(
            16, config.window_size, config.emb_dim, requires_grad=False
        )
        self.neg_past = torch.empty(
            16, config.window_size, config.emb_dim, requires_grad=False
        )

    def sample_transition(self, pos: torch.Tensor, neg: torch.Tensor):
        self.pos_past[:, self.idx, :] = pos.detach()
        self.neg_past[:, self.idx, :] = neg.detach()
        self.idx = (self.idx + 1) % self.pos_past.size(1)

    def get_positive_samples(self):
        return self.pos_past

    def get_negative_samples(self):
        return self.neg_past

    # XXX: must be implemented
    def initialize_state(self, batch):
        return


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

        self.query_key = Parameter(
            torch.empty((self.head_num, self.head_size, self.emb_dim))
        )  # Shape (head_num, head_size, emb_dim)
        self.query_value = Parameter(
            torch.empty((self.head_num, self.head_size, self.head_size))
        )  # Shape (head_num, head_size, head_size)
        self.key_keys = Parameter(
            torch.empty(
                (self.head_num, self.receptive_field, self.head_size, self.emb_dim)
            )
        )  # Shape (head_num, receptive_field, head_size, emb_dim)
        self.key_values = Parameter(
            torch.empty(
                (self.head_num, self.receptive_field, self.head_size, self.head_size)
            )
        )  # Shape (head_num, receptive_field, head_size, head_size)
        self.att_values = Parameter(
            torch.empty((self.head_num, self.receptive_field, self.head_size))
        )  # Shape (head_num, receptive_field, head_size)

        self.layer_optimizer = self.configure_optimizer(config)
        self.sampler = Sampler(config)
        self.apply(self._initialize_parameters)

    def _initialize_parameters(self, module):
        if isinstance(module, Parameter):
            torch.nn.init.normal_(module.data, mean=0.0, std=0.02)

    # incomplete forward
    def forward(self, x, z, mask=None):
        batch = x.size(0)

        # compute positive and negative outputs

        # Shape (batch, head_num, head_size)
        q = torch.einsum("be, hne -> bhn", x, self.query_key)
        # Shape (batch, head_num, receptive_field, head_size)
        q_pos = F.softmax(q, dim=-1)
        if self.first_layer:
            # Shape (batch, head_num, receptive_field, head_size)
            q_neg = F.softmax(-q, dim=-1)
        else:
            with torch.no_grad():
                # Shape (batch, head_num, receptive_field, head_size)
                q_neg = F.softmax(-q, dim=-1)
        # n and m have the same dim
        q_pos = torch.einsum("bhn, hnm -> bhm", q_pos, self.query_value)
        # n and m have the same dim
        q_neg = torch.einsum("bhn, hnm -> bhm", q_neg, self.query_value)

        # Shape (batch, receptive_field, emb_dim). e and i have the same dim within the einsum
        k = torch.einsum("bre, hrne -> bhrn", z, self.key_keys)
        # Shape (batch, receptive_field, emb_dim)
        k_pos = F.softmax(k, dim=-1)
        if self.first_layer:
            k_neg = k_pos
        else:
            k_neg = k_pos.detach()
        # Shape (batch, receptive_field, emb_dim). e and i have the same dim within the einsum
        k_pos = torch.einsum("bhrn, hrnm -> bhrm", k_pos, self.key_values)
        # Shape (batch, receptive_field, emb_dim). e and i have the same dim within the einsum
        k_neg = torch.einsum("bhrn, hrnm -> bhrm", k_neg, self.key_values)

        # Shape (batch, head_num, receptive_field)
        att_pos = torch.einsum("bhn, bhrn -> bhr", q_pos, k_pos) * (
            1.0 / sqrt(self.head_size)
        )
        if mask is not None:
            att_pos = att_pos.masked_fill(mask == 0, float("-inf"))
        # Shape (batch, head_num, receptive_field)
        att_pos = F.softmax(att_pos, dim=-1)
        # Shape (batch, head_num, head_size)
        y_pos = torch.einsum("bhr, hrn -> bhn", att_pos, self.att_values)
        # Re-assemble all head outputs side by side. Shape (batch, emb_dim)
        y_pos = y_pos.reshape(batch, self.emb_dim)

        # Shape (batch, head_num, receptive_field)
        att_neg = torch.einsum("bhn, bhrn -> bhr", q_neg, k_neg) * (
            1.0 / sqrt(self.head_size)
        )
        if mask is not None:
            att_neg = att_neg.masked_fill(mask == 0, float("-inf"))
        # Shape (batch, head_num, receptive_field)
        att_neg = F.softmax(att_neg, dim=-1)
        # Shape (batch, head_num, head_size)
        y_neg = torch.einsum("bhr, hrn -> bhn", att_neg, self.att_values)
        # Re-assemble all head outputs side by side. Shape (batch, emb_dim)
        y_neg = y_neg.reshape(batch, self.emb_dim)

        # add positive and negative outputs to the Sampler
        self.sampler.sample_transition(y_pos, y_neg)
        out = (y_pos  @ self.sampler.get_positive_samples().transpose(-1,-2)).sum(dim=-1).view(-1)
        positive_loss = -torch.log(torch.sigmoid(out) + self.EPS).mean()
        # compute negative loss
        out = (y_pos @ self.sampler.get_negative_samples().transpose(-1,-2)).sum(dim=-1).view(-1)
        negative_loss = -torch.log(1 - torch.sigmoid(out) + self.EPS).mean()

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
