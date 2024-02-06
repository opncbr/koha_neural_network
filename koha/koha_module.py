import torch
from .config import KohaModuleConfig
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from math import sqrt


class LayerNorm(torch.nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, emb_dim):
        super().__init__()
        self.weight = Parameter(torch.ones(emb_dim))

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, None, 1e-5)


class QReceiver(torch.nn.Module):
    def __init__(self, config: KohaModuleConfig):
        super().__init__()
        self.head_size = config.emb_dim // config.head_num
        assert config.emb_dim % config.head_num == 0
        self.R = Parameter(
            torch.empty((config.block_num, config.emb_dim, config.emb_dim))
        )
        self.W = Parameter(
            torch.empty(
                (config.block_num, config.emb_dim, config.head_num, self.head_size)
            )
        )
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.R)
        torch.nn.init.xavier_uniform_(self.W)

    def forward(self, x):
        r = torch.einsum("kbe, kei -> bki", x, self.R)
        q = torch.einsum("bke, kehn -> bkhn", r, self.W)
        return q


class KVReceiver(torch.nn.Module):
    def __init__(self, config: KohaModuleConfig):
        super().__init__()
        self.head_size = config.emb_dim // config.head_num
        self.receptive_field = config.receptive_field + 1
        self.R = Parameter(
            torch.empty(
                (config.block_num, self.receptive_field, config.emb_dim, config.emb_dim)
            )
        )
        self.W = Parameter(
            torch.empty(
                (
                    config.block_num,
                    self.receptive_field,
                    config.emb_dim,
                    config.head_num,
                    self.head_size,
                )
            )
        )
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.R)
        torch.nn.init.xavier_uniform_(self.W)

    def forward(self, z):
        r = torch.einsum("kbre, krei -> bkri", z, self.R)
        kv = torch.einsum("bkre, krehn -> bkhrn", r, self.W)
        return kv


class KohaModule(torch.nn.Module):
    def __init__(self, config: KohaModuleConfig):
        super().__init__()
        self.emb_dim = config.emb_dim
        self.head_size = config.emb_dim // config.head_num
        self.block_num = config.block_num
        self.q = QReceiver(config)
        self.k = KVReceiver(config)
        self.v = KVReceiver(config)
        self.W_o = Parameter(
            torch.empty((config.block_num, self.emb_dim, self.emb_dim))
        )
        self.ln = LayerNorm([self.block_num, self.emb_dim])
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.W_o)

    def _attention(self, q, k, v, mask):
        att = torch.einsum("bkhn, bkhrn -> bhkr", q, k) * (1.0 / sqrt(self.head_size))
        att = att.masked_fill(mask == False, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = torch.nan_to_num(att)  # in case all values were -inf
        return torch.einsum("bhkr, bkhrn -> bkhn", att, v)

    def forward(self, x, z, mask):
        Q = self.q(x)
        neg_Q = self.q(-x)
        K = self.k(z)
        V = self.v(z)

        y_pos = self._attention(Q, K, V, mask).reshape(-1, self.block_num, self.emb_dim)
        y_neg = self._attention(neg_Q, K, V, mask).reshape(
            -1, self.block_num, self.emb_dim
        )
        y_pos = torch.einsum("bke, kei -> bki", y_pos, self.W_o)
        y_pos = self.ln(y_pos)
        y_neg = torch.einsum("bke, kei -> bki", y_neg, self.W_o)
        y_neg = self.ln(y_neg)
        return y_pos, y_neg, y_pos.detach()
