import torch
from .config import KohaBlockConfig
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
    def __init__(self, config: KohaBlockConfig):
        super().__init__()
        self.emb_dim = config.emb_dim
        self.head_num = config.head_num
        self.head_size = self.emb_dim // self.head_num
        self.neg_sample_size = config.neg_sample_size
        assert self.emb_dim % self.head_num == 0
        assert self.neg_sample_size < self.head_size - 1
        self.R = Parameter(torch.empty((self.emb_dim, self.emb_dim * 4)))
        self.W = Parameter(torch.empty((self.emb_dim * 4, self.emb_dim)))
        self.ln = LayerNorm(self.head_size)
        self.gelu = torch.nn.GELU()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.R)
        torch.nn.init.xavier_uniform_(self.W)

    def forward(self, x):
        batch = x.size(0)
        r = x @ self.R
        r_pos = self.gelu(r)

        q_pos = r_pos @ self.W
        q_pos = q_pos.view(batch, -1, self.head_num, self.head_size)
        q_pos = self.ln(q_pos)
        return q_pos, r.detach()

    def negative_forward(self, r):
        # compute negative samples
        r_neg = F.softmax(-r, dim=-1)
        rand_neg_indices = torch.multinomial(
            r_neg, self.neg_sample_size, replacement=False
        )
        neg_inputs = self.R[:, rand_neg_indices].permute(1, 2, 0).detach()
        # negative forward pass
        out, _ = self.forward(neg_inputs)
        return out


class KVReceiver(torch.nn.Module):
    def __init__(self, config: KohaBlockConfig):
        super().__init__()
        self.emb_dim = config.emb_dim
        self.head_num = config.head_num
        self.head_size = self.emb_dim // self.head_num
        self.receptive_field = config.receptive_field + 1
        self.R = Parameter(
            torch.empty((self.receptive_field, self.emb_dim, self.emb_dim * 4))
        )
        self.W = Parameter(
            torch.empty((self.receptive_field, self.emb_dim * 4, self.emb_dim))
        )
        self.ln = LayerNorm(self.head_size)
        self.gelu = torch.nn.GELU()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.R)
        torch.nn.init.xavier_uniform_(self.W)

    def forward(self, z):
        r = torch.einsum("bre, rei -> bri", z, self.R)
        r = self.gelu(r)  # F.softmax(r, dim=-1)
        kv = torch.einsum("bri, rie -> bre", r, self.W)
        kv = kv.view(-1, self.receptive_field, self.head_num, self.head_size).transpose(
            1, 2
        )
        kv = self.ln(kv)
        return kv


class KohaBlock(torch.nn.Module):
    def __init__(self, config: KohaBlockConfig):
        super().__init__()
        self.emb_dim = config.emb_dim
        self.head_size = config.emb_dim // config.head_num
        self.q = QReceiver(config)
        self.k = KVReceiver(config)
        self.v = KVReceiver(config)
        self.W_o = Parameter(torch.empty((self.emb_dim, self.emb_dim)))
        self.ln = LayerNorm(self.emb_dim)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.W_o)

    def _attention(self, q, k, v, mask):
        att = torch.einsum("bshn, bhrn -> bshr", q, k) * (1.0 / sqrt(self.head_size))
        att = att.masked_fill(mask == False, float("-inf"))
        att = F.softmax(att, dim=-1)
        return torch.einsum("bshr, bhrn -> bshn", att, v)

    def forward(self, x, z, mask):
        batch = x.size(0)
        Q, neg_inputs = self.q(x)
        K = self.k(z)
        V = self.v(z)
        neg_Q = self.q.negative_forward(neg_inputs)

        y_pos = self._attention(Q, K, V, mask).reshape(batch, self.emb_dim)
        y_neg = self._attention(neg_Q, K, V, mask).reshape(batch, -1, self.emb_dim)
        y_pos = self.ln(y_pos @ self.W_o)
        y_neg = self.ln(y_neg @ self.W_o)
        return y_pos, y_neg, y_pos.detach()
