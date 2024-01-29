import torch
from .config import KohaBlockConfig
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from math import sqrt


class KohaBlock(torch.nn.Module):
    def __init__(self, config: KohaBlockConfig, first_layer: bool):
        super().__init__()
        self.first_layer = first_layer
        self.emb_dim = config.emb_dim
        self.head_num = config.head_num
        self.head_size = config.emb_dim // config.head_num
        assert self.emb_dim % self.head_num == 0
        self.receptive_field = config.receptive_field + 1

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

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.R_q, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.R_k, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.R_v, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.W_q, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.W_k, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.W_v, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.W_o, mean=0.0, std=0.02)

    def forward_pass(self, x, z, mask, pos: bool):
        batch = x.size(0)

        Q = torch.einsum("be, hne -> bhn", x, self.R_q) * (1.0 / sqrt(self.head_size))
        Q = F.softmax(Q, dim=-1) if pos else F.softmax(-Q, dim=-1)
        Q = torch.einsum("bhn, hnm -> bhm", Q, self.W_q)
        # Q: Shape (batch, head_num, head_size)

        K = torch.einsum("bre, hrne -> bhrn", z, self.R_k) * (
            1.0 / sqrt(self.head_size)
        )
        K = F.softmax(K, dim=-1)
        K = torch.einsum("bhrn, hrnm -> bhrm", K, self.W_k)
        # K: Shape (batch, head_num, receptive_field, head_size)

        V = torch.einsum("bre, hrne -> bhrn", z, self.R_v) * (
            1.0 / sqrt(self.head_size)
        )
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
        return y_pos, y_neg, y_pos.detach()
