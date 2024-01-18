import torch
from .config import KohaBlockConfig
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from math import sqrt
import inspect


class State:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.pos_past: torch.Tensor | None = None
        self.neg_past: torch.Tensor | None = None

    def state_transition(self, pos: torch.Tensor, neg: torch.Tensor):
        self.pos_past = torch.cat(
            [self.pos_past, pos] if self.pos_past is not None else [pos], dim=-2
        )
        self.neg_past = torch.cat(
            [self.neg_past, neg] if self.neg_past is not None else [neg], dim=-2
        )

        if self.neg_past.size(-2) > self.window_size:
            self.pos_past = self.pos_past.narrow(-2, 1, self.window_size)
            self.neg_past = self.neg_past.narrow(-2, 1, self.window_size)

    def get_positive_samples(self):
        return self.pos_past @ self.pos_past.transpose(-1, -2)

    def get_negative_samples(self):
        return self.pos_past @ self.neg_past.transpose(-1, -2)


class KohaBlock(torch.nn.Module):
    def __init__(self, config: KohaBlockConfig, first_layer: bool):
        super().__init__()
        self.first_layer = first_layer
        self.emb_dim = config.emb_dim
        self.head_num = config.head_num
        self.head_size = config.emb_dim // config.head_num
        self.receptive_field = config.receptive_field
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
        self.state = State(config.window_size)
        self.apply(self._initialize_parameters)

    def _initialize_parameters(self, module):
        if isinstance(module, Parameter):
            torch.nn.init.normal_(module.data, mean=0.0, std=0.02)

    # incomplete forward
    def forward(self, x, z):
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
        # Shape (batch, head_num, receptive_field)
        att_neg = F.softmax(att_neg, dim=-1)
        # Shape (batch, head_num, head_size)
        y_neg = torch.einsum("bhr, hrn -> bhn", att_neg, self.att_values)
        # Re-assemble all head outputs side by side. Shape (batch, emb_dim)
        y_neg = y_neg.reshape(batch, self.emb_dim)

        # add positive and negative outputs to the Memory State
        self.state.state_transition(y_pos.unsqueeze(-2), y_pos.unsqueeze(-2))

        # perform backprop
        self.layer_optimizer.zero_grad()
        loss = self.loss()
        loss.backward()
        self.layer_optimizer.step()

        # return positive output
        return y_pos.detach()

    def loss(self):
        # compute positive loss
        out = self.state.get_positive_samples().sum(dim=-1).view(-1)
        positive_loss = -torch.log(torch.sigmoid(out) + self.EPS).mean()

        # compute negative loss
        out = self.state.get_negative_samples().sum(dim=-1).view(-1)
        negative_loss = -torch.log(1 - torch.sigmoid(out) + self.EPS).mean()

        return positive_loss + negative_loss

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
        print(f"using fused AdamW: {use_fused}")

        return optimizer
