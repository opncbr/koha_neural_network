import torch
from torch.nn import Embedding
from math import sqrt
from .config import KohaNetworkConfig, KohaBlockConfig
from .koha_block import KohaBlock
from .helpers import getenv
import inspect

DEBUG = getenv("DEBUG", 0)


class KohaNetwork(torch.nn.Module):
    def __init__(
        self, network_config: KohaNetworkConfig, block_config: KohaBlockConfig
    ):
        super().__init__()
        self.vocab_size = network_config.vocab_size
        self.emb_dim = block_config.emb_dim
        self.context = network_config.context
        self.receptive_field = block_config.receptive_field
        self.embeddings = Embedding(self.vocab_size, self.emb_dim)
        self.koha_blocks = torch.nn.ModuleList(
            [
                KohaBlock(block_config, True)
                if ind == 0
                else KohaBlock(block_config, False)
                for ind in range(network_config.context)
            ]
        )
        self.network_state = None
        self.unfold = torch.nn.Unfold(
            kernel_size=(self.emb_dim, self.receptive_field),
            dilation=1,
            padding=0,
            stride=1,
        )
        self.EPS = 1e-15
        self.mask_int = 1
        self.layer_optimizer = self.configure_optimizer(KohaBlockConfig)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.embeddings.weight, a=sqrt(5))

    def initialize_state(self, batch=1):
        self.mask_int = 1
        self.network_state = torch.zeros(
            batch, self.emb_dim, self.context + self.receptive_field - 1
        )

    def _mask(self):
        extended_context = self.context + self.receptive_field
        remainder = extended_context - self.mask_int
        mask = (
            torch.cat(
                [
                    torch.flip(
                        torch.tril(torch.ones(self.mask_int, self.receptive_field + 1)),
                        dims=[0],
                    ),
                    torch.zeros(remainder, self.receptive_field + 1),
                ]
            )
            .detach()
            .to(torch.bool)
        )
        mask[:, 0] = True
        return mask

    def _increment_mask(self):
        if self.mask_int <= self.context + self.receptive_field - 1:
            self.mask_int += 1

    def forward(self, input_indices):
        batch = input_indices.size(0)
        assert (
            self.network_state is not None
        ), "initialize_state must be called before using forward"
        input = self.embeddings(input_indices).squeeze(1)
        X = torch.cat(
            [
                input.unsqueeze(0),
                self.network_state.permute(2, 0, 1)[: self.context - 1, :, :],
            ]
        )
        Z = (
            self.unfold(self.network_state.unsqueeze(1))
            .permute(2, 0, 1)
            .view(-1, batch, self.emb_dim, self.receptive_field)
            .transpose(-1, -2)
        )
        # add X to Z. Needed for keys
        Z = torch.cat([X.unsqueeze(-2), Z], dim=-2)

        mask = self._mask()
        self._increment_mask()

        positive_pairs = []
        negative_pairs = []
        for block_ind, block in enumerate(self.koha_blocks):
            x, z = X[block_ind], Z[block_ind]
            m = mask[block_ind].view(1, 1, self.receptive_field + 1)
            if block_ind > 0:
                x = x.detach()
                z = z.detach()

            y_pos, y_neg, y_pos_nograd = block(x, z, m)
            self.network_state[:, :, block_ind] = y_pos_nograd
            positive_pairs.append(y_pos.unsqueeze(1))
            negative_pairs.append(y_neg.unsqueeze(1))
        positive_pairs = torch.cat(positive_pairs, dim=1)
        negative_pairs = torch.cat(negative_pairs, dim=1)

        # compute positive & negative scores
        positive_scores = (
            (positive_pairs @ positive_pairs.transpose(-1, -2)).sum(dim=-1).view(-1)
        )
        negative_scores = (
            (positive_pairs @ negative_pairs.transpose(-1, -2)).sum(dim=-1).view(-1)
        )

        # compute positive & negative loss
        positive_loss = -torch.log(torch.sigmoid(positive_scores) + self.EPS).mean()
        negative_loss = -torch.log(1 - torch.sigmoid(negative_scores) + self.EPS).mean()
        loss = positive_loss + negative_loss

        # weight udpate
        self.layer_optimizer.zero_grad()
        loss.backward()
        self.layer_optimizer.step()

        return self.network_state.flatten()

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
