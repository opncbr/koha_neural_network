import torch
from torch.nn import Embedding
from math import sqrt
from .config import KohaNetworkConfig, KohaBlockConfig
from .koha_block import KohaBlock


class KohaNetwork(torch.nn.Module):
    def __init__(
        self, network_config: KohaNetworkConfig, block_config: KohaBlockConfig
    ):
        super().__init__()
        self.vocab_size = network_config.vocab_size
        self.emb_dim = network_config.emb_dim
        self.context = network_config.context
        self.receptive_field = block_config.receptive_field
        self.embeddings = Embedding(self.vocab_size, self.emb_dim, sparse=True)
        self.koha_blocks = [
            KohaBlock(block_config, True)
            if ind == 0
            else KohaBlock(block_config, False)
            for ind in range(network_config.context)
        ]
        self.network_state = None
        self.unfold = torch.nn.Unfold(
            kernel_size=(self.emb_dim, self.receptive_field),
            dilation=1,
            padding=0,
            stride=1,
        )
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.embeddings.weight, a=sqrt(5))

    def initialize_state(self, batch=1):
        self.network_state = torch.zeros(
            batch, self.emb_dim, self.context + self.receptive_field - 1
        )
        for block in self.koha_blocks:
            block.initialize_state(batch)

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
        y = []
        # losses = []
        for block_ind, block in enumerate(self.koha_blocks):
            x, z = X[block_ind], Z[block_ind]
            if block_ind > 0:
                x = x.detach()
            y = block(x, z)
            # losses.append(loss)
            self.network_state[:, :, block_ind] = y
        # XXX TODO: add logic to return block outputs for MLP blocks / other Koha networks
        # return losses
