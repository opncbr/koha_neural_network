import torch
from .config import KohaConfig


class KohaState(torch.nn.Module):
    def __init__(self, config: KohaConfig):
        super().__init__()
        self.device = config.device_type
        self.emb_dim = config.emb_dim
        self.block_num = config.block_num
        self.receptive_field = config.receptive_field
        self.mask_int = 1
        self.unfold = torch.nn.Unfold(
            kernel_size=(self.emb_dim, self.receptive_field),
            dilation=1,
            padding=0,
            stride=1,
        )
        self.state = None

    def initialize_state(self, batch=1):
        self.mask_int = 1
        self.state = torch.zeros(
            batch, self.emb_dim, self.block_num + self.receptive_field - 1
        )

    def get_mask(self):
        extended_context = self.block_num + self.receptive_field
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
        mask = mask[: self.block_num, :]
        return mask.view(1, 1, self.block_num, self.receptive_field + 1)

    def get_state(self):
        self.state[:, :, : self.block_num].transpose(-1, -2)

    def forward(self, x):
        batch = x.size(0)
        assert (
            self.state is not None
        ), "initialize_state must be called before using forward"
        X = torch.cat(
            [
                x.unsqueeze(0),
                self.state.permute(2, 0, 1)[: self.block_num - 1, :, :],
            ]
        )
        Z = (
            self.unfold(self.state.unsqueeze(1))
            .permute(2, 0, 1)
            .view(-1, batch, self.emb_dim, self.receptive_field)
            .transpose(-1, -2)
        )
        # add X to Z. Needed for keys
        Z = torch.cat([X.unsqueeze(-2), Z], dim=-2)
        M = self.get_mask()
        return X, Z, M

    def update_state(self, new_state):
        # update koha state
        new_state = new_state.detach()
        new_state = new_state.transpose(-1, -2)
        self.state[:, :, : self.block_num] = new_state
        # update mask transition
        if self.mask_int < self.block_num + self.receptive_field:
            self.mask_int += 1
