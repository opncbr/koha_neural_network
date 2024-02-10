import torch

from .config import KohaConfig
from .koha_module import KohaModule
from .koha_state import KohaState
import inspect


class KohaLayer(torch.nn.Module):
    def __init__(self, config: KohaConfig):
        super().__init__()
        self.koha_state = KohaState(config)
        self.koha_module = KohaModule(config)
        self.layer_optimizer = self.configure_optimizer(config)

    def forward(self, x):
        X, Z, M = self.koha_state(x)
        pos_outputs, neg_outputs = self.koha_module(X, Z, M)
        self.koha_state.update_state(pos_outputs)
        return pos_outputs, neg_outputs

    def initialize_state(self, batch_size):
        self.koha_state.initialize_state(batch_size)

    def configure_optimizer(self, config: KohaConfig):
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
