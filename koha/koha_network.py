import torch
from torch.nn import Embedding
from math import sqrt
from .config import KohaNetworkConfig, KohaConfig
from .koha_module import KohaModule
from .koha_layer import KohaLayer
from .helpers import getenv
import inspect

DEBUG = getenv("DEBUG", 0)


class KohaNetwork(torch.nn.Module):
    def __init__(self, network_config: KohaNetworkConfig, koha_config: KohaConfig):
        super().__init__()
        self.embeddings = Embedding(network_config.vocab_size, koha_config.emb_dim)
        self.koha_layer = KohaLayer(koha_config)

        self.EPS = 1e-15
        self.layer_optimizer = self.configure_optimizer(koha_config)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.embeddings.weight, a=sqrt(5))

    def forward(self, input_indices):
        batch = input_indices.size(0)
        x = self.embeddings(input_indices).squeeze(1)
        pos_outputs, neg_outputs = self.koha_layer(x)

        # compute positive & negative scores
        positive_scores = (pos_outputs @ pos_outputs.transpose(-1, -2)).view(-1)
        negative_scores = (pos_outputs @ neg_outputs.transpose(-1, -2)).view(-1)

        # compute positive & negative loss
        positive_loss = -torch.log(torch.sigmoid(positive_scores) + self.EPS).mean()
        negative_loss = -torch.log(1 - torch.sigmoid(negative_scores) + self.EPS).mean()
        loss = positive_loss + negative_loss

        # weight update
        self.layer_optimizer.zero_grad()
        loss.backward()
        self.layer_optimizer.step()
        return loss, self.koha_layer.koha_state.get_state()

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
