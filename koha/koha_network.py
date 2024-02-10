import torch
from torch.nn import Embedding
from math import sqrt
from .config import KohaConfig
from .koha_layer import KohaLayer
from .koha_module import LayerNorm
from torch.nn import functional as F
from .helpers import getenv
import inspect

DEBUG = getenv("DEBUG", 0)


class MLP(torch.nn.Module):
    def __init__(self, config: KohaConfig):
        super().__init__()
        self.c_fc = torch.nn.Linear(config.emb_dim, 4 * config.emb_dim, bias=False)
        self.gelu = torch.nn.GELU()
        self.c_proj = torch.nn.Linear(4 * config.emb_dim, config.emb_dim, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class KohaNetwork(torch.nn.Module):
    def __init__(self, vocab_size, koha_config: KohaConfig):
        super().__init__()
        self.embeddings = Embedding(vocab_size, koha_config.emb_dim)
        self.koha_layer = KohaLayer(koha_config)
        self.mlp = MLP(koha_config)
        self.lm_head = torch.nn.Linear(koha_config.emb_dim, vocab_size, bias=False)
        # weights tying
        self.embeddings.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying
        self.ln = LayerNorm(koha_config.emb_dim)
        self.EPS = 1e-15
        self.optimizer = self.configure_optimizer(koha_config)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.embeddings.weight, a=sqrt(5))

    def forward(self, input_indices, targets=None):
        x = self.embeddings(input_indices).squeeze(1)
        pos_outputs, neg_outputs = self.koha_layer(x)

        # compute positive & negative scores
        positive_scores = (pos_outputs @ pos_outputs.transpose(-1, -2)).view(-1)
        negative_scores = (pos_outputs @ neg_outputs.transpose(-1, -2)).view(-1)

        # compute positive & negative loss
        positive_loss = -torch.log(torch.sigmoid(positive_scores) + self.EPS).mean()
        negative_loss = -torch.log(1 - torch.sigmoid(negative_scores) + self.EPS).mean()
        koha_loss = positive_loss + negative_loss

        # koha layer weight update
        self.optimizer.zero_grad()
        koha_loss.backward()
        self.optimizer.step()

        # predict the next token
        first_koha_block = self.koha_layer.koha_state.get_state()[:, 0, :]
        output = self.mlp(first_koha_block)
        output = self.ln(output)
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(output)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            logits = self.lm_head(output)
            loss = None

        return logits, loss, koha_loss

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
