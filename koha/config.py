from dataclasses import dataclass


@dataclass
class KohaBlockConfig:
    emb_dim: int = 16  # embedding dimension
    head_num: int = 4  # number of heads
    receptive_field = 5  # number of connections to other Koha blocks
    window_size: int = 5  # temporal window used for positve and negative sampling
    weight_decay: float = 1e-1
    learning_rate: float = 6e-4
    beta1: float = 0.9
    beta2: float = 0.95
    device_type: str = "cpu"


@dataclass
class KohaNetworkConfig:
    lr: float = 0.001  # signature learning rate
    vocab_size: int = 50256  # size of vocabulary
    context: int = 10
