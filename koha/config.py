from dataclasses import dataclass


@dataclass
class KohaModuleConfig:
    emb_dim: int = 256  # embedding dimension
    head_num: int = 8  # number of heads
    receptive_field = 5  # number of connections to other Koha blocks
    block_num = 10
    weight_decay: float = 1e-1
    learning_rate: float = 5e-4
    beta1: float = 0.9
    beta2: float = 0.95
    device_type: str = "cpu"


@dataclass
class KohaNetworkConfig:
    vocab_size: int = 50256  # size of vocabulary
