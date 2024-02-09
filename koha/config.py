from dataclasses import dataclass


@dataclass
class KohaConfig:
    emb_dim: int = 256
    head_num: int = 8
    receptive_field = 10
    block_num = 10
    weight_decay: float = 1e-1
    learning_rate: float = 5e-4
    beta1: float = 0.9
    beta2: float = 0.95
    device_type: str = "mps"
