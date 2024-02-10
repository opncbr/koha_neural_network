from dataclasses import dataclass


@dataclass
class KohaConfig:
    emb_dim: int = 256
    head_num: int = 8
    receptive_field: int = 10
    block_num: int = 10
    mlp_scaling: int = 4
    weight_decay: float = 1e-1
    learning_rate: float = 5e-3
    beta1: float = 0.9
    beta2: float = 0.95
    device_type: str = "mps"
