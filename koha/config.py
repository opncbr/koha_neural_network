from dataclasses import dataclass

@dataclass
class KohaInputLayerConfig():
    lr: float = 0.01 # signature learning rate
    vocab_size: int = 100 # size of vocabulary
    emb_dim: int = 100 # dimensionality of vocabulary embeddings
    window_size: int = 10 # temporal window to learn the signature embeddings
    neg_sampling_num: int = 20 # number of negative samples
    sample = 1e-3 # constant used to control positive sampling
    unit_filter_scale = 1000 # used for negative sampling

@dataclass
class KohaLayerConfig():
    unit_num: int = 100 # number of units
    emb_dim: int = 100 # dimensionality of signature (output) and weights (input)
    attention_radius = 10 # the number of layers (above & below) from which a unit receives / sends inputs/outputs to.
    n_head: int = 12 # number of heads. Each unit receives inputs from each winner of each head within the attention radius
    window_size: int = 10 # temporal window to learn the signature embeddings
    neg_sampling_num: int = 20 # number of negative samples
