from dataclasses import dataclass

@dataclass
class KohaInputLayerConfig():
    lr: float = 0.01 # signature learning rate
    vocab_size: int = 50256 # size of vocabulary
    emb_dim: int = 100 # dimensionality of vocabulary embeddings
    window_size: int = 10 # temporal window to learn the signature embeddings
    neg_sampling_num: int = 20 # number of negative samples
    neg_unigram_scale = 10 # used for negative sampling
    sparse = True # used for the embeddings of the signatures

@dataclass
class KohaLayerConfig():
    emb_dim: int = 100 # embedding dimension
    head_num: int = 12 # number of heads
    receptive_field = 10 # number of connections that a Koha block has to other Koha blocks
    window_size: int = 10 # temporal window used for positve and negative sampling
    weight_decay: float = 1e-1
    learning_rate: float = 6e-4
    beta1: float = 0.9
    beta2: float = 0.95
    device_type: str = "cpu"