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
    unit_num: int = 100 # number of units
    emb_dim: int = 100 # dimensionality of signature (output) and weights (input)
    receptive_field = 10 # the number of previous time steps from which a unit receives / sends inputs / outputs to.
    context_size : int = 50
    n_head: int = 12 # number of heads. Each unit receives inputs from each winner of each head within the attention radius
    neg_sampling_num: int = 20 # number of negative samples
    neg_unigram_scale: int = 10
    sparse = True # used for the embeddings of the signatures
