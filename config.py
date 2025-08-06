from dataclasses import dataclass
@dataclass
class config:
    block_size : int = 512
    eng_vocab_size : int = None
    spn_vocab_size : int = None
    n_layer : int = None
    n_embd  : int = None
    device : str = None