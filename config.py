from dataclasses import dataclass
import torch
@dataclass
class config:
    #Data & Tokenization
    vocab_size : int = 32000
    block_size : int = 128
    src_vocab_size : int = None
    tgt_vocab_size : int = None
    pad_id : int = 3
    
    #Model acritecture (Transformer Base)
    n_layer : int = 6
    n_embd  : int = 512
    n_head : int = 8
    dropout : float = 0.2
    
    #Training parameters
    batch_size : int = 128
    lr : float = 5e-4
    label_smoothing : float = 0.1
    epochs : int = 50
    
    #Device
    device : str = 'cuda' if torch.cuda.is_available() else "cpu"
    fp16: bool = True  # mixed precision training
    #Paths
    save_dir : str = 'checkpoints'

    