import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.decoder import Decoder
from modules.encoder import Encoder
from config import config

class Transformer(nn.Module):
    def __init__(self,shared_pos : bool = False):
        super().__init__()
        self.pos_embedding = None
        self.token_embedding = nn.Embedding(config.vocab_size,config.n_embd)
        if shared_pos:
            self.pos_embedding = nn.Embedding(config.block_size,config.n_embd)
        self.encoder = Encoder(self.token_embedding, self.pos_embedding)
        self.decoder = Decoder(self.token_embedding, self.pos_embedding)
    def forward(self,src,mask,target):
        memory = self.encoder(src,mask)
        logits = self.decoder(target,mask,memory)
        return logits
        