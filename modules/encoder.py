import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config import config
from .attention import MultiSelfAttention
from .ff import feeedforward
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = MultiSelfAttention(drouout=0.2)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ff = feeedforward()
        self.ln2 = nn.LayerNorm(config.n_embd)
    def forward(self,x): 
        x = x + self.attention(self.ln1(x))
        x = x + self.ff(self.ln1(x))
        return x

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_layer = config.n_layer
        self.blocks = nn.Sequential(*[Block() for _ in range(self.n_layer)])
        self.token_embedding = nn.Embedding(config.eng_vocab_size,config.n_embd)
        self.pos_embedding = nn.Embedding(config.block_size,config.n_embd)
    
    def forward(self,x):
        B,T = x.shape
        pos_embd = self.pos_embedding(torch.arange(T,device=config.device))
        token_embd = self.token_embedding(x)
        x = token_embd + pos_embd
        out = self.blocks(x)
        return out
    
        
        
    
        
        
        