import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config import config
from .attention import MultiSelfAttention,MultiCrossAttention
from .ff import feeedforward
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.casualattention = MultiSelfAttention(drouout=0.2,casual=True)
        self.crossattention = MultiCrossAttention(dropout=0.2)
        self.ff = feeedforward()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ln3 = nn.LayerNorm(config.n_embd)
    def forward(self,x,memory):
        #TODO incapsulate the padding 
        x = x + self.casualattention(self.ln1(x))
        x = x + self.crossattention(self.ln2(x),memory)
        x = x + self.ff(self.ln3(x))
        return x 


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_layer = config.n_layer
        self.token_embedding = nn.Embedding(config.spn_vocab_size,config.n_embd)
        self.pos_embedding = nn.Embedding(config.block_size,config.n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(self.n_layer)])
        self.linear = nn.Linear(config.n_embd,config.spn_vocab_size)
        self.ln_f = nn.LayerNorm(config.n_embd)
    def forward(self,x,memory):
        B,T = x.shape
        pos_embd = self.pos_embedding(torch.arange(T,device=config.device))
        tok_embd = self.token_embedding(x)
        x = tok_embd + pos_embd
        x = self.blocks(x,memory)
        x = self.ln_f(x)
        out = self.linear(x)        
        return out
        
        
        
        
        