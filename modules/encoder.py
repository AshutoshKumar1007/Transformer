import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config
from modules.attention import MultiSelfAttention
from modules.ff import feeedforward
class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = MultiSelfAttention(drouout=0.2)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ff = feeedforward()
        self.ln2 = nn.LayerNorm(config.n_embd)
    def forward(self,x,src_pad): 
        x = x + self.attention(self.ln1(x),src_pad)
        x = x + self.ff(self.ln1(x))
        return x,src_pad

class Encoder(nn.Module):
    def __init__(self,tok_embedding : nn.Embedding,pos_embedding : nn.Embedding):
        super().__init__()
        self.n_layer = config.n_layer
        self.blocks = nn.ModuleList([EncoderBlock() for _ in range(self.n_layer)])
        self.token_embedding = tok_embedding
        self.pos_embedding = pos_embedding
        if pos_embedding is None:
            self.pos_embedding = nn.Embedding(config.block_size,config.n_embd)
         #TODO we need a final layer norm ?  
    def forward(self,x,pad_mask):
        B,T = x.shape
        pos_embd = self.pos_embedding(torch.arange(T,device=config.device))
        token_embd = self.token_embedding(x)
        x = token_embd + pos_embd
        for block in self.blocks:
            x,pad_mask = block(x,pad_mask)
        out = x
        return out
    
        
        
    
        
        
        