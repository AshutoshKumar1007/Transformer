import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config
from modules.attention import MultiSelfAttention,MultiCrossAttention
from modules.ff import feeedforward
class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.casualattention = MultiSelfAttention(drouout=0.2,casual=True)
        self.crossattention = MultiCrossAttention(dropout=0.2)
        self.ff = feeedforward()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ln3 = nn.LayerNorm(config.n_embd)
    def forward(self,x,src_pad,memory):
        #TODO incapsulate the padding 
        x = x + self.casualattention(self.ln1(x))
        x = x + self.crossattention(self.ln2(x),src_pad,memory)
        x = x + self.ff(self.ln3(x))
        return x,src_pad,memory


class Decoder(nn.Module):
    def __init__(self,token_embedding : nn.Embedding,pos_embedding : nn.Embedding):
        super().__init__()
        self.n_layer = config.n_layer
        self.token_embedding = token_embedding
        self.pos_embedding = pos_embedding
        if pos_embedding is None:
            self.pos_embedding = nn.Embedding(config.block_size,config.n_embd)
        self.blocks = nn.ModuleList([DecoderBlock() for _ in range(self.n_layer)])
        self.linear = nn.Linear(config.n_embd,config.vocab_size)
        self.ln_f = nn.LayerNorm(config.n_embd)
    def forward(self,tgt,src_pad,memory):
        B,T = tgt.shape
        pos_embd = self.pos_embedding(torch.arange(T,device=config.device))
        tok_embd = self.token_embedding(tgt)
        tgt = tok_embd + pos_embd
        for block in self.blocks:
            tgt,src_pad,memory = block(tgt,src_pad,memory)
        tgt = self.ln_f(tgt)
        out = self.linear(tgt)        
        return out
        
        
        
        
        