import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..config import config

class MultiSelfAttention(nn.Module):
    def __init__(self,drouout : float= None,casual : bool = False):
        super().__init__()
        assert config.n_embd % config.n_head == 0, f"assertion error number of heads must be a factor of embedding dim."
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.casual = casual
        if self.casual:
            self.register_buffer(
                'casual_mask',
                torch.tril(torch.ones(config.block_size,config.block_size)
                .unsqueeze(0).unsqueeze(1))
            ) # shape (1,1,T,T)
        #! key,query & value projection for all heads
        self.c_attn = nn.Linear(self.n_embd,3*self.n_embd)
        # no-masking "mask only the <pad> tokens!.."
        #output projection
        self.c_proj = nn.Linear(config.n_embd,config.n_embd) 
        if drouout is not None:
            self.dropout = nn.Dropout(p = drouout,inplace=True)
        #TODO
    def forward(self,x,pad_mask = None):
        """ 
        x : torch.Tensor shape #(B,T,C)
        """
        B,T,C = x.shape #T->context len and C is emd dim
        
        qkv = self.c_attn(x) # shape (B,T,3*C)
        # nh -> n_heads , hs -> head_size => C = nh*hs
        hs = C//self.n_head
        q,k,v = torch.chunk(qkv,chunks=3,dim=2) #shape (B,T,C) each
        #TODO helper function
        def _reshape(tensor):
            return tensor.view(B,T,self.n_head,hs).transpose(1,2)
        q,k,v = map(_reshape,q,k,v) #shape (B,nh,T,hs) each
        #attention -> populate the TXT matrix
        att = (q@k.transpose(-2,-1))*(1.0/math.sqrt(k.size(-1)))
        #! apply masking if passed 
        if pad_mask is not None:
            #TODO Stop the <pad> keys and queries to take part in attention.
            mask_keys = pad_mask.unsqeeze(1).unsqeeze(2)  #mask keys: shape → [B, 1, 1, T]
            mask_query = pad_mask.unsqueeze(1).unsqueeze(3) #mask keys: shape → [B, 1, T, 1]
            combined_mask = mask_keys & mask_query # shape [B, 1, T, T]
            att = att.masked_fill(combined_mask==0, float('-inf'))
            
        #! Apply casual masking if required 
        if self.casual:
            att = att.masked_fill(self.casual_mask[:,:,:T,:T] == 0,float('-inf'))          
        att = F.softmax(att,dim=-1)
        if self.dropout is not None:
            att = self.dropout(att)
        y = att @ v  #(B,nh,T,T) @ (B,nh,T,hs) ---> (B,nh,T,hs)
        y = y.transpose(1,2).contiguous().view(B,T,C)
        out = self.c_proj(y)
        return out

#-----------------------------------------------------------------------------

class MultiCrossAttention(nn.Module):
    def __init__(self,dropout : float = None):
        super().__init__()
        assert config.n_embd % config.n_head == 0, f"assertion error number of heads must be a factor of embedding dim."
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        #! key,query & value projection for all heads
        self.cross_q = nn.Linear(self.n_embd,self.n_embd)
        self.cross_kv = nn.Linear(self.n_embd,2*self.n_embd)
        # no-masking "mask only the <pad> tokens!.."
        #output projection
        self.c_proj = nn.Linear(config.n_embd,config.n_embd)
        # self.register_buffer("bias",torch.tril(torch.ones(config.block_size,config.block_size)).view(1,1,config.block_size,config.block_size)) # not really bias --> in our reference its the tril
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout,inplace=True)
    def forward(self,x,memory,pad_mask = None):
        """_summary_

        Args:
            x (_type_): _description_
            memory (_type_): _description_
        """
        B,T,C = x.shape
        hs = C//self.n_head
        q = self.cross_q(x) #shape (B,T,C)
        kv = self.cross_kv(memory)
        k,v = torch.chunk(kv,chunks=2,dim=2) #shape(B,T,C)
        def _reshape(tensor):
            return tensor.view(B,T,self.n_head,hs).transpose(1,2)
        q,k,v = map(_reshape,q,k,v)
        att = (q @ k.transpose(-2,-1))*(1.0/math.sqrt(k.size(-1)))
        if pad_mask is not None:
            #! only key padding.... (depends on the memory)
            att = att.masked_fill(pad_mask.unsqeeze(1).unsqeeze(2) == 0,float('-inf'))
        att = F.softmax(att,dim=-1)
        if self.dropout is not None:
            att = self.dropout(att)
        y = att @ v #(B,nh,T,T) @ (B,nh,T,hs) ---> (B,nh,T,hs)
        y = y.transpose(1,2).contiguous().view(B,T,C)
        #apply projection
        y = self.c_proj(y)
        
        
        
        
        
        
        
    