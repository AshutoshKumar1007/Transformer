import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config
class feeedforward(nn.Module):
    def __init__(self,dropout : float = 0.2,ff_scale : float = 4.0):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd,ff_scale*config.n_embd)
        self.fc2 = nn.Linear(ff_scale*config.n_embd,config.n_embd)
        self.act = nn.GELU(approximate='tanh')
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
         residual = x
         x = self.fc1(x)
         x = self.act(x)
         x = self.dropout(x)
         x = self.fc2(x)
         x = self.dropout(x)
         x = x + residual
         return x
         