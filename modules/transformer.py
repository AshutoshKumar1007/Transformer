import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config import config
from .decoder import Decoder
from .encoder import Encoder


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    def forward(self,x,target):
        memory = self.encoder(x)
        logits = self.decoder(target,memory)
        return logits
        