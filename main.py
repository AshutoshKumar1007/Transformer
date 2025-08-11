"""
You don’t want your model to spend capacity learning how to predict padding.
we use padding mask so that the loss isn't computed on <pad> correctly.
Model isn't penalized for not predicting <pad> correctly.
#! general convention : padding is done at the end of seq -- after <eos> token.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config
from modules.transformer import Transformer
from utils.train import train
from utils.data import SPPDataset, collate_fn
from utils.train_Spm import train_sentencepiece
from torch.utils.data import DataLoader
import sentencepiece as spm
import os

if __name__ == "__main__":
    sp = spm.SentencePieceProcessor()
    # Load the SentencePiece model else train the spm model
    if not os.path.exists('spm_joint_32k.model'):
        print("Training SentencePiece model...")
        train_sentencepiece()
    print("Loading SentencePiece model...")
    sp.Load('spm_joint_32k.model')
    
    config.pad_id = sp.pad_id()
    config.vocab_size = sp.vocab_size()
    print("sp.vocab_size():", sp.vocab_size())
    dataset = SPPDataset("DATA/train.en","DATA/train.de",sp)
    trainloder = DataLoader(dataset,config.batch_size,shuffle=True,collate_fn=collate_fn)
    model = Transformer()
    model.to(config.device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {params}")
    criterion = nn.CrossEntropyLoss(ignore_index=sp.pad_id()) #? to ignore panality on padding
    optim = torch.optim.AdamW(model.parameters(),lr = config.lr,betas=(0.9, 0.98))
    #scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.95)
    print("---------------------------------------------------")
    print("Training starting...\n")
    lossi,avg_lossi = train(model,
                            trainloder,
                            optim,
                            scheduler,
                            criterion)
    
    
    


