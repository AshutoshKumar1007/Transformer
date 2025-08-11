import torch
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
from utils import translate
from modules.transformer import Transformer
from config import config
# from config import config
import os
import torch
from torch.amp import autocast                 # keep autocast from cuda
from torch.amp import GradScaler
def train(model : nn.Module,
          train_loader : torch.utils.data.DataLoader,
          optim,
          scheduler = None,
          criterion = None,
          start_epoch : int = 1,
          scaler : GradScaler | None = None,
          grad_clip : float = 1.0,
          save_dir : str = config.save_dir
         ):
    """ 
    """
    lossi = []
    avg_lossi = []
    os.makedirs(save_dir,exist_ok = True)
    if scaler is None:
        scaler = GradScaler()
    for epoch in range(start_epoch,start_epoch + config.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_lossi = [] # for dynamic ploting 
        pbar = tqdm(train_loader,desc=f"Epoch {epoch}/{config.epochs} ")
        for batch in pbar:
            optim.zero_grad()
            src,mask,tgt = batch
            src,mask,tgt = src.to(config.device),mask.to(config.device),tgt.to(config.device)
 
            with autocast('cuda'):
                logits = model(src,mask,tgt)
                loss = criterion(logits.permute(0, 2, 1),tgt)

            # scale -> backward -> unscale -> clip -> step -> update scaler
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optim)    # will skip step if overflow detected
            scaler.update() 
            if scheduler:
                scheduler.step()
            torch.cuda.empty_cache()
            lossi.append(loss.item())
            epoch_lossi.append(loss.item())
            epoch_loss += loss.item()
            
            pbar.set_postfix({'batch_loss ':  f"{loss.item():.4f}"})
            # Batch-wise plot update (static, but refreshed)
            clear_output(wait=True)
            plt.figure(figsize=(8,4))
            plt.plot(epoch_lossi, label=f'Epoch {epoch} Batch Loss')
            plt.xlabel('Batch')
            plt.ylabel('Loss')
            plt.title(f'Epoch {epoch} Batch Loss')
            plt.legend()
            display(plt.gcf())
            plt.close()
        avg_loss = epoch_loss / len(train_loader)        
        avg_lossi.append(avg_loss)
        print(f"Epoch {epoch} Complete: Avg Loss = {avg_loss:.4f}")
        if epoch % 1 == 0:
            ckpt_point = os.path.join(save_dir,f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                'model' :model.state_dict(),
                'optim' : optim.state_dict(),
                'scaler' : scaler.state_dict(),
                'scheduler' : scheduler.state_dict(),
                'epoch': epoch               
                        }, ckpt_point)
    return lossi,avg_lossi 
            