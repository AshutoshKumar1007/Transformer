from torch.utils.data import Dataset
import torch
from config import config

def collate_fn(batch):
    """
    Args:
        batch : usually list of samples, where each samples is a tuple or dict
                returned from your dataset.__getitem__.
    Returns:
        A single "batch" , padded tensors.
    """
    pad_id = config.pad_id
    input_ids = []
    attention_mask = []
    labels = []
    
    for src,tgt in batch:
        #pad
        src = torch.cat([src,torch.full((config.block_size - len(src),),pad_id,dtype=torch.long)])
        tgt = torch.cat([tgt, torch.full((config.block_size - len(tgt),), pad_id, dtype=torch.long)])
        input_ids.append(src)
        labels.append(tgt)
        attention_mask.append((src != pad_id).long())
    input_ids = torch.stack(input_ids)
    attention_mask = torch.stack(attention_mask)
    labels = torch.stack(labels)
    return input_ids,attention_mask,labels
    
class SPPDataset(Dataset):
    def __init__(self,src_path : str = None, tgt_path : str = None ,sp_processor = None, max_len = 250):
        self.src = open(src_path,encoding="utf-8").read().splitlines()
        self.tgt = open(tgt_path,encoding="utf-8").read().splitlines()
        self.sp = sp_processor
        self.max_len = max_len
    def __len__(self):
        return len(self.src)
    def __getitem__(self, index):
        src_ids = self.sp.EncodeAsIds(self.src[index]) [:config.block_size - 2] #ig, we just take the whole line
        tgt_ids = self.sp.EncodeAsIds(self.tgt[index]) [:config.block_size - 2]
        
        #add BOS/EOS
        src_ids = [self.sp.bos_id()] + src_ids + [self.sp.eos_id()]
        tgt_ids = [self.sp.bos_id()] + tgt_ids + [self.sp.eos_id()]
        
        # Tensor wrapper 
        src_ids = torch.tensor(src_ids,dtype=torch.long)
        tgt_ids = torch.tensor(tgt_ids,dtype=torch.long)
        
        #dynamic masking (must be applied dynamically for each batch)
        return src_ids,tgt_ids
    


    
        
        
        
        
        