import torch
import torch.nn as nn
from config import config
def translate(model,sp,sentence,max_len = config.block_size,device = config.device):
    src_ids = sp.EncodeAsIds(sentence)[:max_len - 2]
    src_ids = [sp.bos_id()] + src_ids + [sp.eos_id()]
    src = torch.tensor(src_ids,dtype = torch.long)
    src_tokens = torch.cat([src,torch.full((config.block_size - len(src),),sp.pad_id(),dtype = torch.long)]).unsqueeze(0).to(device)
    src_mask = (src_tokens != sp.pad_id()).long()

    #ENcode the source
    memory = model.encoder(src_tokens,src_mask)
    #decode 
    tgt_ids = [sp.bos_id()]
    for i in range(1,max_len - 2):
        tgt = torch.tensor(tgt_ids,dtype = torch.long)
        tgt_tokens = torch.cat([tgt,torch.full((config.block_size - len(tgt),),sp.pad_id(),dtype = torch.long)]).unsqueeze(0).to(device)
        logits = model.decoder(tgt_tokens,src_mask,memory)
        next_token = logits[:,i,:].argmax(dim = -1).item()
        tgt_ids.append(next_token)
        if next_token == sp.eos_id():
            break
    output_ids = tgt_ids[1:-1] if tgt_ids[-1] == sp.eos_id() else tgt_ids[1:]
    return sp.DecodeIds(output_ids)