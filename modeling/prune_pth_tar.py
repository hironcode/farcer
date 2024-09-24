"""
Modified: https://github.com/kohjingyu/fromage/blob/main/fromage/prune_model_ckpt.py

Prune pretrained model weights to reduce size.

This keeps only the weights that we finetune, and discards the pretrained LLM / visual encoder weights.
"""

import torch
import json
from collections import OrderedDict

try:
    from src.config import PathHelper
except:
    from ..src.config import PathHelper

ph = PathHelper()

ckpt_path = ph.get_target_dir('farser/best_wo_eos.pth.tar')
pruned_output_path = ph.get_target_dir('farser/best_wo_eos_pruned.pth.tar')


def prune_model_ckpt(ckpt_path, pruned_output_path):
    with open(ckpt_path, 'rb') as f:
        checkpoint = torch.load(f, map_location='cuda:0')
    stripped_state_dict = {
        k: v for k, v in checkpoint['model_state_dict'].items() if 
        'lm.' not in k and 'vm.' not in k and "input_embeddings." not in k
    }
    print(f"keys: {stripped_state_dict.keys()}")
    stripped_state_dict = OrderedDict(sorted(stripped_state_dict.items()))

    del checkpoint['epoch']
    del checkpoint['optimizer_state_dict']
    del checkpoint['train_loss']
    for k, v in stripped_state_dict.items():
        stripped_state_dict[k] = v.detach().clone()

    with open(pruned_output_path, 'wb') as f:
        torch.save({'model_state_dict': stripped_state_dict}, f)
