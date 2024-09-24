from huggingface_hub import login
import os
import torch
import torch.nn as nn
import bitsandbytes as bnb
from datasets import load_dataset
import torch.optim as optim
from peft import LoraConfig, get_peft_model, PeftModel
import torch
from torch import optim
from tqdm import tqdm
import pandas as pd
from typing import Optional
import time
import datetime
import PIL
from wasabi import msg
from sklearn.model_selection import train_test_split
import json
import numpy as np
import gc
from sklearn.metrics import accuracy_score, f1_score
import random
from copy import deepcopy

try:
    from modeling.model import FarcerModel, ParamsFarcer
    from modeling import util
    from modeling.trainer_validator import train_batch_modality_projector, validate
    from modeling.train import train_epochs_modality_projector as train
    from src import features, plots
    from src.config import PathHelper
except:
    from model import FarcerModel, ParamsFarcer
    import util
    from trainer_validator import train_batch_modality_projector, validate
    from train import train_epochs_modality_projector as train
    from ..src import features, plots
    from ..src.config import PathHelper

ph = PathHelper()

### Environment setup ###
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"  # ignore bnb warnings
torch.manual_seed(0)


def hf_login(token):
    login(token=token)
    print("You are logged in.")

def load_lora_config(num_layers=1):
    # align with DialogueLLM LoRA config: https://arxiv.org/abs/2310.11374
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.1
    TARGET_MODULES = [
        "q_proj",
        "k_proj",
        "v_proj",
        # "o_proj",
        # "gate_proj",
        # "up_proj",
        # "down_proj"
    ]
    MODULES_TO_SAVE = [f"mp.{i}" for i in range(num_layers)]
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        modules_to_save=MODULES_TO_SAVE,
        # task_type="CAUSAL_LM",
        bias="none",
    )
    print(f"LoRA config:\n{config}")
    return config

def run(num_layers=1, message="", model_name="LLaMA3-FARCER-LoRA", pretrained=None, dataset="MELD", history=False, lr=0.0002):
    hf_login(token = "HF-TOKEN")

    params = ParamsFarcer()
    # keep the original LM parameters frozen
    params.freeze_lm = True
    params.freeze_vm = True
    params.freeze_mp = False
    params.torch_dtype = torch.bfloat16
    params.num_mp_layers = num_layers
        
    model = FarcerModel(params)
    model.train()
    model.lm.gradient_checkpointing_enable()
    model.lm.enable_input_require_grads()
    model_copy = deepcopy(model)

    if pretrained is not None:
        peft_model = PeftModel.from_pretrained(model, pretrained, is_trainable=True)
        print(f"{pretrained} loaded")
        print(" ")
        print("Trainable parameters:")
        for name, param in peft_model.named_parameters():
            if param.requires_grad:
                print(name)
    else:
        config = load_lora_config()
        peft_model = get_peft_model(model, config)
        peft_model.print_trainable_parameters()

    print(peft_model)

    # align with DialogueLLM: https://arxiv.org/abs/2310.11374
    EPOCHS = 10
    LR = lr
    optimizer = optim.AdamW(peft_model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    BATCHSIZE = 1

    current_time = train(
        peft_model,
        optimizer,
        scheduler,
        num_epochs=EPOCHS,
        batch_size=BATCHSIZE,
        dataset_type=dataset,
        prompt_list=True,
        message=message,
        history=history
        )

    # see newly added lora parameters
    for name, param in peft_model.base_model.named_parameters():
        if "lora" not in name:
            continue
        print(f"New parameter {name:<13} | {param.numel():>5} parameters | updated")

    # load the best model
    state_dict = torch.load(ph.get_target_dir(f"modeling/checkpoints/modality_projector/{current_time}/best.pth.tar"), map_location='cuda')['model_state_dict']
    peft_model.load_state_dict(state_dict)

    # save the model on HF
    user = "YOUR-USERNAME"
    model_id = f"{user}/{model_name}"
    peft_model.push_to_hub(model_id)
    print(f"Model saved to HuggingFace as {model_id}")

    # see updated parameters
    params_before = dict(model_copy.named_parameters())
    for name, param in peft_model.base_model.named_parameters():
        if "lora" in name:
            continue

        name_before = (
            name.partition(".")[-1].replace("original_", "").replace("module.", "").replace("modules_to_save.default.", "")
        )
        param_before = params_before[name_before]
        if torch.allclose(param, param_before):
            print(f"Parameter {name_before:<13} | {param.numel():>7} parameters | not updated")
        else:
            print(f"Parameter {name_before:<13} | {param.numel():>7} parameters | updated")

