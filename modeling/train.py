import PIL.Image
import torch
import sys
import os
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

try:
    from modeling.model import FarcerModel, ParamsFarcer
    from modeling import util
    from src import features, config, plots
    from modeling.trainer_validator import train_batch_modality_projector, validate
except:
    from model import FarcerModel, ParamsFarcer
    import util
    from ..src import features, config, plots
    from trainer_validator import train_batch_modality_projector, validate

ph = config.PathHelper()

def train_epochs_modality_projector(
        model:FarcerModel,
        optimizer, 
        scheduler,
        batch_size=5,
        dataset_type:str="MELD", 
        num_epochs=10,
        early_stopping=False,
        prompt_list:bool=False,
        message:str=None,
        history:bool=False
    ) -> None:
    """
    model: FarcerModel
    data: preprossed (tokenized) data, but image paths are not yet converted to pixels
    """
    if not model.params.vm_only_use_cls_token:
        image_size = model.vm.config.image_size
        patch_size = model.vm.config.patch_size
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        num_image_tokens = int((image_size/patch_size)**2)+1
    else:
        num_image_tokens = 1
    # load Dataset
    if dataset_type == "MELD":
        data: pd.DataFrame = features.format_corpus(dataset_type, 'train', update=False)
        data2 = features.format_corpus('dev', update=False)
        # columns: Dialogue_ID, Utterance_ID, Speaker, Emotion, CaptureTime, image_path

        # dataloader:pd.DataFrame = util.concat_data(data, image_token=model.lm_tokenizer.mask_token, num_image_tokens=num_image_tokens, label_utt_index=-1, eos_token=model.lm_tokenizer.eos_token)
        # dataloader2:pd.DataFrame = util.concat_data(data2, image_token=model.lm_tokenizer.mask_token, num_image_tokens=num_image_tokens, label_utt_index=-1, eos_token=model.lm_tokenizer.eos_token)
        # dataloader = pd.concat([dataloader, dataloader2], ignore_index=True)
        train:pd.DataFrame = util.concat_data(data, image_token=model.lm_tokenizer.mask_token, num_image_tokens=num_image_tokens, label_utt_index=-1, eos_token=model.lm_tokenizer.eos_token)
        valid:pd.DataFrame = util.concat_data(data2, image_token=model.lm_tokenizer.mask_token, num_image_tokens=num_image_tokens, label_utt_index=-1, eos_token=model.lm_tokenizer.eos_token)
        print(f"{dataset_type} Data Size: {len(train)}")
    elif dataset_type == "IEMOCAP":
        data = features.format_corpus(dataset_type, 'train', update=False)
        data:pd.DataFrame = util.concat_data(data, image_token=model.lm_tokenizer.mask_token, num_image_tokens=num_image_tokens, label_utt_index=-1, eos_token=model.lm_tokenizer.eos_token, history=history)
        # make an empty valid set
        train, valid = train_test_split(data, test_size=0.1, random_state=42, stratify=data['Emotion'])
        print(f"{dataset_type} Data Size: {len(train)}")

    print(f"Check dataset")
    print(train.head())

    total_epoch_start = time.time()
    current_time = time.strftime("%m-%d-%Y_%H:%M:%S")
    train_losses = []
    val_losses = []
    train_batch_losses = []
    best_loss = float('inf')
    best_loss_epoch = 0
    
    # Create and load Loss Weights for Cross Entropy Loss
    loss_weights = util.get_loss_weights(train['Emotion'], model.lm_tokenizer, model.lm.config.vocab_size)
    loss_weights[model.lm_tokenizer.eos_token_id] += 0.5
    for i, weight in enumerate(loss_weights):
        if weight != 1:
            print(f"{model.lm_tokenizer.decode(i)}({i}) weight: {weight}")

    train = train.reset_index(drop=True)
    valid = valid.reset_index(drop=True)
    print(f"Label distribution % in train set:\n{train['Emotion'].value_counts()/len(train)*100}")
    print(f"Label distribution % in valid set:\n{valid['Emotion'].value_counts()/len(valid)*100}")

    print("Check a dialogue:")
    print(train.loc[0, "Utterances"])

    if prompt_list:
        prompts = []
        # num is equal to the number of files in the prompts/random_shots_cls directory
        if dataset_type == "MELD":
            path = "modeling/prompts/random_shots_cls"
        elif dataset_type == "IEMOCAP":
            path = "modeling/prompts/random_shots_cls_iemocap"
        else:
            raise ValueError("Dataset type is not valid")
        
        if history:
            path = ph.get_target_dir(path+"_history")
            num = len(os.listdir(path))
        else:
            path = ph.get_target_dir(path)
            num = len(os.listdir(path))

        for i in range(num):
            with open(os.path.join(path, f"prompt_cls_7shots_cls{i}.txt"), "r") as f:
                prompts.append(f.read())
        print(f"Prompt sample: {prompts[0]}")
    
    
    PATH = ph.get_target_dir(f"modeling/checkpoints/modality_projector/{current_time}")

    if early_stopping:
        es = EarlyStopping(patience=7, verbose=True)

    trian_f1s = []
    train_accs = []
    val_f1s = []
    val_accs = []

    for epoch in range(num_epochs):
        print("="*50)
        print(f"Epoch{epoch} Started:\nBatch size: {batch_size} | Dataset: {dataset_type}\n")
        util.print_nvidia_smi()

        training_start = time.time()
        # set the model to train mode to enable gradient calculation
        
        model.load_loss_weights(loss_weights)
        model.train()
        train_loss, train_loss_list, trian_f1, train_acc = train_batch_modality_projector(
            model,
            train,
            optimizer,
            batch_size=batch_size, 
            ep=epoch,
            prompt=prompts,
            loss_weights = loss_weights,
        )
        scheduler.step()
        train_batch_losses.extend(train_loss_list)

        # validate only if the dataset is not IEMOCAP
        model.load_loss_weights(None)
        model.eval()
        val_loss, val_f1, val_acc = validate(
            model,
            valid,
            batch_size=batch_size,
            ep=epoch,
            prompt=prompts[0],
        )

        training_end = time.time()
        time_sec = round(training_end - training_start)
        print(f"\nEpoch{epoch} Ended:\nTraining Loss: {train_loss} | Validation Loss: {val_loss}\nTime: {datetime.timedelta(seconds=time_sec)} | Learning Rate: {scheduler.get_last_lr()[0]}\n")
        print(f"Train F1: {trian_f1} | Train Accuracy: {train_acc}\n")
        print(f"Validation F1: {val_f1} | Validation Accuracy: {val_acc}\n")

        trian_f1s.append(trian_f1)
        train_accs.append(train_acc)
        val_f1s.append(val_f1)
        val_accs.append(val_acc)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # is_best = val_loss < best_loss
        is_best = -val_f1 < best_loss
        if is_best:
            # best_loss = val_loss
            best_loss = -val_f1
            best_loss_epoch = epoch

        # save model
        util.save_checkpoints(epoch, model.state_dict(), optimizer.state_dict(), train_loss, val_loss, PATH, is_best=is_best)
        with open(os.path.join(PATH, "args.json"), "w") as f:
            json.dump(model.params.to_json(), f)
        print(f"Model saved at epoch {epoch}")
        print("="*50)

        if early_stopping:
            # es(val_loss)
            es(-val_f1)
            if es.early_stop:
                print(f"Early stopping at epoch{epoch}")
                break

    result = {
        "description": "Training Modality Projector",
        "datetime": current_time,
        "quantization": model.params.lm_quantize,
        "epoch": num_epochs,
        "batch_size": batch_size,
        "dataset_type": dataset_type,
        "best_loss": best_loss,
        "best_loss_epoch": best_loss_epoch,
        "train_loss_mean": np.mean(train_losses),
        "val_loss_mean": np.mean(val_losses),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_f1s": trian_f1s,
        "train_accs": train_accs,
        "val_f1s": val_f1s,
        "val_accs": val_accs,
        "train_batch_losses": train_batch_losses,
    }
    with open(os.path.join(PATH, "result.json"), "w") as f:
        json.dump(result, f)
    with open(os.path.join(PATH, "message.txt"), "w") as f:
        f.write(message)
    os.makedirs(ph.get_target_dir(f"reports/training/{current_time}"), exist_ok=True)
    with open(os.path.join(ph.get_target_dir(f"reports/training/{current_time}"), "result.json"), "w") as f:
        json.dump(result, f)
    with open(os.path.join(ph.get_target_dir(f"reports/training/{current_time}"), "message.txt"), "w") as f:
        f.write(message)

    plots.draw_loss_line_graph(current_time,"train_val_losses",  np.array(train_losses), np.array(val_losses), loss_type="CrossEntropyLoss")
    plots.draw_loss_line_graph(current_time, "train_batch_losses", train_loss=np.array(train_batch_losses), loss_type="CrossEntropyLoss")

    total_epoch_end = time.time()
    time_sec = round(total_epoch_end - total_epoch_start)
    print(f"Total training time: {datetime.timedelta(seconds=time_sec)}")
    return current_time

def train_modality_projector(device="supercomputer"):
    # Load model
    params = ParamsFarcer()
    if params.device == 'cpu' or device != "supercomputer":
        # if a model is quantized, it assignes a correct device and dtype (most likely CPU)
        params.lm_quantize = 4
        params.device = "cuda:0"
        params.torch_dtype = torch.bfloat16

    if torch.cuda.is_available():
        print(f"AVILABLE CUDA DEVICE COUNT: {torch.cuda.device_count()}")
    
    model = FarcerModel(params)
    model.train()
    EPOCHS = 15
    BATCH_SIZE = 1
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(EPOCHS/2))
    message = "Training conditions: 7 shots, 1 batch size, 15 epochs, MELD dataset, early stopping, radom 10 prompts, eos token added to labels"
    train_epochs_modality_projector(model, optimizer, scheduler, batch_size=BATCH_SIZE, num_epochs=EPOCHS, dataset_type="MELD", early_stopping=True, prompt_list=True, message=message)
    print("Training complete")


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
