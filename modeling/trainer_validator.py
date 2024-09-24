import PIL.Image
import torch
import os
from tqdm import tqdm
import pandas as pd
from typing import Optional
import PIL
from wasabi import msg
from sklearn.model_selection import train_test_split
import json
import numpy as np
import gc
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score, f1_score
import random

try:
    from modeling.model import FarcerModel, ParamsFarcer
    from modeling import util
    from src import features, config, plots
except:
    from model import FarcerModel, ParamsFarcer
    import util
    from ..src import features, config, plots

ph = config.PathHelper()

def train_batch_modality_projector(
        model: FarcerModel,
        train: pd.DataFrame,
        optimizer: torch.optim.AdamW, 
        batch_size: int,
        ep:int,
        prompt =None,
        loss_weights: Optional[dict] = None,
    ) -> None:
    """
    model: FarcerModel
    data: preprossed (tokenized) data, but image paths are not yet converted to pixels
    """

    # Reference: https://www.dskomei.com/entry/2021/09/28/110016

    loop = 1
    losses = 0
    tokenizer = model.lm_tokenizer
    if batch_size > 1:
        pbar = tqdm(range(len(train)//batch_size + 1), dynamic_ncols=True)
    else:
        pbar = tqdm(range(len(train)), dynamic_ncols=True)

    if type(prompt) == list:
        prompt_list = prompt.copy()
        np.random.seed(42)
        prompt_indices = np.random.randint(0, len(prompt), len(train)//batch_size)
    
    loss_list = []
    y_pred = []
    y_true = []
    # run for loop for each batch based on the number of dialogues
    for i in pbar:
        # decrase the batch size if the last batch is smaller than the specified batch size
        if len(train) - batch_size*i < batch_size:
            batch_size_remainder = len(train) - batch_size*i
        else:
            batch_size_remainder = batch_size

        start_idx = i*batch_size
        end_idx = start_idx + batch_size_remainder - 1

        utt_batch:list[str] = train.loc[start_idx:end_idx, "Utterances"].values.tolist()  # (batch_size, 1): 1 is the number of concatenated utterances
        label_batch:list[str] = train.loc[start_idx:end_idx, "Emotion"].values.tolist()    # (batch_size, 1): 1 is the number of emotion labels of the last person in the conversation

        generic_image_batch = train.loc[start_idx:end_idx, [col for col in train.columns if "image_path_" in col]] # (batch_size, num_images)
        image_batch:list[list[str]] = util.drop_pad_image(generic_image_batch).values.tolist()     # (batch_size, num_images)
        
        if type(prompt) == list:
            prompt = prompt_list[prompt_indices[i]]
        prompt_batch:list[str] = [prompt]*batch_size

        # preprocess (pixelize and tokenize) utterances, labels, images, and prompts
        # apply util.get_image_pixels function to the image path list of each conversation
        pixels: list[list[PIL.Image.Image]] = list(map(lambda l: util.get_image_pixels(model.feature_extractor, image_paths=l), image_batch)) # (batch_size, num_images, 3, 224, 224)
        assert type(pixels) == list, "pixels must be a list of tensors"

        tokenizer.padding_side = "left"
        utt_tokens:torch.Tensor = tokenizer(utt_batch, padding="longest", return_tensors="pt", add_special_tokens=False)   # (batch_size, num_conversation_tokens)
        prompt_tokens:torch.Tensor = tokenizer(prompt_batch, padding="longest", return_tensors="pt", add_special_tokens=False) # (batch_size, num_prompt_tokens)
        
        tokenizer.padding_side = "right"
        label_tokens:torch.Tensor = tokenizer(label_batch, padding="longest", return_tensors="pt", add_special_tokens=False)   # (batch_size, 1)


        # multiple forward passes for next token prediction
        input_embs = None
        input_att_masks = None
        loss_avg = 0
        its = label_tokens['input_ids'].shape[1]
        for j in range(its):
            outputs, input_embs, input_att_masks = model(
                utterances=utt_tokens,
                image_pixels=pixels,
                prompt=prompt_tokens,
                iteration=j,
                # labels=label_tokens,
                input_embs=input_embs,
                input_att_masks=input_att_masks,
                output_hidden_states=False,
                output_attentions=False,
                past_key_values=None,
            )
            # if j == its-1:
            #     outputs['loss'].backward()
            # else:
            #     outputs['loss'].backward(retain_graph=True)
            
            # outputs['loss'] = outputs['loss'].detach()
            # outputs['loss'].requires_grad = False
            # outputs['logits'] = outputs['logits'].detach()
            # outputs['logits'].requires_grad = False
            # loss_avg += float(outputs['loss'].item())

        logits = outputs['logits'][:, -its:, :]
        criterion = CrossEntropyLoss(loss_weights)
        loss = criterion(logits.reshape(-1, model.lm.config.vocab_size).to(model.params.regular_device), label_tokens['input_ids'].reshape(-1).to(model.params.regular_device))
        # backward only if require grad is True
        if loss.requires_grad:
            loss.backward()
        loss_avg = loss.detach().requires_grad_(False).item()
        
        # loss_avg = loss_avg/its
        optimizer.step()
        optimizer.zero_grad()

        # specualtion: when running on hipe computer, I get:
        # RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
        # This might be bacause the forward function does not apply self.mp linear layer, which is the only NN requires gradient
        losses += loss_avg
        loss_list.append(loss_avg)

        pbar.set_postfix(cumul_train_loss=losses/loop, lr=optimizer.param_groups[0]['lr'])
        loop += 1

        print(f"EPOCH: {ep} | BATCH: {i} | ITERATIONS :{its}")
        predicted_tokens = model.lm_tokenizer.batch_decode(outputs['logits'][:, -its:, :].argmax(dim=-1))
        target_tokens = model.lm_tokenizer.batch_decode(label_tokens['input_ids'])
        print(f"Probable Token: {predicted_tokens} | Target Token: {target_tokens} | Loss: {loss_avg}")
        print(" ")

        y_pred += predicted_tokens
        y_true += target_tokens

        del utt_batch, label_batch, generic_image_batch, image_batch, prompt_batch, pixels, utt_tokens, prompt_tokens, label_tokens, input_embs, input_att_masks, outputs, predicted_tokens, target_tokens, logits, loss, criterion
        torch.cuda.empty_cache()
        gc.collect()
    f1 = f1_score(y_true, y_pred, average='weighted')
    acc = accuracy_score(y_true, y_pred)
    return losses/loop, loss_list, f1, acc

@torch.no_grad()
def validate(
        model: FarcerModel,
        valid: pd.DataFrame,
        batch_size: int,
        ep:int,
        prompt=None,
    ) -> None:
    """
    model: FarcerModel
    data: preprossed (tokenized) data, but image paths are not yet converted to pixels
    """

    # Reference: https://www.dskomei.com/entry/2021/09/28/110016

    loop = 1
    losses = 0
    tokenizer = model.lm_tokenizer
    
    pbar = tqdm(range(len(valid["Dialogue_ID"].unique()) // batch_size), dynamic_ncols=True)
    
    y_pred = []
    y_true = []

    # run for loop for each batch based on the number of dialogues
    for i in pbar:
        # decrase the batch size if the last batch is smaller than the specified batch size
        if len(valid) - batch_size*i < batch_size:
            batch_size_remainder = len(valid) - batch_size*i
        else:
            batch_size_remainder = batch_size

        start_idx = i*batch_size
        end_idx = start_idx + batch_size_remainder - 1

        utt_batch:list[str] = valid.loc[start_idx:end_idx, "Utterances"].values.tolist()  # (batch_size, 1): 1 is the number of concatenated utterances
        label_batch:list[str] = valid.loc[start_idx:end_idx, "Emotion"].values.tolist()    # (batch_size, 1): 1 is the number of emotion labels of the last person in the conversation

        generic_image_batch = valid.loc[start_idx:end_idx, [col for col in valid.columns if "image_path_" in col]] # (batch_size, num_images)
        image_batch:list[list[str]] = util.drop_pad_image(generic_image_batch).values.tolist()     # (batch_size, num_images)

        prompt_batch:list[str] = [prompt]*batch_size
        
        # preprocess (pixelize and tokenize) utterances, labels, images, and prompts
        # apply util.get_image_pixels function to the image path list of each conversation
        pixels: list[list[PIL.Image.Image]] = list(map(lambda l: util.get_image_pixels(model.feature_extractor, image_paths=l), image_batch)) # (batch_size, num_images, 3, 224, 224)
        assert type(pixels) == list, "pixels must be a list of tensors"

        tokenizer.padding_side = "left"
        utt_tokens:torch.Tensor = tokenizer(utt_batch, padding="longest", return_tensors="pt", add_special_tokens=False)   # (batch_size, num_conversation_tokens)
        prompt_tokens:torch.Tensor = tokenizer(prompt_batch, padding="longest", return_tensors="pt", add_special_tokens=False) # (batch_size, num_prompt_tokens)

        tokenizer.padding_side = "right"
        label_tokens:torch.Tensor = tokenizer(label_batch, padding="longest", return_tensors="pt", add_special_tokens=False)   # (batch_size, 1)

        # multiple forward passes for next token prediction
        multi_losses = []
        input_embs = None
        input_att_masks = None
        its = label_tokens['input_ids'].shape[1]
        for t in range(its):
            outputs, input_embs, input_att_masks = model(
                utterances=utt_tokens,
                image_pixels=pixels,
                prompt=prompt_tokens,
                iteration=t,
                labels=label_tokens,
                input_embs=input_embs,
                input_att_masks=input_att_masks,
                output_hidden_states=False,
                output_attentions=False,
                past_key_values=None,
            )
            multi_losses.append(outputs['loss'])

        loss_sum = 0
        for loss in multi_losses:
            loss_sum += loss.item()

        losses += loss_sum/len(multi_losses)

        pbar.set_postfix(eval_loss=losses/loop)
        loop += 1
        
        print(f"VALIDATION | EPOCH: {ep} | BATCH: {i} | ITERATIONS :{its}")
        predicted_tokens = model.lm_tokenizer.batch_decode(outputs['logits'][:, -its:, :].argmax(dim=-1))
        target_tokens = model.lm_tokenizer.batch_decode(label_tokens['input_ids'])
        print(f"Probable Token: {predicted_tokens} | Target Token: {target_tokens} | Loss: {loss_sum/len(multi_losses)}")
        print(" ")
        y_pred += predicted_tokens
        y_true += target_tokens
    f1 = f1_score(y_true, y_pred, average='weighted')
    acc = accuracy_score(y_true, y_pred)
    return losses/loop, f1, acc