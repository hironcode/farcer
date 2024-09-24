from datasets import load_dataset
import numpy as np
import torch
import pandas as pd
import PIL
from tqdm import tqdm
import time
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import os
import json
from sklearn.metrics import accuracy_score, f1_score

try:
    from modeling.model import FarcerModel, ParamsFarcer, load_farser, Farcer
    from modeling import util
    from src import features, config, plots
except:
    from model import FarcerModel, ParamsFarcer
    import util
    from ..src import features, config, plots


ph = config.PathHelper()

def eval_whisper(pipe, data):
    dataset = load_dataset(data)
    sample = dataset['train'][0]
    result = pipe(sample.copy())
    return result

def load_prompts(dataset="IEMOCAP", history=False, farser=True):
    prompts = []
    
    history = "_history" if history else ""
    model="cls" if farser else "llama"
    print("Loaded paths:")
    for shot in [3, 7]:
        path = ph.get_target_dir(f"modeling/prompts/testing/{dataset}/prompt_cls_{shot}shots_{model}{history}.txt")
        print(ph.remove_unnecessary_paths(path))
        with open(path, "r") as f:
            prompts.append(f.read())
    print(" ")
    return prompts

with open(ph.get_target_dir("modeling/prompts/testing/MELD/prompt_cls_3shots_cls.txt"), "r") as f:
    PROMPT_CLS_3_SHOTS = f.read()

with open(ph.get_target_dir("modeling/prompts/testing/MELD/prompt_cls_7shots_cls.txt"), "r") as f:
    PROMPT_CLS_7_SHOTS = f.read()

PROMPT_CLS = [PROMPT_CLS_3_SHOTS, PROMPT_CLS_7_SHOTS]

with open(ph.get_target_dir("modeling/prompts/testing/MELD/prompt_cls_3shots_llama.txt"), "r") as f:
    PROMPT_CLS_3_SHOTS_LLAMA = f.read()

with open(ph.get_target_dir("modeling/prompts/testing/MELD/prompt_cls_7shots_llama.txt"), "r") as f:
    PROMPT_CLS_7_SHOTS_LLAMA = f.read()

PROMPT_CLS_LLAMA = [PROMPT_CLS_3_SHOTS_LLAMA, PROMPT_CLS_7_SHOTS_LLAMA]

with open(ph.get_target_dir("modeling/prompts/testing/MELD/prompt_cls_3shots_cls_history.txt"), "r") as f:
    PROMPT_CLS_3_SHOTS_HISTORY = f.read()

with open(ph.get_target_dir("modeling/prompts/testing/MELD/prompt_cls_7shots_cls_history.txt"), "r") as f:
    PROMPT_CLS_7_SHOTS_HISTORY = f.read()

PROMPT_CLS_HISTORY = [PROMPT_CLS_3_SHOTS_HISTORY, PROMPT_CLS_7_SHOTS_HISTORY]

with open(ph.get_target_dir("modeling/prompts/testing/MELD/prompt_cls_3shots_llama_history.txt"), "r") as f:
    PROMPT_CLS_3_SHOTS_LLAMA_HISTORY = f.read()

with open(ph.get_target_dir("modeling/prompts/testing/MELD/prompt_cls_7shots_llama_history.txt"), "r") as f:
    PROMPT_CLS_7_SHOTS_LLAMA_HISTORY = f.read()

PROMPT_CLS_LLAMA_HISTORY = [PROMPT_CLS_3_SHOTS_LLAMA_HISTORY, PROMPT_CLS_7_SHOTS_LLAMA_HISTORY]


@torch.no_grad()
def eval_farser_and_llama(
        farser, 
        llama,
        llama_tokenizer,
        batch_size,
        fdata: pd.DataFrame,
        ldata: pd.DataFrame,
        prompt:str,
        prompt_llama:str,
        path_used:str=None,
        dataset_type="MELD",
    ) -> dict[np.ndarray[int]]:
    """
    Description:
        Evaluates the model on the MELD dataset.
    Args:
        model: torch.nn.Module
        batch_size: int
        data: pd.DataFrame with columns 'Dialogue_ID', 'Utterance_ID', 'Emotion', 'Transcript'
    Returns:
        results: dict with keys 'Dialogue_ID', 'Utterance_ID', 'Emotion', 'Transcript', 'Predicted_Emotion'
    """
    result = {}
    llama_eval_loss = np.array([])
    farser_eval_loss = np.array([])
    # Reference: https://www.dskomei.com/entry/2021/09/28/110016

    loop = 1
    farser_tokenizer = farser.lm_tokenizer

    pbar = tqdm(range(len(fdata["Dialogue_ID"].unique()) // batch_size), dynamic_ncols=True)

    current_time = time.strftime("%m-%d-%Y_%H:%M")
    losses_farser = 0
    losses_llama = 0

    # run for loop for each batch based on the number of dialogues
    for i in pbar:
        # decrase the batch size if the last batch is smaller than the specified batch size
        if len(fdata) - batch_size*i < batch_size:
            batch_size_remainder = len(fdata) - batch_size*i
        else:
            batch_size_remainder = batch_size

        start_idx = i*batch_size
        end_idx = start_idx + batch_size_remainder - 1

        # for farser
        futt_batch:list[str] = fdata.loc[start_idx:end_idx, "Utterances"].values.tolist()  # (batch_size, 1): 1 is the number of concatenated utterances
        # for llama
        lutt_batch:list[str] = ldata.loc[start_idx:end_idx, "Utterances"].values.tolist()  # (batch_size, 1): 1 is the number of concatenated utterances
        
        label_batch:list[str] = fdata.loc[start_idx:end_idx, "Emotion"].values.tolist()    # (batch_size, 1): 1 is the number of emotion labels of the last person in the conversation
        generic_image_batch = fdata.loc[start_idx:end_idx, [col for col in fdata.columns if "image_path_" in col]]
        image_batch:list[list[str]] = util.drop_pad_image(generic_image_batch).values.tolist()     # (batch_size, num_images)
        prompt_batch:list[str] = [prompt]*batch_size
        

        # preprocess (pixelize and tokenize) utterances, labels, images, and prompts
        # apply util.get_image_pixels function to the image path list of each conversation
        pixels: list[list[PIL.Image.Image]] = list(map(lambda l: util.get_image_pixels(farser.feature_extractor, image_paths=l), image_batch)) # (batch_size, num_images, 3, 224, 224)
        assert type(pixels) == list, "pixels must be a list of tensors"

        futt_tokens:torch.Tensor = farser_tokenizer(futt_batch, padding="longest", return_tensors="pt", add_special_tokens=False)   # (batch_size, num_conversation_tokens)
        prompt_tokens:torch.Tensor = farser_tokenizer(prompt_batch, padding="longest", return_tensors="pt", add_special_tokens=False) # (batch_size, num_prompt_tokens)
        label_tokens:torch.Tensor = farser_tokenizer(label_batch, padding="longest", return_tensors="pt", add_special_tokens=False)   # (batch_size, 1)


        # multiple forward passes for next token prediction
        multi_losses_farser = []
        input_embs = None
        input_att_masks = None
        its = label_tokens['input_ids'].shape[1]
        for i in range(its):
            outputs, input_embs, input_att_masks = farser(
                utterances=futt_tokens,
                image_pixels=pixels,
                prompt=prompt_tokens,
                iteration=i,
                labels=label_tokens,
                input_embs=input_embs,
                input_att_masks=input_att_masks,
                output_hidden_states=False,
                output_attentions=False,
                past_key_values=None,
            )
            multi_losses_farser.append(outputs['loss'])

        
        # evaluate LLaMA
        message_batch = []
        for i in range(batch_size):
            messages = [
                {"role": "system", "content": prompt_llama},
                {"role": "user", "content": lutt_batch[i]},
            ]
            message_batch.append(messages)
        
        utt_tokens_llama = farser_tokenizer.apply_chat_template(
            message_batch,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=True,
            padding_side="left",
            add_generation_prompt=True,
        )

        utt_token_ids_llama = utt_tokens_llama.to(llama.device)
        labels = label_tokens["input_ids"].to(llama.device)

        multi_outputs_llama, multi_losses_llama = multi_forward(
            llama, 
            utt_token_ids_llama, 
            labels, 
            output_hidden_states=False, 
        )

        loss_farser = 0
        for loss in multi_losses_farser:
            loss_farser += loss.item()
        losses_farser += loss_farser / len(multi_losses_farser)
        
        loss_llama = 0
        for loss in multi_losses_llama:
            loss_llama += loss.item()
        losses_llama += loss_llama / len(multi_losses_llama)

        print(f"FARSER Probable Token: {farser.lm_tokenizer.batch_decode(outputs['logits'][:, -its:, :].view(-1, farser.lm.config.vocab_size).argmax(dim=-1))} | Target Token: {farser.lm_tokenizer.batch_decode(labels)} | Loss: {loss_farser}")
        print(f"LLAMA Probable Token: {farser.lm_tokenizer.batch_decode(multi_outputs_llama[-1]['logits'][:, -its:, :].view(-1, farser.lm.config.vocab_size).argmax(dim=-1))} | Target Token: {farser.lm_tokenizer.batch_decode(labels)} | Loss: {loss_llama}")
        print(" ")

        pbar.set_postfix(farser_loss=losses_farser/loop, llama_loss=losses_llama/loop)
        farser_eval_loss = np.append(farser_eval_loss, loss_farser)
        llama_eval_loss = np.append(llama_eval_loss, loss_llama)
        loop += 1

    print("FARSER Eval Loss Mean:", farser_eval_loss.mean())
    print("LLAMA Eval Loss Mean:", llama_eval_loss.mean())

    PATH = ph.get_target_dir(f"reports/evaluation/{current_time}")
    if not os.path.exists(PATH):
        os.makedirs(PATH, exist_ok=True)
    result = {
        "description": "Evaluation FARSER vs LLaMA",
        "used_checkpoint": str(ph.remove_unnecessary_paths(path_used)),
        "current_time": current_time,
        "instance_size": len(fdata),
        "batch_size": batch_size,
        "dataset_type": "MELD",
        "farser_loss_mean": farser_eval_loss.mean(),
        "val_losses_mean": llama_eval_loss.mean(),
        "quantize": farser.params.lm_quantize,
        "farser_loss": farser_eval_loss.tolist(),
        "val_losses": llama_eval_loss.tolist(),
    }
    with open(os.path.join(PATH, "result.json"), "w") as f:
        json.dump(result, f)
    loss_dict = {
        "FARSER": farser_eval_loss,
        "LLAMA": llama_eval_loss,
    }
    plots.draw_farser_vs_llama(current_time, loss_dict, loss_type="CrossEntropyLoss")

    return result

def multi_forward(
        model, 
        input_ids,
        labels:torch.Tensor=None,
        output_hidden_states:bool=False,
    ):

    criterion = torch.nn.CrossEntropyLoss()
    multi_outputs = []
    multi_losses = []
    for i in range(labels.shape[1]):
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=output_hidden_states,
        )
        multi_outputs.append(outputs)
        logits = outputs['logits']

        inp = logits[:, -1, :].view(-1, model.config.vocab_size)
        tgt = labels[:, i].view(-1)
        loss = criterion(inp, tgt)
        multi_losses.append(loss)

        # concat the newly predicted token to the input embeddings for the next iteration
        pred_token = outputs['logits'][:, -1, :].argmax(dim=-1).view(-1, 1)
        input_ids = torch.cat([input_ids, pred_token], dim=1)

        del pred_token, logits, outputs, inp, tgt, loss
    return multi_outputs, multi_losses


def eval_dataset(model:FarcerModel, dataset_type:str, eos_token=True, history=False):
    if not model.params.vm_only_use_cls_token:
        image_size = model.vm.config.image_size
        patch_size = model.vm.config.patch_size
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        num_image_tokens = int((image_size/patch_size)**2)+1
    else:
        num_image_tokens = 1
        # load Dataset
    eos_token = model.lm_tokenizer.eos_token if eos_token else ""
    iemocap_test = dataset_type == "IEMOCAP"
    data: pd.DataFrame = features.format_corpus(dataset_type, 'test', update=False)
    # columns: Dialogue_ID, Utterance_ID, Speaker, Emotion, CaptureTime, image_path
    dataloader_farser = util.concat_data(data, model.lm_tokenizer.mask_token, num_image_tokens=num_image_tokens, label_utt_index=-1, eos_token=eos_token, history=history, iemocap_test=iemocap_test)
    dataloader_llama = util.concat_data(data, model.lm_tokenizer.mask_token, num_image_tokens=0, label_utt_index=-1, eos_token=eos_token, history=history, iemocap_test=iemocap_test)
    return dataloader_farser, dataloader_llama

def eval(quantize=None, path=None, pruned=True):
    """
    Description:
        Evaluates the model on the MELD dataset.
    Args:
        quantize: int: 4, 8, or None. If None, the model is not quantized.
        path: str: path to the model weights. If None, the best weights are loaded.
    """
    farser_params = ParamsFarcer()
    farser_params.lm_quantize = quantize
    if path is None:
        path = util.get_path_to_best_weights(ph.get_target_dir("modeling/checkpoints/modality_projector"))
    farser = FarcerModel(farser_params)
    checkpoint = torch.load(path, map_location=torch.device(farser.lm.device))
    if pruned:
        state_dict = {k.replace("mp.0.", ""): v for k, v in checkpoint['model_state_dict'].items()}
        farser.mp[0].load_state_dict(state_dict)
    else:
        farser.load_state_dict(state_dict)
    farser.eval()

    lm_name = ph.get_target_dir("data/meta-llama/Meta-Llama-3-8B-Instruct")
    if quantize == 4:
        llama = AutoModelForCausalLM.from_pretrained(
            lm_name,
            torch_dtype=farser_params.torch_dtype,
            quantization_config=farser_params.lm_4quantization_config,
            device_map="auto"
        )
    elif quantize == 8:
        llama = AutoModelForCausalLM.from_pretrained(
            lm_name,
            torch_dtype=farser_params.torch_dtype,
            quantization_config=farser_params.lm_8quantization_config,
            device_map="auto"
        )
    elif quantize is None:
        llama = AutoModelForCausalLM.from_pretrained(lm_name, torch_dtype=farser_params.torch_dtype).to(farser.input_embeddings.weight.device)
    else:
        raise ValueError("quantize must be 4, 8, or None")

    llama_tokenizer = AutoTokenizer.from_pretrained(lm_name)
    llama.eval()
    print(f"LLAMA Model: {llama.device}")

    data_farser, data_llama = eval_dataset(farser, "MELD")
    result = eval_farser_and_llama(farser, llama, llama_tokenizer, 1, data_farser, data_llama, PROMPT_CLS_7_SHOTS, PROMPT_CLS_7_SHOTS_LLAMA, path_used=path)
    print(result)

@torch.no_grad()
def eval_acc_f1(quantize=None, path=None, pruned=True, num_layers=1, lora_path=None, prompt_type_history=False, dataset_type="MELD"):
    params = ParamsFarcer()
    params.freeze_mp = True
    params.freeze_vm = True
    params.freeze_lm = True
    params.lm_quantize = quantize
    params.num_mp_layers = num_layers
    params.modality_projector_device = "cuda:0"
    farser = FarcerModel(params)
    tokenizer = farser.lm_tokenizer
    feature_extractor = farser.feature_extractor

    if lora_path is None:
        if path is None:
            path = util.get_path_to_best_weights(ph.get_target_dir("modeling/checkpoints/modality_projector"))
        state_dict = torch.load(path, map_location=torch.device(farser.lm.device))['model_state_dict']
        if pruned:
            state_dict = {k.replace("mp.", ""): v for k, v in state_dict.items()}
            farser.mp.load_state_dict(state_dict)
        else:
            farser.load_state_dict(state_dict)
    else:
        config = PeftConfig.from_pretrained(lora_path)
        farser = PeftModel.from_pretrained(farser, lora_path, is_trainable=False)
        if path:
            state_dict = {k.replace("base_model.model.mp.", ""): v for k, v in state_dict.items()}
            farser.base_model.mp.load_state_dict(state_dict)
        print("PEFT Model:", type(farser))
    
    prompts = load_prompts(dataset_type, prompt_type_history, farser=True)

    farser.eval()
    if lora_path:
        data, _ = eval_dataset(farser.model, dataset_type=dataset_type, eos_token=False, history=prompt_type_history)
    else:
        data, _ = eval_dataset(farser, dataset_type=dataset_type, eos_token=False, history=prompt_type_history)

    data = data.reset_index(drop=True)
    
    current_time = time.strftime("%m-%d-%Y_%H:%M")
    message = f"Evaluation FARSER on {dataset_type} with 3 layers and weight of best f1 score. Measured by F1 and Accuracy. Current Time: " + current_time


    for i, prompt in enumerate(prompts):
        print("Prompt:", prompt)
        y_pred = []
        y_true = []
        pbar = tqdm(range(len(data)), dynamic_ncols=True)
        if i == 0:
            shots = 3
        elif i == 1:
            shots = 7
        for b in pbar:
            # for farser
            utt_batch:list[str] = data.loc[b:b, "Utterances"].values.tolist()  # (batch_size, 1): 1 is the number of concatenated utterances
            
            labels:list[str] = data.loc[b:b, "Emotion"].values.tolist()    # (batch_size, 1): 1 is the number of emotion labels of the last person in the conversation
            generic_image_batch = data.loc[b:b, [col for col in data.columns if "image_path_" in col]]
            image_batch:list[list[str]] = util.drop_pad_image(generic_image_batch).values.tolist()     # (batch_size, num_images)
            prompt_batch:list[str] = [prompt]    
            print(f"==============NEW BATCH {b}==============")
            print("Utterances:", utt_batch)
            print("Labels:", labels)

            # preprocess (pixelize and tokenize) utterances, labels, images, and prompts
            # apply util.get_image_pixels function to the image path list of each conversation
            pixels: list[list[PIL.Image.Image]] = list(map(lambda l: util.get_image_pixels(feature_extractor, image_paths=l), image_batch)) # (batch_size, num_images, 3, 224, 224)
            assert type(pixels) == list, "pixels must be a list of tensors"

            utt_tokens = tokenizer(utt_batch, padding="longest", return_tensors="pt", add_special_tokens=False)   # (batch_size, num_conversation_tokens)
            prompt_tokens = tokenizer(prompt_batch, padding="longest", return_tensors="pt", add_special_tokens=False) # (batch_size, num_prompt_tokens)    

            # pred_tokens = farser.generate(
            #     max_gen_len=5,
            #     prompt=prompt_tokens,
            #     dialogue=utt_tokens,
            #     image_pixels=pixels,
            #     temperature=0,
            #     top_p=0.9,
            # )

            input_embs = None
            input_att_masks = None
            predicted_tokens = []
            for j in range(5):
                outputs, input_embs, input_att_masks = farser(
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
                next_token = torch.argmax(outputs['logits'][:, -1, :], dim=-1)
                predicted_tokens.append(next_token)
                # break if the predicted token is eos_token
                if next_token.item() == tokenizer.eos_token_id:
                    break
            predicted_tokens = torch.stack(predicted_tokens, dim=1)
            pred_tokens = tokenizer.batch_decode(predicted_tokens, skip_special_tokens=True)
            print("Predicted Tokens:", pred_tokens)
            assert len(pred_tokens) == len(labels)
            y_pred.extend(pred_tokens)
            y_true.extend(labels)
        
        f1 = f1_score(y_true, y_pred, average="weighted")
        acc = accuracy_score(y_true, y_pred)

        print("====================================\n====================================\n====================================")
        # print(f"y_true {shots}-shot:", y_true)
        # print(f"y_pred {shots}-shot:", y_pred)
        print(f"F1 Score {shots}-shot:", f1)
        print(f"Accuracy {shots}-shot:", acc)
        ("====================================\n====================================\n====================================")
 
        result = {
            "description": message,
            "current_time": current_time,
            "dataset_type": dataset_type,
            "shots": shots,
            "f1": f1,
            "accuracy": acc,
            "used_checkpoint": str(ph.remove_unnecessary_paths(path)) if lora_path is None else lora_path,
            "quantize": farser.params.lm_quantize,
            "prompt_type": "history" if prompt_type_history else "cls",
            "y_true": y_true,
            "y_pred": y_pred,
        }
    
        PATH = ph.get_target_dir(f"reports/evaluation/{current_time}_FarserF1_History_{prompt_type_history}_Layer_{num_layers}_{dataset_type}")
        if not os.path.exists(PATH):
            os.makedirs(PATH, exist_ok=True)
        with open(os.path.join(PATH, f"f1_acc_result_{shots}shot.json"), "w") as f:
            json.dump(result, f)
    

@torch.no_grad()
def eval_acc_f1_llama(path=ph.get_target_dir("data/meta-llama/Meta-Llama-3-8B-Instruct"), prompt_type_history=False, dataset_type="MELD"):
    params = ParamsFarcer()
    params.lm_quantize = 4
    if path is None:
        path = util.get_path_to_best_weights(ph.get_target_dir("modeling/checkpoints/modality_projector"))
    farser = FarcerModel(params)

    llama = AutoModelForCausalLM.from_pretrained(path).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(path)
    _, data = eval_dataset(farser, dataset_type, eos_token=False, history=prompt_type_history)
    data = data.reset_index(drop=True)
    
    current_time = time.strftime("%m-%d-%Y_%H:%M")
    message = f"Evaluation LLAMA on {dataset_type}. Measured by F1 and Accuracy. Do Sample True this time. Current Time: " + current_time

    prompts = load_prompts(dataset_type, prompt_type_history, farser=False)

    for i, prompt in enumerate(prompts):
        print("Prompt:", prompt)
        if i == 0:
            shots = 3
        elif i == 1:
            shots = 7

        y_pred = []
        y_true = []

        pbar = tqdm(range(len(data)), dynamic_ncols=True)
        
        for b in pbar:
            # for farser
            utt_batch:list[str] = data.loc[b:b, "Utterances"].values.tolist()  # (batch_size, 1): 1 is the number of concatenated utterances
            
            labels:list[str] = data.loc[b:b, "Emotion"].values.tolist()    # (batch_size, 1): 1 is the number of emotion labels of the last person in the conversation

            print(f"==============NEW BATCH {b}==============")
            print("Utterances:", utt_batch)
            print("Labels:", labels)

            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": utt_batch[0]},
            ] 

            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(llama.device)

            outputs = llama.generate(
                input_ids,
                max_new_tokens=5,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                temperature=0,
                top_p=0.9,
            )
            pred_tokens = outputs[0][input_ids.shape[-1]:]

            pred_tokens = tokenizer.decode(pred_tokens, skip_special_tokens=True)
            print("Predicted Tokens:", pred_tokens)
            y_pred.append(pred_tokens)
            y_true.extend(labels)
        
        f1 = f1_score(y_true, y_pred, average="weighted")
        acc = accuracy_score(y_true, y_pred)
        
        print(f"F1 Score {shots}-shot:", f1)
        print(f"Accuracy {shots}-shot:", acc)
 
        result = {
            "description": message,
            "current_time": current_time,
            "dataset_type": dataset_type,
            "shots": shots,
            "f1": f1,
            "accuracy": acc,
            "used_checkpoint": str(ph.remove_unnecessary_paths(path)),
            "quantize": farser.params.lm_quantize,
            "prompt_type": "history" if prompt_type_history else "cls",
            "y_true": y_true,
            "y_pred": y_pred,
        }
    
        PATH = ph.get_target_dir(f"reports/evaluation/{current_time}_LlamaF1_History_{prompt_type_history}_{dataset_type}")
        if not os.path.exists(PATH):
            os.makedirs(PATH, exist_ok=True)
        with open(os.path.join(PATH, f"f1_acc_result_{shots}shot.json"), "w") as f:
            json.dump(result, f)
    
