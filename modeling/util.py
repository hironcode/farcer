from PIL import Image
from transformers import ViTFeatureExtractor
import torch
import numpy as np 
import pandas as pd 
from typing import Any, Union
import os
import shutil
import subprocess
import random
from sklearn.utils.class_weight import compute_class_weight

try:
    from src.features import format_corpus
    from src.config import PathHelper
except:
    from ..src.features import format_corpus
    from ..src.config import PathHelper
import os
import glob
import datetime
import torch

ph = PathHelper()
# if torch.cuda.is_available(): device = torch.device("cuda:0")

def get_image_pixels(feature_extractor: ViTFeatureExtractor, image_paths: list[str]) -> torch.Tensor:
    for path in image_paths:
        if type(path) == float or path == "PAD": # if path is np.nan or a padding path
            continue
        extension = path.split(".")[-1].upper()
        if extension not in ['JPG', 'JPEG']:
            raise ValueError(f"{extension} images are not supported. Please use JPEG or JPG.")
        
    images = []
    nones = []
    PAD = "PAD"
    for i, image_path in enumerate(image_paths):
        # if image_path is None:
        #     image = np.full((224, 224, 3), float("inf"), dtype=np.uint8)
        #     image_np = image
        # else:
        #     image = Image.open(image_path)
        #     image_np = np.array(image)

        if type(image_path) == float or image_path == PAD: # if path is np.nan
            nones.append(i)
            continue
        image = Image.open(image_path)
        image_np = np.array(image)
        # RGBA to RGB
        # reference: https://www.kaggle.com/code/aungdev/change-a-rgba-image-to-a-rgb-image
        if image_np.shape[2] == 4:
            image = np.delete(arr=image_np, obj=3, axis=2)
        images.append(image)
    
    # reinsert Nones
    if len(images) > 0:
        pixels = feature_extractor(images=images, return_tensors="pt")["pixel_values"]
        pixels = [p for p in pixels]
        for i in nones:
            pixels.insert(i, np.NaN)
    else:   # if all image paths are missing
        pixels = [np.NaN]*len(nones)
    return pixels


def concat_data(df: pd.DataFrame, image_token:str, num_image_tokens=1, label_utt_index=-1, eos_token=None, history=False, iemocap_test=False) -> pd.DataFrame:
    """
    Description:
        Concatenates a list of utterances into a single string with multiple conversations, insert image token, and speaker token.
    Args:
        df: pd.DataFrame with columns 'Dialogue_ID', 'Utterance_ID', 'Utterance', "Speaker"
        image_token: str
        num_image_tokens: int: Number of image tokens (patches) to expect for each image. This is used to determine the number of image tokens to insert.
        label_utt_index: int: The index of the utterance whose speaker's emotoin label is used as the target label.
    Returns:
        df_complete: pd.DataFrame with columns 'Dialogue_ID', 'Utterances'
    """

    dialogue_ids = df['Dialogue_ID'].unique()
    dialogue_ids.sort()
    df_complete = pd.DataFrame({'Dialogue_ID': dialogue_ids, "Utterances": "", "Emotion": ""})

    img_token_series = "".join([image_token]*num_image_tokens)


    # prepare image paths for each conversation
    max_images = df.groupby('Dialogue_ID').size().max()
    PAD = "PAD"
    image_pads = pd.DataFrame(columns=[f"image_path_{i}" for i in range(max_images)])
    image_pads['Dialogue_ID'] = pd.Series(dialogue_ids, name='Dialogue_ID')
    assert len(image_pads) == len(df_complete), (len(image_pads), len(df_complete))
    df_complete = pd.merge(df_complete, image_pads, on='Dialogue_ID', how='outer').fillna(PAD)

    
    for dialogue_id in dialogue_ids:
        dialogue_df = df[df['Dialogue_ID'] == dialogue_id].sort_values('Utterance_ID')

        # concatenate utterances with image tokens
        utterances = []
        idx = 0
        max = len(dialogue_df)
        utt_count = -1
        if history:
            for utt, speaker, emotion in dialogue_df[['Utterance', 'Speaker', 'Emotion']].values:
                utt_count += 1

                # for iemocap test set, only include the last 10 utterances
                if iemocap_test and max - utt_count > 15:
                    idx += 1
                    continue

                if idx == max-1:
                    utterances.append(f"{speaker}: {img_token_series} {utt}\n{speaker}'s EMOTION LABEL:")
                else:
                    utterances.append(f"{speaker}: {img_token_series} {utt}\n({speaker}'s current emotion: {emotion})")
                idx += 1
        else:
            for utt, speaker in dialogue_df[['Utterance', 'Speaker']].values:
                utt_count += 1
                # for iemocap test set, only include the last 10 utterances
                if iemocap_test and max - utt_count > 10:
                    continue

                utterances.append(f"{speaker}: {img_token_series} {utt}")
            utterances.append("EMOTION LABEL:")
        utterances.insert(0, "DIALOGUE:")

        df_complete.loc[df_complete['Dialogue_ID']==dialogue_id, "Utterances"] = '\n'.join(utterances)

        # assign the image paths to each dialogue
        for i, image_path in enumerate(dialogue_df['image_path']):
            df_complete.loc[df_complete['Dialogue_ID']==dialogue_id, f"image_path_{i}"] = image_path
    
        # assign the emotion of the person at the specified index of each dialogue
        emotion = dialogue_df['Emotion'].iloc[label_utt_index].strip()
        if eos_token is not None:
            emotion += eos_token
        df_complete.loc[df_complete['Dialogue_ID']==dialogue_id, "Emotion"] = emotion
    return df_complete


def fetch_proper_dtype(params, mode:str="float", config:dict={"cuda": 16, "cpu": 32}) -> Any:
    """
    Description:
        Returns the proper dtype for the specified device.
    Args:
        device: torch.device
        mode [float, int]: str
        config: dict with keys "cuda" and "cpu" and corresponding dtype value
    Returns:
        dtype: Any
    """
    for k, v in config.items():
        if torch.device(params.device).type == k:
                if k == "cuda" and v == 16 and mode == "float" and "b" in str(params.torch_dtype):
                    return getattr(torch, f"b{mode}{v}")
                else:
                    return getattr(torch, f"{mode}{v}")
                

def drop_pad_image(df: pd.DataFrame, PAD="PAD") -> pd.DataFrame:
    """
    Description:
        Drops the redundant 'PAD' image paths from the image paths.
    Args:
        df: pd.DataFrame with columns 'Dialogue_ID', 'image_path_{i}'
    Returns:
        df: pd.DataFrame with columns 'Dialogue_ID', 'image_path_{i}'
    """
    for i in range(df.shape[1]):
        if f"image_path_{i}" in df.columns and df[f"image_path_{i}"].nunique() == 1 and df[f"image_path_{i}"].iloc[0] == PAD:
            df.drop(columns=[f"image_path_{i}"], inplace=True)
    return df

def move_every_pad_to_left_right(tensors: torch.Tensor, prompt_length:Union[int, None], pad_embs: torch.Tensor=None, att_mask=False, padding_side="left") -> list[torch.Tensor]:
    """
    Description:
        Moves every tensor with PAD to the back of the list.
    Args:
        tensors: list[torch.Tensor]
        pad_embs: torch.Tensor
    Returns:
        tensors: list[torch.Tensor]
    """
    if pad_embs is None and att_mask is False:
        raise ValueError("Padding Embeddings must be provided when att_mask is False.")
    
    promptl = prompt_length-1 if prompt_length is not None else 0
    for i in range(tensors.shape[0]):
        for j in range(tensors.shape[1]):
            if att_mask is False and torch.equal(tensors[i, j], pad_embs.squeeze(0)):
                if j < promptl:
                    continue
                if padding_side== "left":
                    # move the padding at [i, j] to the left of the tensor and shift the rest to the right
                    tensors[i, :j+1] = torch.cat([pad_embs, tensors[i, :j]], dim=0)
                elif padding_side == "right":
                    tensors[i, j:] = torch.cat([tensors[i, j+1:], pad_embs], dim=0)
            elif att_mask is True and tensors[i, j] == 0:
                if padding_side == "left":
                    tensors[i, :j+1] = torch.cat([torch.zeros(1), tensors[i, :j]], dim=0)
                elif padding_side == "right":
                    tensors[i, j:] = torch.cat([tensors[i, j+1:], torch.zeros(1)], dim=0)
    return tensors


def save_checkpoints(epoch, state_dict, optimizer_state_dict, train_loss, val_loss, dir_path, is_best=False):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'train_loss': train_loss,
        "val_loss": val_loss
        }, os.path.join(dir_path, f'epoch{epoch}.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(dir_path, f'epoch{epoch}.pth.tar'), os.path.join(dir_path, f'best.pth.tar'))


def print_nvidia_smi():
    COMMAND = "nvidia-smi"
    subprocess.run(COMMAND, shell=True)


def create_few_shots(num:int, filename:str, image_token:str="<image feature token>", SEED=1, prefix=None, dataset="IEMOCAP", history=False) -> str:
    """
    Description:
        Creates a few-shot prompt for the specified number of shots.
    Args:
        num: int
    Returns:
        prompt: str
    """
    if prefix is None:
        if dataset == "IEMOCAP":
            prefix = """You are FARCER, a high-quality emotion recognizer. Classify the emotion of the last speaker of a dialogue based on past utterances and the images of speakers.\nRules:\n1. Classify the emotion of the last speaker in the dialogue.\n2. The output must be one of the nine emotional labels: "neutral", "happiness", "sadness", "anger", "surprise", "disgust", "fear", "frustration", "excitement", or "other".\n3. The output must be a single emotion label.\n\nFew Shot Examples:\n\n"""
            mode = "train"
        elif dataset == "MELD":
            prefix = """You are FARCER, a high-quality emotion recognizer. Classify the emotion of the last speaker of a dialogue based on past utterances and the images of speakers.\nRules:\n1. Classify the emotion of the last speaker in the dialogue.\n2. The output must be one of the seven emotional labels: "neutral", "joy", "sadness", "anger", "surprise", "disgust", or "fear".\n3. The output must be a single emotion label.\n\nFew Shot Examples:\n\n"""
            mode = "dev"
        else:
            raise ValueError("Dataset not supported.")
    random.seed(SEED)
    
    df = format_corpus(dataset, mode, update=False)

    data = concat_data(df, image_token, num_image_tokens=1, label_utt_index=-1, history=history)
    labels = [[label, 1] for label in data['Emotion'].unique()]
    print(labels)
    max_values = data.groupby('Emotion').size().to_dict()

    if num < len(labels):
        labels = random.sample(labels, k=num)
        print(labels)
    elif num > len(labels):
        if num // len(labels) > 1:
            labels = [[label, i * num//len(labels)] for label, i in labels]
        idxes = random.sample(range(0, len(labels)), num%len(labels))
        for i in idxes:
            labels[i][1] += 1

    # Sample one dialogue with each label
    sampled_dialogues = []
    for label, i in labels:
        if i > max_values[label]:
            diff = i - max_values[label]
            i = max_values[label]
            dialogues = data[data['Emotion'] == label].sample(n=i, random_state=SEED, replace=True)['Utterances'].values
            
            rand_label = random.choice([l for l in sorted(max_values.keys(), key=lambda x: max_values[x])[:3] if l != label])
            dialogues_extra = data[data['Emotion'] == rand_label].sample(n=diff, random_state=SEED, replace=True)['Utterances'].values
            
            sampled_dialogues.extend([(di, label) for di in dialogues])
            sampled_dialogues.extend([(di, rand_label) for di in dialogues_extra])
        else:
            dialogues = data[data['Emotion'] == label].sample(n=i, random_state=SEED, replace=True)['Utterances'].values
            sampled_dialogues.extend([(di, label) for di in dialogues])
    
    USR = "<|start_header_id|>user<|end_header_id|>\n\n"
    EOT = "<|eot_id|>"
    ASSISTATNT = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    sampled_dialogues = [USR+dialogue+EOT+ASSISTATNT+label+EOT for dialogue, label in sampled_dialogues]
    # Combine the sampled dialogues into a single prompt
    random.shuffle(sampled_dialogues)
    prompt = '\n\n'.join(sampled_dialogues)
    prompt = prefix + prompt
    
    os.makedirs(os.path.dirname(ph.get_target_dir(f"modeling/prompts/{filename}")), exist_ok=True)
    with open(ph.get_target_dir(f"modeling/prompts/{filename}"), "w") as f:
        f.write(prompt)
    return prompt

    
def get_path_to_best_weights(dir_path:str) -> str:
    """
    Description:
        Returns the path to the newest and best weights from the specified directory.
    Args:
        dir_path: str
    Returns:
        path: str
    """
    sub_dirs = glob.glob(os.path.join(dir_path, "*"))
    sub_dirs.sort(key=lambda x: datetime.datetime.strptime(os.path.basename(x), "%m-%d-%Y_%H:%M"), reverse=True)
    newest_dir = sub_dirs[0]
    best_path = os.path.join(newest_dir, "best.pth.tar")
    return best_path

def get_loss_weights(target:pd.Series, tokenizer, vocabulary_size:int) -> torch.Tensor:
    """
    Description:
        Returns the loss weights for the specified target column.
    Args:
        data: pd.DataFrame
        target_col: str
        tokenizer: transformers.AutoTokenizer
        vocabulary_size: int
    Returns:
        weights: torch.Tensor of shape (vocabulary_size,)
    """
    labels = target.values.tolist()
    labels = [tokenizer.encode(x, add_special_tokens=False) for x in labels]
    labels = [item for sublist in labels for item in sublist] # Flatten the labels
    unique_labels = np.unique(labels)
    w = compute_class_weight("balanced", classes=unique_labels, y=labels)
    weights = torch.ones(vocabulary_size)
    for token_id, weight in zip(unique_labels, w):
        weights[token_id] = weight
    assert len(weights) == vocabulary_size, (len(weights), vocabulary_size)
    return weights
