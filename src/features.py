import pandas as pd
import sys
import os
from pathlib import Path
from wasabi import msg
from datasets import Dataset
from moviepy.editor import VideoFileClip
from PIL import Image
from collections import defaultdict
import json
from tqdm import tqdm
import numpy as np
import warnings

# Local imports
# ".config" explicitly specifies that the config.py file is in the same directory as the dataset.py file
try:
    from .config import PathHelper
except:
    from config import PathHelper
import cv2



ph = PathHelper()

def format_MELD_json(mode, update=False) -> dict:
    """
    Combines different utterances of a dialogue into a single row and
    concatenates screenshots of each utterance's mp4 as a single sequence.
    mode: str (train, dev, test)
    """
    csv_path = os.path.join(ph.get_target_dir('data'), f"processed/MELD/{mode}/{mode}_sent_emo.csv")
    df = pd.read_csv(csv_path)

    # prepare the sequence dictionary
    seq = defaultdict(dict)

    # Create the sequence
    for _, row in tqdm(df.iterrows()):
        dialogue_id = row['Dialogue_ID']
        utterance_id = row['Utterance_ID']
        speaker = row['Speaker']
        utterance = row['Utterance']
        emotion = row['Emotion']
        scrshot_path = row['image_path']
        capture_time = row['CaptureTime']

        seq[dialogue_id][utterance_id] = {}
        seq[dialogue_id][utterance_id]['speaker'] = speaker
        seq[dialogue_id][utterance_id]['utterance'] = utterance
        seq[dialogue_id][utterance_id]['screenshot'] = scrshot_path
        seq[dialogue_id][utterance_id]['emotion'] = emotion
        seq[dialogue_id][utterance_id]['capture_time'] = capture_time

    # Save the sequence as json if Update is true
    if update is True:
        json_path = os.path.join(ph.get_target_dir('data'), f"processed/MELD/{mode}/{mode}_seq.json")
        with open(json_path, 'w') as f:
            json.dump(seq, f)

    return seq


def format_corpus(dataset:str, mode:str, update=False) -> pd.DataFrame:
    """
    Args:
        mode: str (train, dev, test)
    Notes:
        Format the MELD dataframe in a desired format.
        (1) Make sure the data type of IDs are int.
        (2) If image did not exist, replace it with Nan.
        (3) Change the wording of emotion labels to (neutral, joy, ).
        Save the dataframe as csv if Update is true.
    """
    if mode not in ['train', 'dev', 'test']:
        raise ValueError("mode must be one of 'train', 'dev', 'test'.")
    elif dataset not in ['MELD', 'IEMOCAP']:
        raise ValueError("dataset must be one of 'MELD', 'IEMOCAP'.")
    
    if dataset == 'IEMOCAP':
        csv_path = os.path.join(ph.get_target_dir('data'), f"processed/{dataset}/{mode}/{mode}_sent_emo_div.csv")
    else:
        csv_path = os.path.join(ph.get_target_dir('data'), f"processed/{dataset}/{mode}/{mode}_sent_emo.csv")

    df = pd.read_csv(csv_path)

    # Create dataframe only with essential information
    df = df[['Dialogue_ID', 'Utterance_ID', "Utterance", 'Speaker', 'Emotion', 'CaptureTime', 'image_path']]
    df['Utterance_ID'] = df['Utterance_ID'].astype(int)
    df['Dialogue_ID'] = df['Dialogue_ID'].astype(int)
    df['image_path'] = df['image_path'].apply(lambda x: x if os.path.exists(x) else np.NaN)
    nones_ratio = df['image_path'].isna().sum() / len(df)

    if dataset == "IEMOCAP":
        emomap = {
            'ang': 'anger',
            'hap': 'happiness',
            'neu': 'neutral',
            'sad': 'sadness',
            'fru': 'frustration',
            'exc': 'excitement',
            'fea': 'fear',
            'sur': 'surprise',
            'dis': 'disgust',
            'oth': 'other',
        }
        df['Emotion'] = df['Emotion'].map(emomap)
        namemap = {"F": "Person A", "M": "Person B"}
        df['Speaker'] = df['Speaker'].map(namemap)

    if nones_ratio >= 0.1:
        msg.warn(f"""{nones_ratio*100}% of image paths do not exist in "{mode}". Please check the image paths.""")

    # Save the sequence as json if Update is true
    if update is True:
        csv_path = os.path.join(ph.get_target_dir('data'), f"processed/{dataset}/{mode}/{mode}_essential.csv")
        df.to_csv(csv_path)
    return df



if __name__ == "__main__":
    df = format_corpus('train', update=True)
    format_corpus('dev', update=True)
    format_corpus('test', update=True)
