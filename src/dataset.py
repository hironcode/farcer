import pandas as pd
import sys
import os
from pathlib import Path
from wasabi import msg
from datasets import Dataset
import requests
import cv2
from collections import defaultdict
from tqdm import tqdm
import subprocess
import datetime
import re
import random

# Local imports
# ".config" explicitly specifies that the config.py file is in the same directory as the dataset.py file
try:
    from .config import PathHelper
except:
    from config import PathHelper
import tarfile
import re


ph = PathHelper()      
    

class Processor:
    def __init__(self) -> None:
        self.datapath = ph.get_target_dir("data")
        self.scrshot_failed = []
        
    def add_path_MELD(self, mode='train', update=False) -> pd.DataFrame:
        """
        Creates a new CSV file with the mp4_path column added to the original CSV file.\n
        Also creates a new HuggingFace dataset from the new CSV file.\n

        mode: str
        update: bool
        """
        if mode == 'train':
            csv_path = "train/train_sent_emo.csv"
            mp4_path = "train/train_splits"
        elif mode == 'dev':
            csv_path = "dev_sent_emo.csv"
            mp4_path = "dev/dev_splits_complete"
        else:
            csv_path = "test_sent_emo.csv"
            mp4_path = "test/output_repeated_splits_test"

        csv_path = os.path.join(self.datapath, f"external/MELD/MELD.Raw/{csv_path}")
        mp4_path = os.path.join(self.datapath, f"external/MELD/MELD.Raw/{mp4_path}")
        
        df  = pd.read_csv(csv_path)
        
        def concat_mp4(df):
            df['mp4_path'] = df[['Dialogue_ID', "Utterance_ID"]].apply(lambda x: os.path.join(mp4_path, f"dia{x[0]}_utt{x[1]}.mp4"), axis=1)
            return df
        
        # check the data type
        df['StartTime'] = pd.to_datetime(df['StartTime'], format='%H:%M:%S,%f')
        df['EndTime'] = pd.to_datetime(df['EndTime'], format='%H:%M:%S,%f')
        diff = df['EndTime'] - df['StartTime']
        df['CaptureTime'] = (diff.dt.seconds + diff.dt.microseconds * 0.000001)/2
        
        df = concat_mp4(df)
        
        # save this updated file in data/processed/MELD folder
        new_path_csv = Path(os.path.join(self.datapath, f"interim/MELD/{mode}/{mode}_sent_emo.csv"))
        if not os.path.exists(new_path_csv) or update is True:
            if not os.path.exists(new_path_csv):
                os.makedirs(new_path_csv.parent)
            df.to_csv(new_path_csv)
            msg.info(f"Saved the new CSV at: {new_path_csv}")


        # create a new HuggingFace dataset
        # dataset = Dataset.from_pandas(df)

        # new_path_dataset = os.path.join(self.datapath, f"processed/MELD/{mode}/{mode}_sent_emo_hf_dataset")
        # if not os.path.exists(new_path_dataset) or update is True:
        #     if not os.path.exists(new_path_dataset):
        #         os.makedirs(new_path_dataset)
        #     dataset.save_to_disk(new_path_dataset)
        #     msg.info(f"Saved the new Hugging Face dataset at: {new_path_dataset}")

        return df
    
    def take_screenshot(self, video_path:str, sec, screenshot_path, side=None, overwrite=False):
        """
        Takes a screenshot from an mp4 file at the specified time.
        video_path: Path to the mp4 file
        sec: Time in seconds to take the screenshot
        screenshot_path: Path to save the screenshot
        side: side of the video to take the screenshot from. If this is provided, it assume that we take screenshots from IEOMCAP dataset
        """

        if os.path.exists(screenshot_path) and overwrite is False:
            return
            # p.add_path_MELD('train', update=True)
            # p.add_path_MELD('dev', update=True)
            # p.add_path_MELD('test', update=True)

        cap = cv2.VideoCapture(video_path)
        # Get the length of the video
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Calculate the frame number to capture
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        if length/fps <= 0.5:
            output_path = video_path.replace(video_path.split(".")[0], video_path.split(".")[0]+'_preprocessed').replace('external', 'interim')
            dir = os.path.dirname(output_path)
            if not os.path.exists(dir):
                os.makedirs(dir)
            self.preprocess_short_video(video_path, output_path)
            video_path = output_path
            sec = 0
    
        cap = cv2.VideoCapture(video_path)
    
        frame_number = int(fps * sec)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = cap.read()
        
        cap.release()
        
        if success:
            height, width, _ = frame.shape
            if side == "left":
                # also crop the screen
                frame = frame[height//5 : -height//5, :width//2]
            elif side == "right":
                frame = frame[height//5 : -height//5, frame.shape[1]//2:]
            else:
                pass

        compression_params = [cv2.IMWRITE_JPEG_QUALITY, 50]

        if success:
            # Save the frame as an image
            cv2.imwrite(screenshot_path, frame, compression_params)
            # print(f"Screenshot saved at {screen_path}", end='\r')
        else:
            # print("Failed to capture the frame", end='\r')
            
            self.scrshot_failed.append(screenshot_path.split('/')[-1])

        cap, fps, frame_number, frame, success, screenshot_path = None, None, None, None, None, None

    def preprocess_short_video(self, input_path, output_path):
        if os.path.exists(output_path):
            return

        # (
        #     ffmpeg
        #     .input(input_path)
        #     .output(output_path, c="copy", movflags='faststart')
        #     .run(overwrite_output=True)
        # )

        ffmpeg_command = [
            'ffmpeg',
            '-i', input_path,
            '-movflags', 'faststart',
            '-c', 'copy',
            output_path
        ]
        subprocess.run(ffmpeg_command, check=True, stdout=subprocess.DEVNULL)

    def reset_failed_screenshots(self):
        self.scrshot_failed = []

    def screenshots_MELD(self, mode, update=False, start=0, end=None) -> list:
        """
        Combines different utterances of a dialogue into a single row and
        concatenates screenshots of each utterance's mp4 as a single sequence.
        mode: str (train, dev, test)
        """
        self.reset_failed_screenshots()

        csv_path = os.path.join(ph.get_target_dir('data'), f"processed/MELD/{mode}/{mode}_sent_emo.csv")
        df = pd.read_csv(csv_path)


        scr_path = os.path.join(ph.get_target_dir('data'), f"processed/MELD/{mode}/screenshots")
        if not os.path.exists(scr_path):
            os.makedirs(scr_path)

        if update is True:
            # save the screenshot paths
            tqdm.pandas()
            
            df['image_path'] = df[['Dialogue_ID', 'Utterance_ID']].apply(lambda x: os.path.join(scr_path, f"dia{x[0]}_utt{x[1]}.jpeg"), axis=1)
            # take screenshots of each utterance
            df[['mp4_path', 'CaptureTime', 'image_path']].iloc[start:end].progress_apply(lambda x: self.take_screenshot(x[0], x[1], x[2]), axis=1)

            # save the updated CSV
            df.to_csv(csv_path)

            print(f"Failed to save {len(self.scrshot_failed)} screenshots")
        return self.scrshot_failed
    
    def _create_preprocessed_videos_on_Ubuntu(self, mode, threshold=0.5):
        csv_path = os.path.join(ph.get_target_dir('data'), f"interim/MELD/{mode}/{mode}_sent_emo.csv")
        df = pd.read_csv(csv_path)

        def preprocess(mp4_path, sec):
            output_path = mp4_path.replace('.mp4', '_preprocessed.mp4').replace('external', 'interim')
            dir = os.path.dirname(output_path)
            if not os.path.exists(dir):
                os.makedirs(dir)
            self.preprocess_short_video(mp4_path, output_path)

        def take_screenshot_demo(mp4_path, sec):
            if sec < threshold:
                preprocess(mp4_path, sec)
            else:
                cap = cv2.VideoCapture(mp4_path)
                # Calculate the frame number to capture
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_number = int(fps * sec)

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                success, frame = cap.read()
                if success is False:
                    preprocess(mp4_path, sec)

        
        tqdm.pandas()
        df[['mp4_path', 'CaptureTime']].progress_apply(lambda x: take_screenshot_demo(x[0], x[1]), axis=1)

    def process_iempcap(self):
        """
        Processes the iemocap dataset and saves the processed data in the data/processed/iemocap folder.
        """
        path = os.path.join(self.datapath, "external/IEMOCAP/IEMOCAP_full_release")
        sessions  = [1, 2, 3, 4, 5]
        # thanks to https://github.com/Aditya3107/IEMOCAP_EMOTION_Recognition/blob/master/1_extract_emotion_labels.ipynb
        emo_regex = re.compile(r'\[.+\]\n', re.IGNORECASE)
        alt_emo_regex = re.compile(r'C-E\d{1}:\t.+\n', re.IGNORECASE)

        def impro_config(x):
            if "a" in x or "b" in x:
                return float('inf')
            else:
                return int(x.split('_')[-1].split('.')[0][-1])
        
        def script_config(x):
            if "a" in x or "b" in x:
                return (int(x.split('_')[1][-1]), float('inf'))
            else:
                return (int(x.split('_')[1][-1]), int(x.split('_')[2].split('.')[0][-1]))

        # for each session
        for i in sessions:
            text_path = os.path.join(path, f"Session{i}/dialog/transcriptions")
            emo_path = os.path.join(path, f"Session{i}/dialog/EmoEvaluation")
            vd_path = os.path.join(path, f"Session{i}/dialog/avi/DivX")

            text_files = [f for f in os.listdir(text_path) if f.endswith('.txt') and not f.startswith('.')] # Ses\d{2}\w_script\d{2}_\d.txt or Ses\d{2}\w_impro\d{2}.txt
            emo_files = [f for f in os.listdir(emo_path) if f.endswith('.txt') and not f.startswith('.')]   # Ses\d{2}\w_script\d{2}_\d.txt or Ses\d{2}\w_impro\d{2}.txt
            vd_files = [f for f in os.listdir(vd_path) if f.endswith('.avi') and not f.startswith('.')]     # Ses\d{2}\w_script\d{2}_\d.txt or Ses\d{2}\w_impro\d{2}.avi
            # Sort the files in ascending order

            merger_impro = {
                title.split(".")[0]: [title] for title in text_files if "impro" in title
            }
            merger_scr = {
                title.split(".")[0]: [title] for title in text_files if "script" in title
            }

            for emo_file in emo_files:
                title = emo_file.split(".")[0]
                if "impro" in title:
                    merger_impro[title].append(emo_file)
                else:
                    merger_scr[title].append(emo_file)
            
            for vd_file in vd_files:
                title = vd_file.split(".")[0]
                if "impro" in title:
                    merger_impro[title].append(vd_file)
                else:
                    merger_scr[title].append(vd_file)

            print(merger_impro)
            print(merger_scr)
            impro_keys = sorted(list(merger_impro.keys()), key=impro_config)
            scr_keys = sorted(list(merger_scr.keys()), key=script_config)
            text_files.clear()
            emo_files.clear()
            vd_files.clear()
            for key in impro_keys + scr_keys:
                try:
                    txt, emo, vd = merger_impro[key]
                except:
                    txt, emo, vd = merger_scr[key]
                text_files.append(txt)
                emo_files.append(emo)
                vd_files.append(vd)

            print(text_files)
            print(emo_files)
            print(vd_files)


            # impro = []
            # script = []
            # for text_file in text_files:
            #     if 'impro' in text_file:
            #         impro.append(text_file)
            #     else:
            #         script.append(text_file)
            # impro.sort(key=impro_config)
            # script.sort(key=script_config)
            # text_files = impro + script

            # impro = []
            # script = []
            # for emo_file in emo_files:
            #     if 'impro' in emo_file:
            #         impro.append(emo_file)
            #     else:
            #         script.append(emo_file)
            # impro.sort(key=impro_config)
            # script.sort(key=script_config)
            # emo_files = impro + script

            # impro = []
            # script = []
            # for vd_file in vd_files:
            #     if 'impro' in vd_file:
            #         impro.append(vd_file)
            #     else:
            #         script.append(vd_file)
            # impro.sort(key=impro_config)
            # script.sort(key=script_config)
            # vd_files = impro + script


            result = {
                "Dialogue_ID": [],
                "Utterance_ID": [],
                "Speaker": [],
                "Utterance": [],
                "Emotion": [],
                "StartTime": [],
                "EndTime": [],
                "CaptureTime": [],
                "avi_path": []
            }

            # for each dialogue
            dialogue_id = 0
            for txtf, emof, vdf in zip(text_files, emo_files, vd_files):
                # init dict

                with open(os.path.join(text_path, txtf), 'r') as f:
                    utterances = f.read().strip()
                with open(os.path.join(emo_path, emof), 'r') as f:
                    emotionsf = f.read().strip()

                emotions = re.findall(emo_regex, emotionsf)[1:]   # remove the first line
                alt_emotions = re.findall(alt_emo_regex, emotionsf)
                assert len(alt_emotions)%3 == 0, len(alt_emotions)
                alt_emotions_pairs = []
                for j, e in enumerate(alt_emotions):
                    if j % 3 == 0:
                        alt_emotions_pairs.append([])
                    e:str = e.strip().split('\t')[1][:-1]
                    e = e.lower()
                    e = e[:3]
                    alt_emotions_pairs[-1].append(e)
            
                utterances = utterances.split('\n')
                # skip invalid utterances
                utterances = [u for u in utterances if u.startswith('Ses') and not "XX" in u.split(" ")[0]]
                
                # sort based on start time
                utterances.sort(key=lambda x: float(x.strip().split(" ")[1][1:-2].split("-")[0]))
                # concatenate emotions and alt_emotions_pairs and sort them together and detach them
                joined = list(zip(emotions, alt_emotions_pairs))
                joined.sort(key=lambda x: float(x[0].strip().split("\t")[0][1:-1].split("-")[0]))
                # emotions.sort(key=lambda x: float(x.strip().split("\t")[0][1:-1].split("-")[0]))
                emotions = [x[0] for x in joined]
                alt_emotions_pairs = [x[1] for x in joined]

                utterance_id = 0
                # for each emotion annotation
                for utterance, emotion in zip(utterances, emotions):
                    start_end_time, title, label, _ = emotion.strip().split('\t')  # ["[start_time - end_time]", "Ses??_impro/script??_F/M???", emotion_label, annotation]
                    if label == "xxx":
                        print(f"xxx found at {utterance_id} in {txtf}")
                        label = random.choice(alt_emotions_pairs[utterance_id])
                        if ";" in label:
                            temp = label.split(";")[0] if ";" in label else label
                            temp = label.split(";")[1] if temp == "other" else temp
                            label = temp
                        print(f"Replaced with {label}")
                    
                    utt = re.search(r':\s*(.*)', utterance).group(1)
                        
                    # make sure title is the same
                    assert title == utterance.strip().split(" ")[0], (title, utterance.split(" ")[0])

                    # create start and end time
                    start, end = map(float, start_end_time[1:-1].split(' - '))

                    # get seconds
                    capture_time = (end + start) / 2
                    
                    result["Dialogue_ID"].append(dialogue_id)
                    result["Utterance_ID"].append(utterance_id)
                    result["Speaker"].append(title.split('_')[-1][0])   # F or M
                    result["Utterance"].append(utt)
                    result["Emotion"].append(label)
                    result["StartTime"].append(start)
                    result["EndTime"].append(end)
                    result["CaptureTime"].append(capture_time)
                    result["avi_path"].append(os.path.join(vd_path, vdf))

                    utterance_id += 1
                dialogue_id += 1

            df = pd.DataFrame(result)
            # save the dataframe
            os.makedirs(os.path.join(self.datapath, f"interim/IEMOCAP/session{i}"), exist_ok=True)
            df.to_csv(os.path.join(self.datapath, f"interim/IEMOCAP/session{i}/session{i}_sent_emo.csv"))

    def screenshots_IEMOCAP(self, session:int, update=False, start=0, end=None, overwrite=False) -> list:
        """
        Combines different utterances of a dialogue into a single row and
        concatenates screenshots of each utterance's mp4 as a single sequence.
        mode: str (train, dev, test)
        """
        if session not in [1, 2, 3, 4, 5]:
            raise ValueError("Session number must be between 1 and 5")
        
        self.reset_failed_screenshots()

        csv_path = os.path.join(ph.get_target_dir('data'), f"interim/IEMOCAP/session{session}/session{session}_sent_emo.csv")
        df = pd.read_csv(csv_path)


        scr_path = os.path.join(ph.get_target_dir('data'), f"processed/IEMOCAP/session{session}/screenshots")
        if not os.path.exists(scr_path):
            os.makedirs(scr_path)

        def screenshot_config(row, overwrite=False):
            f = row["avi_path"].split("/")[-1]
            subj = re.search(r"Ses\d{2}(\w)", f).group(1)   # M or F
            if subj == row['Speaker']:
                # if the speaker is the subject (who is on the left side by default), take the screenshot from the left side
                self.take_screenshot(row['avi_path'], row['CaptureTime'], row['image_path'], side='left', overwrite=overwrite)
            else:
                # if the speaker is not the subject, take the screenshot from the right side
                self.take_screenshot(row['avi_path'], row['CaptureTime'], row['image_path'], side='right', overwrite=overwrite)

        if update is True:
            # save the screenshot paths
            tqdm.pandas()
            
            df['image_path'] = df[['Dialogue_ID', 'Utterance_ID']].apply(lambda x: os.path.join(scr_path, f"dia{x[0]}_utt{x[1]}.jpeg"), axis=1)
            # take screenshots of each utterance
            # if speaker is F (female): side is right
            df.iloc[start:end].progress_apply(screenshot_config, axis=1, overwrite=overwrite)

            # save the updated CSV
            csv_path = csv_path.replace('interim', 'processed')
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            df.to_csv(csv_path, index=False)

        print(f"Failed to save {len(self.scrshot_failed)} screenshots")
        print(f"Failed screenshots:\n{self.scrshot_failed}")
        return self.scrshot_failed

    def merge_sessions_IEMOCAP(self, sessions:list = [1, 2, 3, 4], mode='train', update=False) -> list:
        """
        Combines different utterances of a dialogue into a single row and
        concatenates screenshots of each utterance's mp4 as a single sequence.
        mode: str (train, dev, test)
        """
        cols = pd.read_csv(os.path.join(ph.get_target_dir('data'), f"processed/IEMOCAP/session{sessions[0]}/session{sessions[0]}_sent_emo.csv"), index_col=0).columns
        df = pd.DataFrame(columns=cols)
        for session in sessions:
            csv_path = os.path.join(ph.get_target_dir('data'), f"processed/IEMOCAP/session{session}/session{session}_sent_emo.csv")
            df = pd.concat([df, pd.read_csv(csv_path, index_col=0)], axis=0)

        save_path = os.path.join(ph.get_target_dir('data'), f"processed/IEMOCAP/{mode}/{mode}_sent_emo.csv")
        if not os.path.exists(save_path) or update is True:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_csv(save_path)

        print(f"Saved the merged IEMOCAP data at: {save_path}")
    
    def _create_preprocessed_video_IEMOCAP(self, session, threshold=0.5):
        csv_path = os.path.join(ph.get_target_dir('data'), f"interim/IEMOCAP/session{session}/session{session}_sent_emo.csv")
        df = pd.read_csv(csv_path)

        def preprocess(mp4_path, sec):
            output_path = mp4_path.replace('.avi', '_preprocessed.avi').replace('external', 'interim')
            dir = os.path.dirname(output_path)
            if not os.path.exists(dir):
                os.makedirs(dir)
            self.preprocess_short_video(mp4_path, output_path)

        def take_screenshot_demo(mp4_path, sec):
            if sec < threshold:
                preprocess(mp4_path, sec)
            else:
                cap = cv2.VideoCapture(mp4_path)
                # Calculate the frame number to capture
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_number = int(fps * sec)

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                success, frame = cap.read()
                if success is False:
                    preprocess(mp4_path, sec)

        
        tqdm.pandas()
        df[['avi_path', 'CaptureTime']].progress_apply(lambda x: take_screenshot_demo(x[0], x[1]), axis=1)
    
    def divide_dialogues_IEMOCAP(self, min_utt=5, max_utt=14, update=False, mode="train"):
        df = pd.read_csv(os.path.join(ph.get_target_dir('data'), f"processed/IEMOCAP/{mode}/{mode}_sent_emo.csv"))
        dialogue_ids = df['Dialogue_ID'].unique()

        #create temporary id column
        df['Temp_ID'] = df['Dialogue_ID'].copy()
        
        last_id = 0
        last_index = 0
        for id in dialogue_ids:
            num_utt = random.randint(min_utt, max_utt)
            size = len(df[df['Dialogue_ID'] == id])
            # skip if there is no need to divide the dialogue
            if size < num_utt:
                continue
            # get how many new sub-dialogues will be created
            new_dia_size = size // num_utt


            # for each sub-dialogue
            for i, new_id in enumerate(range(last_id, last_id + new_dia_size)):
                start_idx = last_index

                if i == new_dia_size - 1:
                    end_idx = start_idx + (size - i*num_utt) -1
                else:
                    end_idx = start_idx + num_utt - 1

                df.loc[start_idx:end_idx, 'Temp_ID'] = new_id
                df.loc[start_idx:end_idx, 'Utterance_ID'] = range(end_idx - start_idx + 1)
                last_index = end_idx + 1

            last_id = new_id + 1
        
        # replace the dialogue id with the new temp id
        df['Dialogue_ID'] = df['Temp_ID']
        df.drop(columns=['Temp_ID'], inplace=True)

        if update is True:
            df.to_csv(os.path.join(ph.get_target_dir('data'), f"processed/IEMOCAP/{mode}/{mode}_sent_emo_div.csv"), index=False)
        print("Divided the dialogues successfully.")
        return df
