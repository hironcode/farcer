import torch
from modeling import util
# from modeling.eval import eval, eval_acc_f1
# import os
# from modeling.train import train_modality_projector
from src.config import PathHelper

import pandas as    pd
import os
ph = PathHelper()


import torch
from torch import optim

from modeling.model import FarcerModel, ParamsFarcer
from modeling import util
from src import features, config, dataset
from modeling.train import train_epochs_modality_projector

# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# # Load model
# params = ParamsFarcer()

# if torch.cuda.is_available():
#     print(f"AVILABLE CUDA DEVICE COUNT: {torch.cuda.device_count()}")

# params.lm_quantize = None
# params.num_mp_layers = 1
# params.vm_only_use_cls_token = True

# model = FarcerModel(params)
# state_dict = torch.load(ph.get_target_dir("farser/best_single_history.pt"), map_location="cuda")['model_state_dict']
# state_dict = {k.replace("mp.", ""): v for k, v in state_dict.items()}
# model.mp.load_state_dict(state_dict)
# print("Model loaded")


if __name__ == "__main__":
    p = dataset.Processor()
    # p.process_iempcap(update=True)
    # for i in [3, 7]:
    #     util.create_few_shots(i, filename=f"testing/IEMOCAP/prompt_cls_{i}shots_cls.txt", image_token="<image feature token>", SEED=i, history=False, dataset="IEMOCAP")
    #     util.create_few_shots(i, filename=f"testing/IEMOCAP/prompt_cls_{i}shots_llama.txt", image_token="", SEED=i, history=False, dataset="IEMOCAP")
    #     util.create_few_shots(i, filename=f"testing/IEMOCAP/prompt_cls_{i}shots_cls_history.txt", image_token="<image feature token>", SEED=i, history=True, dataset="IEMOCAP")
    #     util.create_few_shots(i, filename=f"testing/IEMOCAP/prompt_cls_{i}shots_llama_history.txt", image_token="", SEED=i, history=True, dataset="IEMOCAP")

    # for i in range(10):
    #     util.create_few_shots(7, filename=f"random_shots_cls_iemocap/prompt_cls_7shots_cls{i}.txt", image_token="<image feature token>", SEED=i, history=False, dataset="IEMOCAP")
    #     util.create_few_shots(7, filename=f"random_shots_cls_iemocap_history/prompt_cls_7shots_cls{i}.txt", image_token="<image feature token>", SEED=i, history=True, dataset="IEMOCAP")

    # from src import features, dataset
    # p = dataset.Processor()
    # df = pd.read_csv(ph.get_target_dir("data/processed/IEMOCAP/train/train_sent_emo_div.csv"))
    # # while any emotion labels are 1, keep shuffling
    # lastvalues = df.groupby("Dialogue_ID").last()['Emotion'].value_counts()
    # values = df['Emotion'].value_counts()
    # print(lastvalues)
    # print(values)
    