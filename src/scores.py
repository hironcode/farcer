from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from config import PathHelper
import json
import numpy as np
import pprint
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

ph = PathHelper()

def MELD():
    reports = {
        "LLaMA3-Instruct": "reports/evaluation/eval_folder_name",
        "LLaMA3-Instruct PL":"reports/evaluation/eval_folder_name",
        "Base FARCER": "reports/evaluation/eval_folder_name",
        "Base FARCER PL": "reports/evaluation/eval_folder_name",
        "Three-layer FARCER": "reports/evaluation/eval_folder_name",
        "Three-layer FARCER PL": "reports/evaluation/eval_folder_name",
        "Fine-tuned FARCER": "reports/evaluation/eval_folder_name",
        "Fine-tuned FARCER PL": "reports/evaluation/eval_folder_name",
    }
    
    scores = {}

    latex = ""

    for key, value in reports.items():
        # if "PL" in key:
        #     continue
        for i in [3, 7]:
            path = ph.get_target_dir(f"{value}/f1_acc_result_{i}shot.json")
            with open(path, 'r') as j:
                results = json.load(j)
            y_true = results['y_true']
            y_pred = results['y_pred']
            labels = ['neutral', 'joy', 'surprise', 'anger', 'sadness','disgust', 'fear']
            f1s = f1_score(y_true, y_pred, labels=labels, average=None)
            w_f1 = results['f1']
            acc = accuracy_score(y_true, y_pred)
            precs = precision_score(y_true, y_pred, labels=labels, average=None)
            recs = recall_score(y_true, y_pred, labels=labels, average=None)
            print(" ")
            print(f"{key} {i}shot")
            if i == 3:
                scores[key] = {}
            scores[key][f"{i}shot"] = {}

            latex += f"{key} - {i} shot"

            for label, f1, prec, rec in zip(labels, f1s, precs, recs):
                print(f"{label}: f1 -> {f1}")
                latex += f" & {round(f1*100, 2)}"
                scores[key][f"{i}shot"][label] = {"f1": f1}
            print(f"Accuracy: {round(acc*100, 2)}")
            print(f"Weighted F1: {round(w_f1*100, 2)}")
            latex += f" & {round(acc*100, 2)} & {round(w_f1*100, 2)} \\\\ \n"

            cm_path = ph.get_target_dir(f"reports/confusion_matrix/MELD/{key}_{i}shot_cm.png")
            draw_confusion_matrix(y_true, y_pred, labels, cm_path, title=f"MELD {key} {i} shot")
    print(latex)    
    scores = pprint.pformat(scores, compact=True).replace("'",'"')

    txt_path = ph.get_target_dir("reports/f1scores_MELD.json")
    with open(txt_path, 'w') as f:
        f.write(scores)

def draw_confusion_matrix(y_true, y_pred, labels, path_to_save, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    fig, ax = plt.subplots(figsize=(20, 20))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           title=title,
           ylabel="True label",
           xlabel="Predicted label")
    ax.yaxis.label.set_size(30)  # Increase ylabel font size
    ax.xaxis.label.set_size(30)  # Increase xlabel font size
    ax.title.set_size(40)  # Increase title font size
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor", fontsize=28)  # Increase x label font size
    plt.setp(ax.get_yticklabels(), fontsize=28)  # Increase y label font size
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, round(cm[i, j], 2),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2. else "black",
                    fontsize=28)
    fig.tight_layout()
    plt.savefig(path_to_save)
    # plt.show()

def IEMOCAP():
    # picture unfixed
    reports = {
        "LLaMA3-Instruct": "reports/evaluation/eval_folder_name",
        "Base FARCER": "reports/evaluation/eval_folder_name",
        "Three-layer FARCER": "reports/evaluation/eval_folder_name",
        "Fine-tuned FARCER": "reports/evaluation/eval_folder_name",
        
        "LLaMA3-Instruct PL":"reports/evaluation/eval_folder_name",
        "Base FARCER PL":"reports/evaluation/eval_folder_name",
        "Three-layer FARCER PL": "reports/evaluation/eval_folder_name",
        "Fine-tuned FARCER PL": "reports/evaluation/eval_folder_name",
    }
    # そのまま
    reports = {
        "LLaMA3-Instruct": "reports/evaluation/eval_folder_name",
        "Base FARCER": "reports/evaluation/eval_folder_name",
        "Three-layer FARCER": "reports/evaluation/eval_folder_name",
        "Fine-tuned FARCER": "reports/evaluation/eval_folder_name",
        
        "LLaMA3-Instruct PL":"reports/evaluation/eval_folder_name",
        "Base FARCER PL": "reports/evaluation/eval_folder_name",
        "Three-layer FARCER PL": "reports/evaluation/eval_folder_name",
        "Fine-tuned FARCER PL": "reports/evaluation/eval_folder_name",
    }
    # transfer
    reports = {
        "LLaMA3-Instruct": "reports/evaluation/eval_folder_name",
        "Base FARCER": "reports/evaluation/eval_folder_name",
        "Three-layer FARCER": "reports/evaluation/eval_folder_name",
        "Fine-tuned FARCER": "reports/evaluation/eval_folder_name",
        
        "LLaMA3-Instruct PL": "reports/evaluation/eval_folder_name",
        "Base FARCER PL": "reports/evaluation/eval_folder_name",
        "Three-layer FARCER PL": "reports/evaluation/eval_folder_name",
        "Fine-tuned FARCER PL": "reports/evaluation/eval_folder_name",
    }
    
    scores = {}

    latex = ""

    for key, value in reports.items():
        for i in [3, 7]:
            path = ph.get_target_dir(f"{value}/f1_acc_result_{i}shot.json")
            with open(path, 'r') as j:
                results = json.load(j)
            y_true = results['y_true']
            y_pred = results['y_pred']
            labels = ['frustration', 'neutral', 'anger', 'sadness','excitement', 'happiness']
            f1s = f1_score(y_true, y_pred, labels=labels, average=None)
            w_f1 = results['f1']
            acc = accuracy_score(y_true, y_pred)
            precs = precision_score(y_true, y_pred, labels=labels, average=None)
            recs = recall_score(y_true, y_pred, labels=labels, average=None)
            print(" ")
            print(f"{key} {i}shot")
            if i == 3:
                scores[key] = {}
            scores[key][f"{i}shot"] = {}

            latex += f"{key} - {i} shot"
            if "LLaMA" in key:
                continue
            
            for label, f1, prec, rec in zip(labels, f1s, precs, recs):
                if label != "happiness":
                    continue
                print(f"{label}: f1 -> {round(f1*100, 2)}")
                print(f"{label}: prec -> {round(prec*100, 2)}")
                print(f"{label}: rec -> {round(rec*100, 2)}")
                latex += f" & {round(f1*100, 2)}"
                scores[key][f"{i}shot"][label] = {"f1": f1}
            print(f"Accuracy: {round(acc*100, 2)}")
            print(f"Weighted F1: {round(w_f1*100, 2)}")
            latex += f" & {round(acc*100, 2)} & {round(w_f1*100, 2)} \\\\ \n"

            cm_path = ph.get_target_dir(f"reports/confusion_matrix/IEMOCAP/{key}_{i}shot_cm.png")
            # draw_confusion_matrix(y_true, y_pred, labels, cm_path, title=f"IEMOCAP {key} {i} shot")
    print(latex)    
    scores = pprint.pformat(scores, compact=True).replace("'",'"')

    txt_path = ph.get_target_dir("reports/f1scores_IEMOCAP.json")
    # with open(txt_path, 'w') as f:
    #     f.write(scores)

if __name__ == "__main__":
    # MELD()
    IEMOCAP()


