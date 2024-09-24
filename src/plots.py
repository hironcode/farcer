from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

try:
    from src.config import PathHelper
except:
    from config import PathHelper

ph = PathHelper()


PATH = ph.get_target_dir("reports")

def draw_loss_line_graph(current_datetime, filename, train_loss:np.ndarray, eval_loss:Optional[np.ndarray]=None, loss_type="CrossEntropyLoss"):
    fig, ax = plt.subplots(figsize=(20, 20))
    epochs = range(len(train_loss))
    if eval_loss is not None:
        assert len(train_loss) == len(eval_loss), (len(train_loss), len(eval_loss))
        
    # Add extra margin on the top
    ax.margins(0, 0.1)


    ax.set_xlabel("Epoch")
    ax.set_ylabel(loss_type)

    ax.set_title(f"{loss_type} each Epoch")
    ax.set_aspect('auto')
    ax.grid(True)

    if eval_loss is not None:
        concat = np.concatenate([train_loss, eval_loss])
        ran = max(concat) - min(concat)
        ax.set_ylim(min(concat), max(concat)+2)
        ax.set_xlim(-0.2, len(train_loss))

    train_color = "blue"
    train_label = f"Train {loss_type}"
    ax.plot(epochs, train_loss, color=train_color, label=train_label, marker="o")


    if eval_loss is not None:
        eval_color = "red"
        eval_label = f"Val {loss_type}"
        ax.plot(epochs, eval_loss, color=eval_color, label=eval_label, marker="o")
        diff = (eval_loss - train_loss)/ran * 100
        for i, d in enumerate(diff):
            if d > 0:
                ax.text(i-0.2, max(concat)+0.5, f"+{d:.2f}%\nloss", fontsize=8, color="red")
            else:
                ax.text(i-0.2, max(concat)+0.5, f"-{-d:.2f}%\nloss", fontsize=8, color="green")
    
    
    ax.legend(loc="best")
    fig.tight_layout()

    path = f"{PATH}/training/{current_datetime}/{filename}.png"
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    plt.savefig(path)
    
    
    
def draw_farser_vs_llama(current_datetime, loss_dict:dict[str: np.ndarray], loss_type="CrossEntropyLoss"):
    fig, ax = plt.subplots(figsize=(20, 20))
    x = list(loss_dict.keys())
    y = [loss.mean() for loss in loss_dict.values()]
    yerr = [loss.std() for loss in loss_dict.values()]

    ax.set_xlabel(loss_type)
    ax.set_ylabel("Models")

    ax.set_title(f"{loss_type} each batch")
    ax.set_aspect('auto')
    ax.grid(True)

    colors = ["red", "blue", "green", "orange", "purple"]  # Add more colors if needed
    labels = [f"{model} {loss_type}" for model in list(loss_dict.keys())]
    
    ax.barh(x, y, color=colors[:len(loss_dict)], label=labels)
    ax.errorbar(y, x, xerr=yerr, fmt='.', color='Black', elinewidth=2, capthick=10, errorevery=1, alpha=0.5, ms=4, capsize=2)
    ax.grid(color ='grey',
        linestyle ='-.', linewidth = 0.5,
        alpha = 0.2)
    
    for i in ax.patches:
        plt.text(i.get_width()+0.05, i.get_y()+0.5, 
             str(round((i.get_width()), 2)),
             fontsize = 10, fontweight ='bold',
             color ='grey')

    ax.legend(loc="best")
    fig.tight_layout()
    
    path = f"{PATH}/evaluation/{current_datetime}/farser_vs_llama_loss_graph.png"
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    plt.savefig(path)
    

if __name__ == "__main__":
    result = {
        "train": np.random.randint(0, 10, 5),
        "valid": np.random.randint(0, 10, 5)
    }
    print(result['train'].mean())
    print(result['valid'].mean())
    draw_loss_line_graph("test", result['train'], result['valid'])
