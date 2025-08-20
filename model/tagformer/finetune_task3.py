import argparse
import os, json, pickle
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import ruamel_yaml as yaml
from accelerate import Accelerator
import torch
from transformers.optimization import (
    AdamW,
    get_polynomial_decay_schedule_with_warmup,
)
import numpy as np
from dataset_proc.load_dataset import load_train_valid_dataset_finetune, load_test_dataset_finetune_one_design

from models.model_pretrain import RTL_Fusion
from models.model_net import Net_Encoder

from accelerate import DistributedDataParallelKwargs
from xgboost import XGBRegressor, XGBClassifier
### sklearn mlp
from sklearn.neural_network import MLPRegressor, MLPClassifier

from torch.utils.tensorboard import SummaryWriter  
from utils.eval import regression_metrics, classify_metrics

date ='pretrain_net_align_7B_1024'


ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])


def train():
    embed, real = [], []
    with open(f"../../data_collect/data_js/train_list.json", 'r') as f:
        design_lst = json.load(f)
    # design_lst = ['b22_1', 'b20', 'b14_1', 'b14']
    for design in design_lst:
        if not os.path.exists(f"{embed_save_dir}/embeds_{design}.npy"):
            continue
        with open(f"{embed_save_dir}/embeds_{design}.npy", 'rb') as f:
            embed_lst = np.load(f)
        with open(f"{embed_save_dir}/reals_{design}.npy", 'rb') as f:
            real_lst = np.load(f)
        embed.extend(embed_lst)
        real.extend(real_lst)
    embed_lst = np.array(embed)
    real_lst = np.array(real)

    print('---- Training ----')
    model = XGBClassifier(n_estimators=500, max_depth=100, scale_pos_weight=500)
    model.fit(embed_lst, real_lst)
    print('---- Training Finish ----')

    # print(embed_lst)
    # print(real_lst)
    # exit()

    with open(f"{ft_model_save_dir}/{task}.pkl", 'wb') as f:
        pickle.dump(model, f)

    return model

def test(model):
    with open(f"../../data_collect/data_js/test_list.json", 'r') as f:
        design_lst = json.load(f)
    for design in design_lst:
        if not os.path.exists(f"{embed_save_dir}/embeds_{design}.npy"):
            continue
        with open(f"{embed_save_dir}/embeds_{design}.npy", 'rb') as f:
            embed_lst = np.load(f)
        with open(f"{embed_save_dir}/reals_{design}.npy", 'rb') as f:
            real_lst = np.load(f)
        pred_lst = model.predict(embed_lst)


        n_0 = len(real_lst[real_lst == 0])
        n_1 = len(real_lst[real_lst == 1])
        print(n_0, n_1)
        
        print(design)
        # print(pred_lst)
        # input()
        classify_metrics(pred_lst, real_lst)

    

if __name__ == '__main__':
    epoch = 5
    global task
    task = "task3"

    embed_save_dir = f"./embeds/{date}_{epoch}/{task}"

    ft_model_save_dir = f"./finetune_model/{date}"
    if not os.path.exists(ft_model_save_dir):
        os.mkdir(ft_model_save_dir)
    

    
    # model = train()
    with open(f"{ft_model_save_dir}/{task}.pkl", 'rb') as f:
        model = pickle.load(f)
    test(model)