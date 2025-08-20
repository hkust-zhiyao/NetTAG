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
from sklearn.neural_network import MLPRegressor, MLPClassifier
from torch.utils.tensorboard import SummaryWriter  
from utils.eval import regression_metrics, classify_metrics

date ='pretrain_net_align_7B_1024'


ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])


def train():
    embed, real = [], []
    with open(f"../../data_collect/data_js/ft_train_list_task2.json", 'r') as f:
        design_lst = json.load(f)
    for design in design_lst:
        if not os.path.exists(f"{embed_save_dir}/embeds_{design}.npy"):
            continue
        with open(f"{embed_save_dir}/embeds_{design}.npy", 'rb') as f:
            embed_lst = np.load(f)
        with open(f"../../data_collect/data_pt/{stage}/{design}/design.json", 'r') as f:
            design_dict = json.load(f)
        label = design_dict[task]
        with open(f"../../data_collect/data_pt/init/{design}/design.json", 'r') as f:
            design_dict_feat = json.load(f)
        feat = design_dict_feat[task]
        embed_ = np.mean(embed_lst, axis=0)
        # embed_ = embed_[-17:]
        embed_ = np.concatenate((embed_, np.array([feat])))
        # print(embed_)
        print(feat, label)
        embed.append(embed_)
        real.append(label)
    embed_lst = np.array(embed)
    real_lst = np.array(real)

    print('---- Training ----')
    model = XGBRegressor(n_estimators=50, max_depth=30)
    # model = MLPRegressor(hidden_layer_sizes=(256, 32), max_iter=500)
    model.fit(embed_lst, real_lst)
    print('---- Training Finish ----')
    # print(embed_lst)
    print(real_lst)
    # exit()

    with open(f"{ft_model_save_dir}/{task}.pkl", 'wb') as f:
        pickle.dump(model, f)

    return model

def test(model):
    with open(f"../../data_collect/data_js/ft_test_list_task2.json", 'r') as f:
        design_lst = json.load(f)
    embed, real = [], []
    for design in design_lst:
        if not os.path.exists(f"{embed_save_dir}/embeds_{design}.npy"):
            continue
        with open(f"{embed_save_dir}/embeds_{design}.npy", 'rb') as f:
            embed_lst = np.load(f)
        with open(f"../../data_collect/data_pt/{stage}/{design}/design.json", 'r') as f:
            design_dict = json.load(f)
        label = design_dict[task]
        with open(f"../../data_collect/data_pt/init/{design}/design.json", 'r') as f:
            design_dict_feat = json.load(f)
        feat = design_dict_feat[task]
        embed_ = np.mean(embed_lst, axis=0)
        # embed_ = embed_[-17:]
        embed_ = np.concatenate((embed_, np.array([feat])))
        embed.append(embed_)
        real.append(label)
        print(feat, label)
    pred_lst = model.predict(embed)
    print(pred_lst)
    
    print(design)
    regression_metrics(pred_lst, real)

    

if __name__ == '__main__':
    epoch = 5
    global task, stage
    task = "pwr"
    stage = "place_opt"

    embed_save_dir = f"./embeds/{date}_{epoch}/task1"

    ft_model_save_dir = f"./finetune_model/{date}"
    if not os.path.exists(ft_model_save_dir):
        os.mkdir(ft_model_save_dir)

    model = train()
    test(model)