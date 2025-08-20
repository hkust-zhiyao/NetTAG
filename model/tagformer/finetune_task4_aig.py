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
from sklearn.metrics import classification_report

date ='pretrain_net_align_7B_1024'


ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])


def train():
    embed, real = [], []
    feat = []
    with open(f"../../data_collect/data_js_gnnre/train_list.json", 'r') as f:
        design_lst = json.load(f)
    for design in design_lst:
        with open(f"{embed_save_dir}/embeds_{design}.npy", 'rb') as f:
            embed_lst = np.load(f)
        with open(f"{embed_save_dir}/reals_{design}.npy", 'rb') as f:
            real_lst = np.load(f)

        print(embed_lst.shape, real_lst.shape)
        embed.extend(embed_lst)
        real.extend(real_lst)
    embed_lst = np.array(embed)
    real_lst = np.array(real)
    print(embed_lst.shape, real_lst.shape)
    print('---- Training ----')
    model = XGBClassifier(n_estimators=1000, max_depth=100, n_jobs=96)
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
    with open(f"../../data_collect/data_js_gnnre/test_list.json", 'r') as f:
        design_lst = json.load(f)
    for design in design_lst:
        with open(f"{embed_save_dir}/embeds_{design}.npy", 'rb') as f:
            embed_lst = np.load(f)
        with open(f"{embed_save_dir}/reals_{design}.npy", 'rb') as f:
            real_lst = np.load(f)

        pred_lst = model.predict(embed_lst)
        y_true_set = set(real_lst) | set(pred_lst)
        print(design)
        # classify_metrics(pred_lst, real_lst)

        target_names = []
        if 0 in y_true_set:
            target_names.append('adder')
        if 1 in y_true_set:
            target_names.append('multiplier')
        if 2 in y_true_set:
            target_names.append('subtractor')
        if 3 in y_true_set:
            target_names.append('comparator')
        if 4 in y_true_set:
            target_names.append('control')

        # rpt_dct = classification_report(real_lst, pred_lst, target_names=target_names, output_dict=True)
        rpt = classification_report(real_lst, pred_lst, target_names=target_names)
        print(rpt)
    

if __name__ == '__main__':
    epoch = 5
    global task
    task = "task4_aig"

    embed_save_dir = f"./embeds/{date}_{epoch}/{task}"

    ft_model_save_dir = f"./finetune_model/{date}_{epoch}"
    if not os.path.exists(ft_model_save_dir):
        os.mkdir(ft_model_save_dir)
    
    

    
    model = train()
    with open(f"{ft_model_save_dir}/{task}.pkl", 'rb') as f:
        model = pickle.load(f)
    test(model)