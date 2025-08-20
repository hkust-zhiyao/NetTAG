import argparse
import os, json
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
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

from torch.utils.tensorboard import SummaryWriter  


date ='pretrain_net_align_7B_1024'


ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])


def infer_one_design(net_enc, design):
    print('Current Design: ', design)
    if design == 'vga_lcd':
        return
    if os.path.exists(f"{embed_save_dir}/embeds_{design}.npy"):
        return
    train_net_loader = load_test_dataset_finetune_one_design(batch_size=8, design=design, task=task)

    embed_lst, real_lst = [], []
    train_net_loader, net_enc = accelerator.prepare(train_net_loader, net_enc)
    if len(train_net_loader) == 0:
        return

    with torch.no_grad():
        for idx, data in enumerate(train_net_loader):
            if not data:
                continue
            _, graph_embeds = net_enc(data, mode='infer')
            # graph_pred = net_enc.downstream_mlp(graph_embeds)
            graph_embeds = graph_embeds.detach().cpu().numpy()
            feat = data.y0.detach().cpu().numpy().reshape(-1,1)
            graph_embeds = np.concatenate((graph_embeds, feat), axis=1)
            
            embed_lst.extend(graph_embeds)
            real = np.concatenate((data.y1.detach().cpu().numpy().reshape(-1,1), data.y2.detach().cpu().numpy().reshape(-1,1)), axis=1)
            real_lst.extend(real)

    with open(f"../../dataset/design/{design}_vec.npy", 'rb') as f:
        vec_arr = np.load(f)
    
    embed_lst = np.array(embed_lst)
    embed_lst = np.concatenate((embed_lst, np.tile(vec_arr, (embed_lst.shape[0], 1))), axis=1)
    real_lst = np.array(real_lst)

    print(embed_lst.shape, real_lst.shape)

    with open(f"{embed_save_dir}/embeds_{design}.npy", 'wb') as f:
        np.save(f, embed_lst)
    with open(f"{embed_save_dir}/reals_{design}.npy", 'wb') as f:
        np.save(f, real_lst)




def enc2vec(net_enc):
    with open(f"../../data_collect/data_js/design_list.json", 'r') as f:
        design_lst = json.load(f)
    for design in design_lst:
        infer_one_design(net_enc, design)



if __name__ == '__main__':
    epoch = 5
    global task
    task = "task1"

    embed_save_dir = f"./embeds/{date}_{epoch}"
    if not os.path.exists(embed_save_dir):
        os.makedirs(embed_save_dir)
    embed_save_dir = f"{embed_save_dir}/{task}"
    if not os.path.exists(embed_save_dir):
        os.makedirs(embed_save_dir)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/PretrainStage.yaml')

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    net_enc = Net_Encoder(config=config, device=accelerator.device, accelerator=accelerator)
    model_save_dir = f"./pretrain_model/{date}"
    model_save_path = f"{model_save_dir}/net_enc.{epoch}.pt"
    net_enc.load_state_dict(torch.load(model_save_path, weights_only=True))
    net_enc.eval()

    ## get model size


    # enc2vec_training(net_enc)
    enc2vec(net_enc)