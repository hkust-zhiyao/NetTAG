import argparse
import os, json
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import ruamel_yaml as yaml
from accelerate import Accelerator
import torch
from transformers.optimization import (
    AdamW,
    get_polynomial_decay_schedule_with_warmup,
)
import numpy as np
from dataset_proc.load_dataset import load_test_dataset_finetune_gnnre

from models.model_pretrain import RTL_Fusion
from models.model_net import Net_Encoder

from accelerate import DistributedDataParallelKwargs

from torch.utils.tensorboard import SummaryWriter  


date ='pretrain_net_align_7B_1024'


ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

# def enc2vec_training(net_enc):
#     train_net_loader = load_train_valid_dataset_finetune(batch_size=4, train_valid="demo", task=task)

#     embed_lst, real_lst = [], []
#     train_net_loader, net_enc = accelerator.prepare(train_net_loader, net_enc)
#     with torch.no_grad():
#         for idx, data in enumerate(train_net_loader):
#             _, graph_embeds = net_enc(data, mode='infer')
#             # graph_pred = net_enc.downstream_mlp(graph_embeds)
#             embed_lst.extend(graph_embeds.detach().cpu().numpy().tolist())
#             real_lst.extend(data.y.detach().cpu().numpy().tolist())
    
#     embed_lst = np.array(embed_lst)
#     real_lst = np.array(real_lst)

#     print(real_lst)
#     exit()

#     print(embed_lst.shape, real_lst.shape)


#     with open(f"{embed_save_dir}/embeds_train.npy", 'wb') as f:
#         np.save(f, embed_lst)
#     with open(f"{embed_save_dir}/reals_train.npy", 'wb') as f:
#         np.save(f, real_lst)


def infer_one_design(net_enc, design):
    print('Current Design: ', design)
    train_net_loader = load_test_dataset_finetune_gnnre(batch_size=8, design=design, task=task)
    train_net_loader = train_net_loader
    embed_lst, real_lst = [], []
    train_net_loader, net_enc = accelerator.prepare(train_net_loader, net_enc)
    with torch.no_grad():
        for idx, data in enumerate(train_net_loader):
            node_embeds, _ = net_enc(data, mode='infer')
            # graph_pred = net_enc.downstream_mlp(graph_embeds)
            embed_lst.extend(node_embeds.detach().cpu().numpy().tolist())
            real_lst.extend(data.y.detach().cpu().numpy().tolist())

    with open(f"../../dataset/design/{design}_vec.npy", 'rb') as f:
        vec_arr = np.load(f)
    
    embed_lst = np.array(embed_lst)
    print(embed_lst.shape)
    print(vec_arr.shape)
    ## vec_arr (17, ), embed_lst (n, 768)
    embed_lst = np.concatenate((embed_lst, np.tile(vec_arr, (embed_lst.shape[0], 1))), axis=1)
    real_lst = np.array(real_lst)

    
    print(embed_lst.shape, real_lst.shape)


    with open(f"{embed_save_dir}/embeds_{design}.npy", 'wb') as f:
        np.save(f, embed_lst)
    with open(f"{embed_save_dir}/reals_{design}.npy", 'wb') as f:
        np.save(f, real_lst)




def enc2vec(net_enc):
    with open(f"../../data_collect/data_js_gnnre/design_list.json", 'r') as f:
        design_lst = json.load(f)
    for design in design_lst:
        infer_one_design(net_enc, design)



if __name__ == '__main__':
    epoch = 5
    global task
    task = "task4_aig"

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
    net_enc.load_state_dict(torch.load(model_save_path, weights_only=True, map_location='cpu'))
    net_enc.eval()

    # enc2vec_training(net_enc)
    enc2vec(net_enc)