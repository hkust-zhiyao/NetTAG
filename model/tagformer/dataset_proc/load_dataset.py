import pickle, json
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
# from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
import numpy as np

class NetDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collate_fn = self.custom_collate_fn

    def custom_collate_fn(self, batch):
        tpe_cnt_vec = np.array([item.tpe_cnt for item in batch])
        tpe_cnt_vec = torch.tensor(tpe_cnt_vec, dtype=torch.float32)
        # summed_tpe_cnt = sum(item.tpe_cnt for item in batch)
    
        ## return the batched graph in the original pyg dataloader format
        batched_graph = Batch.from_data_list(batch)
        # Add the summed 'tpe_cnt' to the batched graph
        batched_graph.tpe_cnt = tpe_cnt_vec
        return batched_graph


def load_train_valid_dataset_stage_align(batch_size, idx, train_valid="train"):
    shuffle_tf = False

    dataset_dir = f"../../dataset/dataset_pretrain_align/data"

    
    if idx==None:
        rtl_dir = f'{dataset_dir}/{train_valid}_align_rtl.pkl'
        net_ori_dir = f'{dataset_dir}/{train_valid}_align_netlist_ori.pkl'
        net_pos_dir = f'{dataset_dir}/{train_valid}_align_netlist_pos.pkl'
        layout_dir = f'{dataset_dir}/{train_valid}_align_layout.pkl'
    else:
        print(f"----- Align Idx: {idx} -----")
        rtl_dir = f'{dataset_dir}/{train_valid}_align_rtl_{idx}.pkl'
        net_ori_dir = f'{dataset_dir}/{train_valid}_align_netlist_ori_{idx}.pkl'
        net_pos_dir = f'{dataset_dir}/{train_valid}_align_netlist_pos_{idx}.pkl'
        layout_dir = f'{dataset_dir}/{train_valid}_align_layout_{idx}.pkl'

    print(f"Loading RTL dataset ...")
    with open(rtl_dir, 'rb') as f:
        rtl_vec_lst = pickle.load(f)
    rtl_loader = torch.utils.data.DataLoader(
        rtl_vec_lst,
        batch_size=batch_size,
        shuffle=shuffle_tf,
    )
    del rtl_vec_lst

    print(f"Loading Netlist dataset ...")
    with open(net_ori_dir, 'rb') as f:
        net_ori_lst = pickle.load(f)
    net_ori_loader = NetDataLoader(
        net_ori_lst,
        batch_size=batch_size,
        shuffle=shuffle_tf,
    )
    del net_ori_lst

    with open(net_pos_dir, 'rb') as f:
        net_pos_lst = pickle.load(f)
    net_pos_loader = DataLoader(
        net_pos_lst,
        batch_size=batch_size,
        shuffle=shuffle_tf,
    )
    del net_pos_lst

    print(f"Loading Layout dataset ...")
    with open(layout_dir, 'rb') as f:
        layout_lst = pickle.load(f)
    layout_loader = DataLoader(
        layout_lst,
        batch_size=batch_size,
        shuffle=shuffle_tf,
    )
    del layout_lst

    loader_align = (rtl_loader, net_ori_loader, net_pos_loader, layout_loader)

    return loader_align


def load_train_valid_dataset_pretrain_netlist(batch_size, idx, train_valid="train"):
    shuffle_tf = False

    dataset_dir = f"../../dataset/dataset_pretrain_align/data"

    print(f"Loading RTL dataset ...")
    if idx==None:
        net_ori_dir = f'{dataset_dir}/{train_valid}_net_netlist_ori.pkl'
        net_pos_dir = f'{dataset_dir}/{train_valid}_net_netlist_pos.pkl'
    else:
        print(f"Net Idx: {idx}")
        net_ori_dir = f'{dataset_dir}/{train_valid}_net_netlist_ori_{idx}.pkl'
        net_pos_dir = f'{dataset_dir}/{train_valid}_net_netlist_pos_{idx}.pkl'


    print(f"Loading Netlist dataset ...")
    with open(net_ori_dir, 'rb') as f:
        net_ori_lst = pickle.load(f)
    net_ori_loader = NetDataLoader(
        net_ori_lst,
        batch_size=batch_size,
        shuffle=shuffle_tf,
    )
    del net_ori_lst

    with open(net_pos_dir, 'rb') as f:
        net_pos_lst = pickle.load(f)
    net_pos_loader = DataLoader(
        net_pos_lst,
        batch_size=batch_size,
        shuffle=shuffle_tf,
    )
    del net_pos_lst


    loader_align = (net_ori_loader, net_pos_loader)

    return loader_align



def load_train_valid_dataset_finetune(batch_size, train_valid="train", task="task1"):
    shuffle_tf = False
    dataset_dir = f"../../dataset/dataset_finetune/data"
    print(f"Loading Netlist dataset ...")
    with open(f'{dataset_dir}/{train_valid}_{task}.pkl', 'rb') as f:
        net_ori_lst = pickle.load(f)
    net_ori_loader = DataLoader(
        net_ori_lst,
        batch_size=batch_size,
        shuffle=shuffle_tf,
        
    )
    del net_ori_lst

    return net_ori_loader


def load_test_dataset_finetune_one_design(batch_size, design="", task="task1"):
    shuffle_tf = False
    dataset_dir = f"../../dataset/dataset_finetune/data"
    print(f"Loading Netlist dataset ...")
    with open(f'{dataset_dir}/{design}_{task}.pkl', 'rb') as f:
        net_ori_lst = pickle.load(f)
    net_ori_loader = DataLoader(
        net_ori_lst,
        batch_size=batch_size,
        shuffle=shuffle_tf,
    )
    del net_ori_lst

    return net_ori_loader

def load_test_dataset_finetune_gnnre(batch_size, design="", task="task4"):
    shuffle_tf = False
    dataset_dir = f"../../dataset/dataset_finetune/data"
    print(f"Loading Netlist dataset ...")
    with open(f'{dataset_dir}/{design}_{task}.pkl', 'rb') as f:
        net_ori = pickle.load(f)
    net_ori_lst = [net_ori]
    net_ori_loader = DataLoader(
        net_ori_lst,
        batch_size=batch_size,
        shuffle=shuffle_tf,
    )
    del net_ori_lst

    return net_ori_loader