import torch
import numpy as np
import pickle, json, time, re, sys
import networkx as nx
from multiprocessing import Pool
from torch_geometric.loader import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import os, random


def run_one_ep(design, ep):
    ### RTL dataset
    if "node_dict"  in ep:
        return None

    ### Netlist dataset
    ## ori
    netlist_data_dir = f"../../dataset/tag/ori/{design}/{ep}_tag.pkl"
    if not os.path.exists(netlist_data_dir):
        print(f"Netlist ori is not found for {netlist_data_dir}")
        return None

    with open(netlist_data_dir, 'rb') as f:
        graph_data_ori = pickle.load(f)

    ## pos
    netlist_data_dir = f"../../dataset/tag/pos/{design}/{ep}_tag.pkl"
    if not os.path.exists(netlist_data_dir):
        print(f"Netlist pos not found for {netlist_data_dir}")
        return None
    
    with open(netlist_data_dir, 'rb') as f:
        graph_data_pos = pickle.load(f)

    

    global idx
    idx += 1


    return (graph_data_ori, graph_data_pos)


def save_dataset(design_lst, tag):
    print(f'Tag: {tag}')
    global idx
    idx = 0
    netlist_ori_dataset, netlist_pos_dataset = [], []
    for design in design_lst:
        print(f"Current Design: {design}")
        with open (f"../../data_collect/data_subgraph_js/{design}_list.json", 'r') as f:
            reg_lst = json.load(f)
        for ep in reg_lst:
            aligned_data = run_one_ep(design, ep)
            if not aligned_data:
                continue
            graph_data_ori, graph_data_pos = aligned_data
            netlist_ori_dataset.append(graph_data_ori)
            netlist_pos_dataset.append(graph_data_pos)

    print(f"# of data: {len(graph_data_ori)}")
    print(f"# of data: {idx}")

    ## random shuffle these two lists with same sequence
    random.seed(42)
    random.shuffle(netlist_ori_dataset)
    random.seed(42)
    random.shuffle(netlist_pos_dataset)

    ## split into k parts and save
    k = 10
    ll = len(netlist_ori_dataset)
    for idx in range(k):
        start_idx = idx*ll//k
        end_idx = (idx+1)*ll//k
        print(f"Start: {start_idx}, End: {end_idx}")
        with open (f"./data/{tag}_netlist_ori_{idx}.pkl", 'wb') as f:
            pickle.dump(netlist_ori_dataset[start_idx:end_idx], f)
        with open (f"./data/{tag}_netlist_pos_{idx}.pkl", 'wb') as f:
            pickle.dump(netlist_pos_dataset[start_idx:end_idx], f)

    # with open (f"./data/{tag}_netlist_ori.pkl", 'wb') as f:
    #     pickle.dump(netlist_ori_dataset, f)
    # with open (f"./data/{tag}_netlist_pos.pkl", 'wb') as f:
    #     pickle.dump(netlist_pos_dataset, f)

if __name__ == '__main__':
    tag = 'train'
    cmd = "net"
    tag_cmd = f"{tag}_{cmd}"
    with open (f"../../data_collect/data_js/{tag}_list.json", 'r') as f:
        design_lst = json.load(f)
    save_dataset(design_lst, tag_cmd)

