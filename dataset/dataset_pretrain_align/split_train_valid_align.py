import torch
import numpy as np
import pickle, json, time, re, sys
import networkx as nx
from multiprocessing import Pool
from torch_geometric.loader import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import os


def run_one_ep(design, ep):
    ### RTL dataset
    if "node_dict"  in ep:
        return None
    ep_rtl = re.sub(r'_reg_(\d+)__(\d+)_$', '', ep)
    ep_rtl = re.sub(r'_reg_(\d+)_$', '', ep_rtl)
    ep_rtl = re.sub(r'_reg$', '', ep_rtl)
    ep_rtl = re.sub(r'^\\', '', ep_rtl)
    ep_rtl = re.sub(r'^\\', '', ep_rtl)
    rtl_data_dir = f"/home/coguest5/hdl_fusion/dataset/dataset_context/llm_enc/rtl_emb/ori/{design}/{ep_rtl}.pkl"
    if not os.path.exists(rtl_data_dir):
        return None

    with open(rtl_data_dir, 'rb') as f:
        emb_tuple = pickle.load(f)
    rtl_emb = emb_tuple[2]
    
    ### Netlist dataset
    ## ori
    netlist_data_dir = f"../../dataset/tag/ori/{design}/{ep}_tag.pkl"
    if not os.path.exists(netlist_data_dir):
        return None

    with open(netlist_data_dir, 'rb') as f:
        graph_data_ori = pickle.load(f)
    # print(graph_data_ori.tpe_cnt)

    ## pos
    netlist_data_dir = f"../../dataset/tag/pos/{design}/{ep}_tag.pkl"
    if not os.path.exists(netlist_data_dir):
        return None
    
    with open(netlist_data_dir, 'rb') as f:
        graph_data_pos = pickle.load(f)

    ### Layout dataset
    layout_data_dir = f"../../dataset/graph/layout/{design}/{ep}_graph.pkl"
    if not os.path.exists(layout_data_dir):
        return None
    
    with open(layout_data_dir, 'rb') as f:
        graph_data_layout = pickle.load(f)
    
    global idx
    idx += 1


    return (rtl_emb, graph_data_ori, graph_data_pos, graph_data_layout)


def save_dataset(design_lst, tag):
    print(f'Tag: {tag}')
    global idx
    idx = 0
    rtl_dataset, netlist_ori_dataset, netlist_pos_dataset, layout_dataset = [], [], [], []
    for design in design_lst:
        print(f"Current Design: {design}")
        with open (f"../../data_collect/data_subgraph_js/{design}_list.json", 'r') as f:
            reg_lst = json.load(f)
        for ep in reg_lst:
            aligned_data = run_one_ep(design, ep)
            if not aligned_data:
                continue
            rtl_emb, graph_data_ori, graph_data_pos, graph_data_layout = aligned_data
            rtl_dataset.append(rtl_emb)
            netlist_ori_dataset.append(graph_data_ori)
            netlist_pos_dataset.append(graph_data_pos)
            layout_dataset.append(graph_data_layout)

    print(f"# of data: {len(rtl_dataset)}")
    print(f"# of data: {idx}")

    # ## split the dataset into k parts
    # k = 10
    # ll = idx
    # print(f"Length of dataset: {ll}")
    # for idx in range(k):
    #     print(idx*ll//k, (idx+1)*ll//k)
    #     with open (f"./data/{tag}_rtl_{idx}.pkl", 'wb') as f:
    #         pickle.dump(rtl_dataset[idx*ll//k:(idx+1)*ll//k], f)
    #     with open (f"./data/{tag}_netlist_ori_{idx}.pkl", 'wb') as f:
    #         pickle.dump(netlist_ori_dataset[idx*ll//k:(idx+1)*ll//k], f)
    #     with open (f"./data/{tag}_netlist_pos_{idx}.pkl", 'wb') as f:
    #         pickle.dump(netlist_pos_dataset[idx*ll//k:(idx+1)*ll//k], f)
    #     with open (f"./data/{tag}_layout_{idx}.pkl", 'wb') as f:
    #         pickle.dump(layout_dataset[idx*ll//k:(idx+1)*ll//k], f)
 
    with open (f"./data/{tag}_rtl.pkl", 'wb') as f:
        pickle.dump(rtl_dataset, f)
    with open (f"./data/{tag}_netlist_ori.pkl", 'wb') as f:
        pickle.dump(netlist_ori_dataset, f)
    with open (f"./data/{tag}_netlist_pos.pkl", 'wb') as f:
        pickle.dump(netlist_pos_dataset, f)
    with open (f"./data/{tag}_layout.pkl", 'wb') as f:
        pickle.dump(layout_dataset, f)


if __name__ == '__main__':
    tag = 'train'
    # tag = 'val'
    tag = 'demo'
    cmd = "align"
    tag_cmd = f"{tag}_{cmd}"
    with open (f"../../data_collect/data_js/{tag}_list.json", 'r') as f:
        design_lst = json.load(f)
    save_dataset(design_lst, tag_cmd)

