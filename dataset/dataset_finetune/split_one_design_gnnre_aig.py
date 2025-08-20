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
    ### Netlist dataset
    ## ori
    if "node_dict"  in ep:
        return None
    netlist_data_dir = f"../../dataset/tag/ori/{design}/{ep}_tag.pkl"
    if not os.path.exists(netlist_data_dir):
        print(f"Netlist ori is not found for {netlist_data_dir}")
        return None

    with open(netlist_data_dir, 'rb') as f:
        graph_data_ori = pickle.load(f)

    return graph_data_ori


def run_one_design_task4(design):
    ### Task 4: Node functional type prediction ###

    netlist_data_dir = f"../../dataset/tag/gnnre/{design}/{design}_tag.pkl"
    with open(netlist_data_dir, 'rb') as f:
        graph_data_ori = pickle.load(f)
    
    print(f"Design: {design}")
    
    label_dir = f"../graph/gnnre/{design}/{design}_label.pkl"
    with open(label_dir, 'rb') as f:
        node_label_lst = pickle.load(f)
    assert len(node_label_lst) == graph_data_ori.x.shape[0]
    graph_data_ori.y = torch.tensor(node_label_lst, dtype=torch.long)
    print(graph_data_ori.x.shape, graph_data_ori.y.shape)
    
    return graph_data_ori




def save_dataset(design_lst):
    global dataset_lst_task4

    global idx
    idx = 0
    for design in design_lst:
        dataset_lst_task4 = run_one_design_task4(design)
        print(f"# of data (Task 4): {len(dataset_lst_task4)}")
        with open(f"./data/{design}_task4_aig.pkl", 'wb') as f:
            pickle.dump(dataset_lst_task4, f)


if __name__ == '__main__':
    with open (f"../../data_collect/data_js_gnnre/design_list.json", 'r') as f:
        design_lst = json.load(f)
    save_dataset(design_lst)

