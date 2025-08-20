### check the number of files in the directory
import os
import json, pickle, re
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx
from pysmt.shortcuts import Symbol, And, Or, Not, Implies, Iff, is_sat, Ite, Xor, Plus, Equals, Times, Real, GE, LT, LE, GT, Minus, EqualsOrIff
from pysmt.typing import BOOL
from multiprocessing import Pool
import numpy as np


def run_one_graph(g_nx, node_idx_dict, vec_arr, design_name):

    g_nx = nx.DiGraph(g_nx)
    ### Add [CLS] node
    g_nx.add_node('[CLS]')
    for node in g_nx.nodes():
        if node != '[CLS]':
            g_nx.add_edge('[CLS]', node)

    x = []
    for n in g_nx.copy().nodes():
        if n == '[CLS]':
            node_feat_vec = np.zeros(4121)
        else:
            node_idx = node_idx_dict[n]
            node_feat_vec = vec_arr[node_idx]
        x.append(torch.tensor(node_feat_vec, dtype=torch.float))


    #### Convert to PyTorch Geometric Data object
    for u, v in g_nx.edges():
        if g_nx[u][v]:
            g_nx[u][v].clear()    
    graph_data = from_networkx(g_nx)
    graph_data.x = torch.stack(x, dim=0)


    save_dataset_path = f"../../dataset/tag/{cmd}/{design_name}"
    if not os.path.exists(save_dataset_path):
        os.makedirs(save_dataset_path)
    with open(f"{save_dataset_path}/{design_name}_tag.pkl", 'wb') as f:
        pickle.dump(graph_data, f)


def load_netgraph(design_lst):

    graph_dir = "./saved_graph_split"
    node_dict_dir = "./saved_node_dict_tag"

    for design in design_lst:
        print('Current design:', design)
        with open(f"{node_dict_dir}/{design}_node.pkl", 'rb') as f:
            node_lst = pickle.load(f)
        with open(f"{node_dict_dir}/{design}_vec.npy", 'rb') as f:
            vec_arr = np.load(f)

        node_idx_dict = {}
        for i, node in enumerate(node_lst):
            node_idx_dict[node] = i


        saved_path = f"{graph_dir}/{design}/{design}_graph.pkl"
        with open(saved_path, 'rb') as f:
            g_nx = pickle.load(f)
        run_one_graph(g_nx, node_idx_dict, vec_arr, design)



def check_dir(dir_path):
    return len(os.listdir(dir_path))

if __name__ == '__main__':
    global cmd
    cmd = "gnnre"
    with open(f"../../data_collect/data_js_gnnre/design_list.json", 'r') as f:
        train_lst = json.load(f)
    load_netgraph(train_lst)
