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


def run_one_subgraph(param):
    # try:
        g_nx, node_idx_dict, vec_arr, design_name, subgraph_name = param
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

        ## x.shape: (n, 4121)
        ## tpe_x (n, 17)
        tpe_x = graph_data.x[:, 8:25]
        tpe_cnt = tpe_x.sum(dim=0)
        graph_data.tpe_cnt = tpe_cnt

        save_dataset_path = f"../../dataset/tag/{cmd}/{design_name}"
        if not os.path.exists(save_dataset_path):
            os.makedirs(save_dataset_path)
        with open(f"{save_dataset_path}/{subgraph_name}_tag.pkl", 'wb') as f:
            pickle.dump(graph_data, f)

    # except:
    #     print(f"Error in {design_name} {subgraph_name}")
    #     return


def load_netgraph(design_lst):

    graph_dir = "./saved_graph_split"
    node_dict_dir = "./saved_node_dict_tag"

    for design in design_lst:
        print('Current design:', design)
        if not os.path.exists(f"{node_dict_dir}/{design}_node.pkl"):
            print(f"Node dict not found for {design}")
            continue
        with open(f"{node_dict_dir}/{design}_node.pkl", 'rb') as f:
            node_lst = pickle.load(f)
        with open(f"{node_dict_dir}/{design}_vec.npy", 'rb') as f:
            vec_arr = np.load(f)

        node_idx_dict = {}
        for i, node in enumerate(node_lst):
            node_idx_dict[node] = i

        with open(f"../../data_collect/data_subgraph_js/{design}_list.json", 'r') as f:
            subgraph_lst = json.load(f)
        param_lst = []
        for subgraph in subgraph_lst:
            saved_path = f"{graph_dir}/{design}/{subgraph}.pkl"
            if not os.path.exists(saved_path):
                subgraph = re.sub(r'\\', '', subgraph)
                subgraph = re.sub(r'\\', '_', subgraph)
                saved_path = f"{graph_dir}/{design}/{subgraph}.pkl"
            with open(saved_path, 'rb') as f:
                g_nx = pickle.load(f)
            param_lst.append((g_nx, node_idx_dict, vec_arr, design, subgraph))

        for param in param_lst:
            run_one_subgraph(param)

        # print('Running parallel')
        # with Pool(80) as p:
        #     p.map(run_one_subgraph, param_lst)
        #     p.close()
        #     p.join()

def check_dir(dir_path):
    return len(os.listdir(dir_path))

if __name__ == '__main__':
    global cmd
    cmd = "ori"
    with open(f"../../data_collect/data_js/design_list.json", 'r') as f:
        train_lst = json.load(f)
    train_lst = ['vga_lcd']
    load_netgraph(train_lst)

    # with open(f"../../data_collect/data_js/test_list.json", 'r') as f:
    #     test_lst = json.load(f)
    # load_netgraph(test_lst)