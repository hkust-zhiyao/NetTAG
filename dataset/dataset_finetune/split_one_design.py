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


def run_one_design_task1(design):
    ### Task 1: Register slack prediction ###
    label_dir = f"../../data_collect/data_pt/init/{design}/slack.json"
    with open (label_dir, 'r') as f:
        slack_dict_init_tmp = json.load(f)
    slack_dict_init = {}
    for k, v in slack_dict_init_tmp.items():
        k = re.sub(r"/", "_", k)
        slack_dict_init[k] = v

    label_dir = f"../../data_collect/data_pt/place/{design}/slack.json"
    with open (label_dir, 'r') as f:
        slack_dict_place_tmp = json.load(f)
    slack_dict_place = {}
    for k, v in slack_dict_place_tmp.items():
        k = re.sub(r"/", "_", k)
        slack_dict_place[k] = v
    
    label_dir = f"../../data_collect/data_pt/place_opt/{design}/slack.json"
    with open (label_dir, 'r') as f:
        slack_dict_place_opt_tmp = json.load(f)
    slack_dict_place_opt = {}
    for k, v in slack_dict_place_opt_tmp.items():
        k = re.sub(r"/", "_", k)
        slack_dict_place_opt[k] = v
    
    with open (f"../../data_collect/data_subgraph_js/{design}_list.json", 'r') as f:
        reg_lst_tmp = json.load(f)
    reg_dict = {}
    for r in reg_lst_tmp:
        r_new = re.sub(r"\\", "", r)
        r_new = re.sub(r"\\", "", r_new)
        reg_dict[r_new] = r

    for ep_label, ep in reg_dict.items():
        labeled_data = run_one_ep(design, ep)
        if not labeled_data:
            continue
        if (ep_label not in slack_dict_init.keys()) or (ep_label not in slack_dict_place.keys()) or (ep_label not in slack_dict_place_opt.keys()):
            continue
        
        slack_init = torch.tensor([slack_dict_init[ep_label]], dtype=torch.float)
        labeled_data.y0 = slack_init
        slack_place = torch.tensor([slack_dict_place[ep_label]], dtype=torch.float)
        labeled_data.y1 = slack_place
        slack_place_opt = torch.tensor([slack_dict_place_opt[ep_label]], dtype=torch.float)
        labeled_data.y2 = slack_place_opt
        dataset_lst_task1.append(labeled_data)



def run_one_design_task3(design):
    ### Task 3: Register type classification ###
    label_dir = f"../../data_collect/data_pt/func_label/{design}.json"
    with open (label_dir, 'r') as f:
        reg_dict = json.load(f)

    print(f"Current Design: {design}")
    with open (f"../../data_collect/data_subgraph_js/{design}_list.json", 'r') as f:
        reg_lst = json.load(f)
    for ep in reg_lst:
        labeled_data = run_one_ep(design, ep)
        if not labeled_data:
            continue
        ep_rtl = re.sub(r'_reg_(\d+)__(\d+)_$', '', ep)
        ep_rtl = re.sub(r'_reg_(\d+)_$', '', ep_rtl)
        ep_rtl = re.sub(r'_reg$', '', ep_rtl)
        ep_rtl = re.sub(r'^\\', '', ep_rtl)
        ep_rtl = re.sub(r'^\\', '', ep_rtl)
        ep_rtl = re.sub(r'/', '_', ep_rtl)
        if ep_rtl not in reg_dict.keys():
            continue
        reg_tpe = reg_dict[ep_rtl]
        labeled_data.y = torch.tensor([reg_tpe], dtype=torch.int)
        dataset_lst_task3.append(labeled_data)
        ## delete ep_rtl in reg_dict
        del reg_dict[ep_rtl]



def save_dataset(design_lst):
    global dataset_lst_task1, dataset_lst_task2, dataset_lst_task3

    global idx
    idx = 0
    for design in design_lst:
        dataset_lst_task1 = []
        dataset_lst_task2 = []
        dataset_lst_task3 = []
        # run_one_design_task1(design)
        # print(f"{design}: # of data (Task 1): {len(dataset_lst_task1)}")
        # with open(f"./data/{design}_task1.pkl", 'wb') as f:
        #     pickle.dump(dataset_lst_task1, f)

        run_one_design_task3(design)
        print(f"# of data (Task 3): {len(dataset_lst_task3)}")
        with open(f"./data/{design}_task3.pkl", 'wb') as f:
            pickle.dump(dataset_lst_task3, f)
 

if __name__ == '__main__':
    with open (f"../../data_collect/data_js/design_list.json", 'r') as f:
        design_lst = json.load(f)
    save_dataset(design_lst)

