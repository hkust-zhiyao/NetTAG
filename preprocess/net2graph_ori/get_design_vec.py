### check the number of files in the directory
import os
import json, pickle, re
import networkx as nx
import torch
from AST_analyzer import AST_analyzer
import numpy as np
from multiprocessing import Pool

def run_one_design(design):
    print("Current Design: ", design)
    graph_dir = "./saved_analyzer"
    with open(f"{graph_dir}/{design}_analyzer.pkl", 'rb') as f:
        ast_analysis = pickle.load(f)
    
    graph = ast_analysis.graph.graph
    graph = nx.DiGraph(graph)
    node_dict = ast_analysis.graph.node_dict

    tpe_lst = ['Input', 'Output', \
               'DFF', 'INV', 'BUF', 'XOR', 'AOI', \
               'OAI', 'OR', 'NAND', 'AND', 'MUX',\
               'XNOR', 'HA', 'FA', 'DLL', 'DLH']
    vec_lst = np.zeros(len(tpe_lst))

    for n in graph.nodes():
        if node_dict[n].tpe in tpe_lst:
            vec_lst[tpe_lst.index(node_dict[n].tpe)] += 1

    save_dir = "../../dataset/design/"
    with open(f"{save_dir}/{design}_vec.npy", 'wb') as f:
        np.save(f, vec_lst)


def run_all(design_lst):

    with Pool(40) as p:
        p.map(run_one_design, design_lst)
        p.close()
        p.join()

    # for design in design_lst:
    #     run_one_design(design)

if __name__ == '__main__':
    with open(f"../../data_collect/data_js/design_list.json", 'r') as f:
        design_list = json.load(f) 
    run_all(design_list)

    # with open(f"../../data_collect/data_js/test_list.json", 'r') as f:
    #     test_lst = json.load(f)
    # load_netgraph(test_lst)