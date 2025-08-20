### check the number of files in the directory
import os
import json, pickle, re
import networkx as nx
import torch
from torch_geometric.data import Data
from pysmt.shortcuts import Symbol, And, Or, Not, Implies, Iff, is_sat, Ite, Xor, Plus, Equals, Times, Real, GE, LT, LE, GT, Minus, EqualsOrIff
from pysmt.typing import BOOL
from multiprocessing import Pool

def expr2text_attr(n_name, n_tpe, expr, node_out):
    if expr:
        expr = str(expr.serialize())
    expr = f"{n_name} = {expr}"
    expr = re.sub(r"(\\(adder|multiplier|subtractor|comparator)_(\d+)|(sub|add|mul|comp))", "", expr)
    expr = re.sub(r'(/|\\|\'|//)*', '', expr)
    expr = re.sub(r'(_)+\'', '', expr)

    # node_out_text = ""
    # if node_out:
    #     for out_n, out_tpe in node_out.items():
    #         node_out_text += f"{out_n} ({out_tpe})," 
    
    text_attr = f"{n_name} is a {n_tpe} node with the following symbolic expression: {expr}."
    #  Its fan-out nodes are: {node_out_text}.
    
    return text_attr

def get_physical_attr(node):
    pwr = node.pwr if node.pwr else 0
    area = node.area if node.area else 0
    delay = node.delay if node.delay else 0
    load = node.load if node.load else 0
    tr = node.tr if node.tr else 0
    prob = node.prob if node.prob else 0
    cap = node.cap if node.cap else 0
    res = node.res if node.res else 0

    feat_vec = [pwr, area, delay, load, tr, prob, cap, res]
    return feat_vec

def run_one_subgraph(param):
    g_nx, node_dict = param
    expr_lst = []

    node_name_to_index = {name: idx for idx, name in enumerate(g_nx.nodes())}

    # Add [CLS] node
    cls_node_index = len(node_name_to_index)
    node_name_to_index['[CLS]'] = cls_node_index
    g_nx.add_node('[CLS]')

    # Create edges from [CLS] node to all other nodes
    for node in g_nx.nodes():
        if node != '[CLS]':
            g_nx.add_edge('[CLS]', node)
            # g_nx.add_edge(node, '[CLS]')

    # Create one-hot encoded feature vectors
    x = []
    y = []
    for n in g_nx.nodes():
        if n == '[CLS]':
            y.append(1.0)
            continue
        node = node_dict[n]
        node_text_attr = node.text_attr
        node_feat_vec = node.feat_vec
        print(node_feat_vec)
        print(node_text_attr)
        x.append(torch.tensor(node_feat_vec, dtype=torch.float))
        expr_lst.append(node_text_attr)
        y.append(1.0)

    # Convert to PyTorch Geometric Data object
    edge_index = torch.tensor([[node_name_to_index[src], node_name_to_index[dst]] for src, dst in g_nx.edges()], dtype=torch.long).t().contiguous()
    y = torch.tensor(y, dtype=torch.float)

    graph_data = Data(x=x, edge_index=edge_index, y=y)
    return (graph_data, expr_lst)


def update_one_node(param):
    n, node = param
    node_text_attr = expr2text_attr(n, node.tpe, node.in_expr, node.out_expr)
    node_feat_vec = get_physical_attr(node)
    node.text_attr = node_text_attr
    node.feat_vec = node_feat_vec
    return (n,node)


def load_netgraph(design_lst):
    graph_lst, text_lst = [],[]

    graph_dir = "./saved_graph_split"

    ll = 0

    for design in design_lst:
        if not os.path.exists(f"{graph_dir}/{design}/{design}_node_dict.pkl"):
            continue
        subgraph_lst = []
        for subgraph in os.listdir(f"{graph_dir}/{design}"):
            subgraph = subgraph.split('.')[0]
            # print(f"Loading {subgraph}...")
            subgraph_lst.append(subgraph)
        
        # with open(f"../../data_collect/data_subgraph_js/{design}_list.json", 'w') as f:
        #     json.dump(subgraph_lst, f, indent=4)
        print(f"Finished {design}: {len(subgraph_lst)} subgraphs.")
        ll += len(subgraph_lst)
    print(f"Total # of subgraphs: {ll}")

if __name__ == '__main__':
    with open(f"../../data_collect/data_js/train_list.json", 'r') as f:
        train_lst = json.load(f)
    # train_lst = ['b14']
    load_netgraph(train_lst)