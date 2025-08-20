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
import signal

class TimeoutException(Exception):
    """Custom exception to raise on a timeout"""
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Function execution timed out")

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

def run_one_node(param):
    n, node = param
    node_text_attr = expr2text_attr(n, node.tpe, node.in_expr, node.out_expr)
    node_feat_vec = get_physical_attr(node)
    node.text_attr = node_text_attr
    node.feat_vec = node_feat_vec
    return node

def run_one_subgraph(param):
    g_nx, node_dict, design_name, subgraph_name = param
    
    try:
        expr_lst = []
        g_nx = nx.DiGraph(g_nx)
        ### Add [CLS] node
        g_nx.add_node('[CLS]')
        for node in g_nx.nodes():
            if node != '[CLS]':
                g_nx.add_edge('[CLS]', node)

        x = []
        for n in g_nx.copy().nodes():
            if n == '[CLS]':
                node_feat_vec = [0,0,0,0,0,0,0,0]
                node_text_attr = 'This is a [CLS] node.'
            else:
                node = node_dict[n]
                node_text_attr = node.text_attr
                node_feat_vec = node.feat_vec
            # print(node_feat_vec)
            # print(node_text_attr)
            x.append(torch.tensor(node_feat_vec, dtype=torch.float))
            expr_lst.append(node_text_attr)


        #### Convert to PyTorch Geometric Data object
        for u, v in g_nx.edges():
            if g_nx[u][v]:
                g_nx[u][v].clear()    
        graph_data = from_networkx(g_nx)
        graph_data.x = torch.stack(x, dim=0)

        save_dataset_path = f"../../dataset/graph/{cmd}/{design_name}"
        if not os.path.exists(save_dataset_path):
            os.makedirs(save_dataset_path)
        with open(f"{save_dataset_path}/{subgraph_name}_graph.pkl", 'wb') as f:
            pickle.dump(graph_data, f)
        with open(f"{save_dataset_path}/{subgraph_name}_text.pkl", 'wb') as f:
            pickle.dump(expr_lst, f)
    except:
        return


def update_one_node(param):
    n, node = param
    node_text_attr = expr2text_attr(n, node.tpe, node.in_expr, node.out_expr)
    node_feat_vec = get_physical_attr(node)
    node.text_attr = node_text_attr
    node.feat_vec = node_feat_vec
    return (n,node)


def run_one_design(graph_dir, design):
    if not os.path.exists(f"{graph_dir}/{design}/{design}_node_dict.pkl"):
        return
    with open(f"../../data_collect/data_subgraph_js/{design}_list.json", 'r') as f:
        subgraph_lst = json.load(f)
    print(f"Current Design: {design}, # of subgraphs: {len(subgraph_lst)}")        
    with open(f"{graph_dir}/{design}/{design}_node_dict.pkl", 'rb') as f:
        node_dict = pickle.load(f)

    print(f"Total Nodes: {len(node_dict)}")
    param_lst = []
    for n, node in node_dict.copy().items():
        if node.tpe == 'Wire':
            continue
        param_lst.append((n, node))
    print('Updating node_dict (parallel)')
    with Pool(100) as p:
        node_data = p.map(update_one_node, param_lst)
        p.close()
        p.join()
    node_dict_update = {}
    for d in node_data:
        node_dict_update[d[0]] = d[1]
    ## parallel
    param_lst = []
    for subgraph in subgraph_lst:
        saved_path = f"{graph_dir}/{design}/{subgraph}.pkl"
        if not os.path.exists(saved_path):
            subgraph = re.sub(r'\\', '', subgraph)
            subgraph = re.sub(r'\\', '_', subgraph)
            saved_path = f"{graph_dir}/{design}/{subgraph}.pkl"
        with open(saved_path, 'rb') as f:
            g_nx = pickle.load(f)
        param_lst.append((g_nx, node_dict_update, design, subgraph))

    # for param in param_lst:
    #     run_one_subgraph(param)

    print('Running parallel')
    with Pool(32) as p:
        p.map(run_one_subgraph, param_lst)
        p.close()
        p.join()


def load_netgraph(design_lst):
    graph_lst, text_lst = [],[]

    graph_dir = "./saved_graph_split"

    for design in design_lst:
        if os.path.exists(f"../../dataset/graph/{cmd}/{design}"):
            continue
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60*20)
        try:
            run_one_design(graph_dir, design)
        except TimeoutException:
            print(f"Timeout: {design}")
            continue
        finally:
            signal.alarm(0)
        

def check_dir(dir_path):
    return len(os.listdir(dir_path))

if __name__ == '__main__':
    global cmd
    cmd = "layout"
    with open(f"../../data_collect/data_js/train_list.json", 'r') as f:
        train_lst = json.load(f)
    load_netgraph(train_lst)

    # with open(f"../../data_collect/data_js/test_list.json", 'r') as f:
    #     test_lst = json.load(f)
    # load_netgraph(test_lst)