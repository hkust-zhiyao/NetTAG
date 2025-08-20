from __future__ import absolute_import
from __future__ import print_function
import sys, copy, random
import os, time, json, re, pickle
from optparse import OptionParser
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pyverilog
from pyverilog.vparser.parser import parse
from DG import *
from multiprocessing import Pool
import networkx as nx
from pysmt.shortcuts import Symbol, And, Or, Not, Implies, Iff, is_sat, Ite, Xor, Plus, Equals, Times, Real, GE, LT, LE, GT, Minus, EqualsOrIff
from pysmt.typing import BOOL
from equal_transform import apply_random_trans
import signal

class TimeoutException(Exception):
    """Custom exception to raise on a timeout"""
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Function execution timed out")


def clean_str(expr):
    expr = re.sub(r"(\\(adder|multiplier|subtractor|comparator)_(\d+)|(sub|add|mul|comp))", "", expr)
    expr = re.sub(r'(/|\\|\'|//)*', '', expr)
    expr = re.sub(r'(_)+\'', '', expr)
    return expr


def aug_k_pos(expr, n_name, k):
    expr_ori = copy.deepcopy(expr)
    try:
        expr_ori = clean_str(str(expr_ori.serialize()))
    except:
        return set(), ""
    aug_set = set()
    aug_set.add(expr_ori)
    for i in range(k):
        n = random.randint(3, 8)
        try:
            expr = apply_random_trans(copy.deepcopy(expr), n)
            expr_str = clean_str(str(expr.serialize()))
            aug_set.add(expr_str)
        except:
            continue
    expr_smpl = clean_str(str(copy.deepcopy(expr.simplify()).serialize()))
    aug_set.add(expr_smpl)
    return aug_set, expr_ori

def run_one_node(param):
    node, aug_save_dir = param
    n_name = node.name
    n_name = re.sub(r'/', '_', n_name)
    n_name = re.sub(r'\\', '', n_name)
    
    in_expr = node.in_expr
    aug_set, expr_ori = aug_k_pos(in_expr, n_name, 10)
    aug_set = aug_set - {expr_ori}
    if len(aug_set) == 0:
        return
    
    expr_ori = clean_str(f"{n_name} = {expr_ori}")
    aug_dct = {'ori': expr_ori, 'aug': []}
    for expr in aug_set:
        expr = clean_str(f"{n_name} = {expr}")
        aug_dct['aug'].append(expr)
    with open(f"{aug_save_dir}/{n_name}.json", 'w') as f:
        json.dump(aug_dct, f, indent=4)


def run_one_design(design_name, graph_dir):
    print("Current Design: ", design_name)
    node_dct_path = f"{graph_dir}/{design_name}/{design_name}_node_dict.pkl"
    if not os.path.exists(node_dct_path):
        return
    with open (node_dct_path, 'rb') as f:
        node_dict = pickle.load(f)
    
    node_lst = []
    for node in node_dict.values():
        if node.tpe in ['Input', 'Output', 'Wire']:
            continue
        if not node.in_expr:
            continue
        node_lst.append(node)
    print(f"    Total Nodes: {len(node_lst)}")

    aug_save_dir = f"../saved_expr/{design_name}"
    if not os.path.exists(aug_save_dir):
        os.makedirs(aug_save_dir)

    param_lst = [(node, aug_save_dir) for node in node_lst]
    with Pool(64) as p:
        p.map(run_one_node, param_lst)
        p.close()
        p.join()


def run_all_designs(graph_dir):
    with open(f"../../../data_collect/data_js/train_list.json", 'r') as f:
        design_list = json.load(f)
    for design in design_list:
        signal.signal(signal.SIGALRM, timeout_handler)
        # Schedule an alarm in 10 seconds
        # signal.alarm(5)
        signal.alarm(60*10)
        try:
            run_one_design(design, graph_dir)
        except TimeoutException:
            print(f"Timeout: {design}")
            continue
        finally:
            signal.alarm(0)


if __name__ == '__main__':
    graph_dir = "../saved_graph_split"
    run_all_designs(graph_dir)