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

def run_one_design(expr_save_dir):
    global dataset_dct
    for node_dct in list(os.listdir(expr_save_dir)):
        try:
            with open(f"{expr_save_dir}/{node_dct}", 'r') as f:
                dct = json.load(f)
            ori = dct['ori']
            pos = dct['aug'][0]
            dataset_dct['ori'].append(ori)
            dataset_dct['pos'].append(pos)
        except:
            continue
    

def run_all_designs():
    global dataset_dct
    dataset_dct = {"ori": [], "pos": []}
    with open(f"../../../data_collect/data_js/train_list.json", 'r') as f:
        design_list = json.load(f)
    for design in design_list:
        expr_save_dir = f"../saved_expr/{design}"
        if not os.path.exists(expr_save_dir):
            continue
        print(design)
        run_one_design(expr_save_dir)

    for design in design_list:
        expr_save_dir = f"../../net2graph_pos/saved_expr/{design}"
        if not os.path.exists(expr_save_dir):
            continue
        print(design)
        run_one_design(expr_save_dir)

    print(f"Number of samples: {len(dataset_dct['ori'])}")

    with open (f"../../../dataset/expr/expr.pkl", 'wb') as f:
        pickle.dump(dataset_dct, f)


if __name__ == '__main__':
    run_all_designs()