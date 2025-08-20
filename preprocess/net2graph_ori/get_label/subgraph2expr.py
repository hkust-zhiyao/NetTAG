from __future__ import absolute_import
from __future__ import print_function
import sys
import os, time, json, re, pickle
from optparse import OptionParser
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pyverilog
from pyverilog.vparser.parser import parse
from DG import *
from multiprocessing import Pool
import networkx as nx
import torch
from transformers import pipeline

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def merge_lines(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    merged_lines = []
    for line in lines:
        if re.findall(r"        (\S+)", line):
            if merged_lines:
                merged_lines[-1] = merged_lines[-1].rstrip() + line.lstrip()
        else:
            merged_lines.append(line)
    with open(file_path, 'w') as file:
        file.writelines(merged_lines)
    
    return merged_lines

def run_one_design(design_name, save_dir):
    with open (f"{save_dir}/{design_name}_graph.pkl", 'rb') as f:
        g = pickle.load(f)
    with open (f"{save_dir}/{design_name}_node_dict.pkl", 'rb') as f:
        node_dict = pickle.load(f)
    
    g_nx = nx.DiGraph(g)
    for n in g_nx.nodes():
        node = node_dict[n]
        n_name = node.name

        # label = re.findall(r"\\(adder|multiplier|subtractor|comparator)_(\d+)", n_name)
        label_re = re.findall(r"\\((\w+)_(\d+))", n_name)
        if label_re:
            label = label_re[0][1]
        else:
            continue
        n_text = re.sub(r"(\\(adder|multiplier|subtractor|comparator)_(\d+)|(sub|add|mul|comp))", "", node.node_text)
        in_text = ""
        for t in node.input_text:
            in_text += re.sub(r"(\\(adder|multiplier|subtractor|comparator)_(\d+)|(sub|add|mul|comp))", "", t)
        out_text = ""
        for t in node.output_text:
            out_text += re.sub(r"(\\(adder|multiplier|subtractor|comparator)_(\d+)|(sub|add|mul|comp))", "", t)

        print(n_name, label, n_text, in_text, out_text)

        




    

def run_all_designs():

    design_dir = "/home/coguest5/net_tag/dataset/labeled_netlist"
    save_dir = "../saved_graph"
    design_list = os.listdir(design_dir)
    for design in design_list:
        print("Current Design: ", design)
        design_name = design.split(".")[0]
        if not os.path.exists(f"{save_dir}/{design_name}_graph.pkl"):
            continue
        run_one_design(design_name, save_dir)
        print("Finish: ", design_name)


if __name__ == '__main__':

    run_all_designs()
    exit()

    design_name = "add_mul_2_bit"
    design_dir = "/home/coguest5/net_tag/dataset/netlist/Test_add_mul_2_bit_Syn_65nm.v"
    save_dir = "../saved_graph"
    run_one_design(design_name, design_dir, save_dir)