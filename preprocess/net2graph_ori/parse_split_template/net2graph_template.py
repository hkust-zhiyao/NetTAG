from __future__ import absolute_import
from __future__ import print_function
import sys
import os, time, json, re, pickle
from optparse import OptionParser
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pyverilog
from pyverilog.vparser.parser import parse
from AST_analyzer import *
from AST_analyzer import AST_analyzer
from multiprocessing import Pool
from pysmt.shortcuts import get_env


def merge_lines(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    merged_lines = []
    merged_lines_2 = []
    for line in lines:
        ## if there is no blank space before ')', add one
        if re.findall(r"        (\S+)", line):
            if merged_lines:
                merged_lines[-1] = merged_lines[-1].rstrip() + line.lstrip()
        else:
            merged_lines.append(line)
    for line in merged_lines:
        line = re.sub(r'(\S)(\))', r'\1 \2', line)
        merged_lines_2.append(line)
        
    with open(file_path, 'w') as file:
        file.writelines(merged_lines_2)
    
    return merged_lines

def run_one_design(design_name, design_dir, save_dir):
    print("Current Design: ", design_name)
    graph_save_path = f"{save_dir}/{design_name}_graph.pkl"
    if os.path.exists(graph_save_path):
        return

    analyzer_save_path = f"../saved_analyzer/{design_name}_analyzer.pkl"
    if os.path.exists(analyzer_save_path):
        with open (analyzer_save_path, 'rb') as f:
            ast_analysis = pickle.load(f)
    else:
        netlines = merge_lines(design_dir)
        with open(f"./tmp.v", 'w') as f:
            f.writelines(netlines)
        filelist = [design_dir]

        for f in filelist:
            if not os.path.exists(f):
                raise IOError("file not found: " + f)

        ast, directives = parse(filelist)
        print('Verilog2AST Finish!')
        ast_analysis = AST_analyzer(ast)
        
        ast_analysis.AST2Graph(ast)

        with open (analyzer_save_path, 'wb') as f:
            pickle.dump(ast_analysis, f)
    
    ast_analysis.graph_split(save_dir, design_name=design_name)
    


def run_all_designs(design_lst):
    design_dir = "../../../data_collect/data_pt/init"
    save_dir = "../saved_graph_split"
    
    for design in design_lst:
        in_dir = f"{design_dir}/{design}/{design}.init.v"
        if os.path.exists(f"{save_dir}/{design}/{design}_node_dict.pkl"):
            # with open(f"{save_dir}/{design}/{design}_node_dict.pkl", 'rb') as f:
            #     node_dict = pickle.load(f)
            # for n, node in node_dict.items():
            #     if node.tpe != 'Wire':
            #         print(node)
            continue
        run_one_design(design, in_dir, save_dir)
        print(f"Finish: {design}\n")


if __name__ == '__main__':
    run_all_designs(['DESIGN_NAME_HERE'])