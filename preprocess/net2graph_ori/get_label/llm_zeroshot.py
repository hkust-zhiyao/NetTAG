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

        llm_input = f"Please reason the functionality of the given netlist gate, and provide the functionality of the gate (only four possible types: adder|multiplier|subtractor|comparator). Here is the netlist gate: {n_text}, and here is the 2-hop fan-in nodes: {in_text}, and here is the 2-hop fan-out nodes: {out_text}"
        llm_infer(llm_input, label)
        input()

def llm_infer(in_text, label):
    messages = [
    {"role": "system", "content": "You are a professional VLSI designer and an expert at Verilog coding and netlist analysis. Please provide the answer step by step."},
    {"role": "user", "content": in_text},                        
    ]

    terminators = [
    pipe.tokenizer.eos_token_id,
    pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipe(
    messages,
    max_new_tokens=512,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    )
    assistant_response = outputs[0]["generated_text"][-1]["content"]
    print(in_text)
    print('Label: ', label)
    print(assistant_response)


    

def run_all_designs():

    ### load llm ###
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    access_token = "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"

    global pipe
    pipe = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    # device_map="auto",
    device="cuda",
    )



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