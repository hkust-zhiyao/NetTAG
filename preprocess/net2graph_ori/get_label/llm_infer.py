import torch
import numpy as np
import pickle, json, time, re, sys, os
import networkx as nx
from multiprocessing import Pool
import dgl
from dgl import from_networkx
import dgl
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import requests
from transformers import pipeline

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

def run_one_ep(design_name, ep):
    # print(design_name, ep)
    folder_dir = f'/home/coguest5/hdl_fusion/data_collect/vlg/data/ep_vlg'
    with open(f'{folder_dir}/{design_name}/{ep}.v', 'r') as f:
            lines = f.readlines()
    documents = ""
    ### get output signal name
    out_sig = ""
    for line in lines:
        out_re = re.findall(r'(\s+)(\S+)(\s+)<=', line)
        if out_re:
            out_sig = out_re[0][1]
            break
    if not out_sig:
        assert False, "No output signal found!"
    for line in lines:
        line = re.sub(r'module coi ', f'module {out_sig} ', line)
        line = re.sub(r'\n', '', line)

        documents += line
    # print(documents)
    return documents
    





def get_dataset(design_lst):
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    access_token = "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"

    pipe = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    # device_map="auto",
    device="cuda",
    )
    

    for design in design_lst:
        print("Current design: ", design)
        with open (f"/home/coguest5/hdl_fusion/data_collect/label/ep_lst/{design}.json", 'r') as f:
            reg_lst = json.load(f)


        save_dir = f"../rtl_func_ori/{design}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    
        time_start = time.time()

        for ep in reg_lst:
            if os.path.exists(f"../rtl_func_ori/{design}/{ep}.txt"):
                continue
            else:
                print("     Current ep: ", ep)
            documents = run_one_ep(design, ep)


            try:
                messages = [
                {"role": "system", "content": "You are a professional VLSI designer and an expert at Verilog coding."},
                # {"role": "user", "content": f"I will provide you with a combinational Verilog design with multiple input and single output. Please use less than 300 words summarize the functionality of this Verilog design (based on the input/output/intermediate signal names, the computational operations, etc.). Here is the Verilog design: {documents}"},    
                {"role": "user", "content": f"I will provide you with a combinational Verilog design with multiple input and single output. Please use less than 450 words summarize this Verilog design, specifically, first summarize the entire functionality, and then describe the implementation details. Here is the Verilog design: {documents}"},   
                # {"role": "user", "content": f"who are you"},                          
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
                # print(assistant_response)
                with open(f"../rtl_func_ori/{design}/{ep}.txt", 'w') as f:
                    f.write(assistant_response)
            except:
                print("OOM")
                with open(f"../oom/{design}_{ep}.txt", 'w') as f:
                    f.write("OOM")
            # input()
            
            
        time_end = time.time()
        print("Time used: ", time_end-time_start)
        with open(f"../runtime/{design}.txt", 'w') as f:
            f.write(str(time_end-time_start))
    

if __name__ == '__main__':

    global design_lst_all
    with open("/home/coguest5/hdl_fusion/models/pretrain_all/dataset/dataset_js/design_all.json", 'r') as f:
        design_lst_all = json.load(f)
    

    idx = 0
    design_lst = design_lst_all[5*idx:5*idx+5]

    get_dataset(design_lst)