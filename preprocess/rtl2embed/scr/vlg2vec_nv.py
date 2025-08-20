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


def run_one_ep(design_name, ep):
    # print(design_name, ep)
    folder_dir = f'/home/usr/xxx/data_collect/vlg/data/ep_vlg'
    with open(f'{folder_dir}/{design_name}/{ep}.v', 'r') as f:
            lines = f.readlines()
    documents = ""
    for line in lines:
        line = re.sub(r'\n', '', line)
        documents += line

    return documents


def get_dataset(design_lst):
    # model = SentenceTransformer("intfloat/e5-mistral-7b-instruct")
    model = AutoModel.from_pretrained('nvidia/NV-Embed-v1', trust_remote_code=True)
    # model.max_seq_length = 4096
    max_length = 1024

    for design in design_lst:
        print("Current design: ", design)
        with open (f"/home/usr/hdl_fusion/data_collect/label/ep_lst/{design}.json", 'r') as f:
            reg_lst = json.load(f)
        # task_name_to_instruct = {"example": "Please act as a professional VLSI Verilog designer. ",}

        # query_prefix = "Instruct: "+task_name_to_instruct["example"]+"\nQuery: "
        # queries = [
        # "analyze the functionality of the given Verilog code and retrieve the most similar design code"
        # ]

        save_dir = f"../rtl_emb/{design}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        else:
            os.system(f"rm -rf {save_dir}")
            os.makedirs(save_dir)
    
        time_start = time.time()
        for ep in reg_lst:
            documents = run_one_ep(design, ep)
            document_embeddings = model.encode(documents, max_length=max_length)
            document_embeddings = F.normalize(document_embeddings, p=2, dim=1)
            print(document_embeddings)
            exit()
            with open(f"{save_dir}/{ep}.pkl", 'wb') as f:
                pickle.dump(document_embeddings, f)
        time_end = time.time()
        print("Time used: ", time_end-time_start)
        with open(f"../runtime/{design}.txt", 'w') as f:
            f.write(str(time_end-time_start))
    

if __name__ == '__main__':

    global design_lst_all
    with open("/home/usr/hdl_fusion/dataset_js/design_all.json", 'r') as f:
        design_lst_all = json.load(f)
    
    get_dataset(design_lst_all)