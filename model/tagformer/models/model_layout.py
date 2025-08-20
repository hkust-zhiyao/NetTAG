import torch
import torch.nn.functional as F
from torch import nn
from functools import partial
import numpy as np
import random, json
# import dgl

# from models.gnn import GCN
from models.model_graph import GCN, SGFormer
from models.gt import MLP_dec

from models.loss_fn import TripletLoss, InfoNCE

def load_config_from_json_file(config_file):
    with open(config_file, 'r') as f:
        config_dict = json.load(f)
    return config_dict

def all_to_device(lst, device):
    return (x.to(device) for x in lst)

def setup_loss_fn(loss_fn):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        # elif loss_fn == "sce":
        #     criterion = partial(sce_loss, alpha=alpha_l)
        elif loss_fn == "nll":
            criterion = nn.NLLLoss()
        elif loss_fn == "ce":
            criterion = nn.CrossEntropyLoss()
        elif loss_fn == "cos":
            criterion = nn.CosineSimilarity()
        elif loss_fn == "mae":
            criterion = nn.L1Loss()
        else:
            raise NotImplementedError
        return criterion

class Net_Encoder(nn.Module):
    def __init__(self,   
                 config,
                 device,
                 accelerator=None
                 ):
        super(Net_Encoder, self).__init__()

        self.device = device
        self.config = config
        if accelerator:
            self.accelerator = accelerator

        self.embed_dim = config['embed_dim']

        ### loss function  initialization ###
        temp_num = 0.1
        self.loss_cl = InfoNCE(temperature=temp_num)
        self.loss_gmae = setup_loss_fn("ce")
        

        ### GNN initialization ###
        self.gnn_config = load_config_from_json_file(config['gnn_config'])
        self.gcn = GCN(
            config=self.gnn_config
        )
        self.gnn = SGFormer(
            config=self.gnn_config,
            gnn=self.gcn
        )
        self.graph_proj = nn.Linear(self.gnn_config['embed_dim'], self.embed_dim)
        self.gmae_decoder = MLP_dec(
            input_dim=self.embed_dim,
            hidden_dim=512,
            num_layers=3,
            output_dim=self.gnn_config['tpe_dim'],
            activation="relu",
            norm="batchnorm"
        )
        self.encoder_to_decoder = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.enc_mask_token = nn.Parameter(torch.zeros(1, self.gnn_config['feat_dim']))

        self._mask_rate = self.gnn_config['mask_rate']
        self._drop_edge_rate = self.gnn_config['drop_edge_rate']
        self._replace_rate = self.gnn_config['replace_rate']
        self._mask_token_rate = 1 - self._replace_rate

        ### downstream task model initialization ###
        self.downstream_mlp = MLP_dec(
            input_dim=self.embed_dim,
            hidden_dim=512,
            num_layers=3,
            output_dim=1,
            activation="gelu",
            norm="batchnorm"
        )

    def forward(self, data):
        node_embeds, graph_embeds = self.graph_encode(data)
        return node_embeds, graph_embeds

    



