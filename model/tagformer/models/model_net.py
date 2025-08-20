import torch
import torch.nn.functional as F
from torch import nn
from functools import partial
import numpy as np
import random, json, copy
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


        ### Cross-stage alignment initialization ###
        self.rtl_proj = nn.Linear(4096, self.embed_dim, bias=False)
        self.layout_config = copy.deepcopy(self.gnn_config)
        self.layout_config['feat_dim'] = 8
        self.layout_gcn = GCN(
            config=self.layout_config
        )
        self.layout_proj = SGFormer(
            config=self.layout_config,
            gnn=self.layout_gcn
        )

        ### downstream task model initialization ###
        self.downstream_mlp = MLP_dec(
            input_dim=self.embed_dim,
            hidden_dim=512,
            num_layers=3,
            output_dim=1,
            activation="gelu",
            norm="batchnorm"
        )

        self.node_num_cnt_mlp = MLP_dec(
            input_dim=self.embed_dim,
            hidden_dim=128,
            num_layers=3,
            output_dim=17,
            activation="gelu",
            norm="layernorm"
        )



    def forward(self, data, mode = 'pretrain'):
        self.mode = mode
        if self.mode == 'pretrain':
            return self.pretrain_forward(data)
        elif self.mode == 'pretrain_net':
            return self.pretrain_forward_net(data)
        elif self.mode == 'infer':
            return self.finetune_forward(data)
        else:
            raise ValueError("Invalid mode")

    def graph_encode(self, graph_data):
        node_rep, graph_rep = self.gnn(graph_data.to(self.device))
        embeds = F.normalize(self.graph_proj(graph_rep),dim=-1)

        return node_rep, embeds

    def finetune_forward(self, data):
        node_embeds, graph_embeds = self.graph_encode(data)
        return node_embeds, graph_embeds
    
    def pretrain_forward_net(self, data):
        graph_ori, graph_pos = data[0], data[1]


        _, graph_embeds_ori = self.graph_encode(graph_ori)
        
        loss_cl_net = self.pretrain_task_cl_net(graph_embeds_ori, graph_pos)

        loss_gmae = self.pretrain_task_gmae(graph_ori)

        loss_node_cnt = self.pretrain_node_num_count(graph_embeds_ori, graph_ori.tpe_cnt)

        # print(f"loss_cl_net: {loss_cl_net}, loss_cl_rtl: {loss_cl_rtl}, loss_cl_layout: {loss_cl_layout}, loss_gmae: {loss_gmae}, loss_node_cnt: {loss_node_cnt}")
        loss = 1.0*loss_cl_net + 1.0*loss_gmae + 0.2*loss_node_cnt

        return loss, loss_cl_net, loss_gmae, loss_node_cnt

    
    def pretrain_forward(self, data):
        rtl_emb = data[0]
        graph_ori, graph_pos = data[1], data[2]
        layout = data[3]


        _, graph_embeds_ori = self.graph_encode(graph_ori)
        
        loss_cl_net = self.pretrain_task_cl_net(graph_embeds_ori, graph_pos)

        loss_cl_rtl = self.pretrain_task_cl_rtl(rtl_emb, graph_embeds_ori)

        loss_cl_layout = self.pretrain_task_cl_layout(layout, graph_embeds_ori)

        loss_gmae = self.pretrain_task_gmae(graph_ori)

        loss_node_cnt = self.pretrain_node_num_count(graph_embeds_ori, graph_ori.tpe_cnt)

        

        # print(f"loss_cl_net: {loss_cl_net}, loss_cl_rtl: {loss_cl_rtl}, loss_cl_layout: {loss_cl_layout}, loss_gmae: {loss_gmae}, loss_node_cnt: {loss_node_cnt}")
        loss = 1.0*loss_cl_net + 1.0*loss_gmae + 0.2*loss_node_cnt + 0.1*(loss_cl_rtl + loss_cl_layout)

        return loss, loss_cl_net, loss_gmae, loss_node_cnt, loss_cl_rtl, loss_cl_layout

    def pretrain_task_cl_rtl(self, rtl_emb, graph_embeds_ori):
        ## (n, 1, 4096) --> (n, 4096)
        rtl_emb = rtl_emb.squeeze(1)
        rtl_emb = self.rtl_proj(rtl_emb.to(self.device).to(torch.float32))
        if rtl_emb.dtype != graph_embeds_ori.dtype:
            rtl_emb = rtl_emb.to(graph_embeds_ori.dtype)
        loss = self.loss_cl(rtl_emb, graph_embeds_ori)
        return loss
    
    def pretrain_task_cl_layout(self, layout_data, graph_embeds_ori):
        _, layout_graph_emb = self.layout_proj(layout_data.to(self.device))
        loss = self.loss_cl(layout_graph_emb, graph_embeds_ori)
        return loss


    def pretrain_task_cl_net(self, graph_embeds_ori, graph_pos):
        # _, graph_embeds_ori = self.graph_encode(graph_ori)
        _, graph_embeds_pos = self.graph_encode(graph_pos)
        loss = self.loss_cl(graph_embeds_ori, graph_embeds_pos)
        return loss


    def pretrain_task_gmae(self, g):
        x = g.x.to(self.device)
        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self._mask_rate)

        
        use_g = pre_use_g

        enc_rep, graph_emb = self.gnn(use_g.to(self.device), use_x)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)
        recon = self.gmae_decoder(rep)


        ### x: 8bits phy_vec + 17 bits type_vec + 2048 bits text_vec
        ### reconstruct the type_vec (one-hot)
        x_init = x[mask_nodes][:, 8:25]
        x_rec = recon[mask_nodes]

        x_init = torch.argmax(x_init, dim=1)

        loss = self.loss_gmae(x_rec, x_init)
        return loss

    def encoding_mask_noise(self, g, x, mask_rate=0.3):
        num_nodes = g.num_nodes
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]

        if self._replace_rate > 0:
            num_noise_nodes = 1 if int(self._replace_rate * num_mask_nodes) < 1 else int(self._replace_rate * num_mask_nodes)
            # num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x = out_x.to(self.device)
        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()

        return use_g, out_x, (mask_nodes, keep_nodes)


    # def drop_edge(self, graph, drop_rate, return_edges=False):
    #     if drop_rate <= 0:
    #         return graph

    #     n_node = graph.num_nodes()
    #     edge_mask = self.mask_edge(graph, drop_rate)
    #     src = graph.edges()[0]
    #     dst = graph.edges()[1]

    #     nsrc = src[edge_mask]
    #     ndst = dst[edge_mask]

    #     ng = dgl.graph((nsrc, ndst), num_nodes=n_node)
    #     ng = ng.add_self_loop()

    #     dsrc = src[~edge_mask]
    #     ddst = dst[~edge_mask]

    #     if return_edges:
    #         return ng, (dsrc, ddst)
    #     return ng

    def mask_edge(self, graph, mask_prob):
        E = graph.num_edges()
        mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
        masks = torch.bernoulli(1 - mask_rates)
        mask_idx = masks.nonzero().squeeze(1)
        return mask_idx


    def pretrain_node_num_count(self, graph_embeds, label_node_num):
        pred = self.node_num_cnt_mlp(graph_embeds)
        loss = F.mse_loss(pred, label_node_num)*10**(-5)
        ## make sure loss_node_cnt is less than 0.5 and greater than 0.1
        loss = torch.clamp(loss, 0.05, 0.5)
        return loss