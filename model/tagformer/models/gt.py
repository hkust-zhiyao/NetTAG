from torch import nn
import torch
from dgl.nn import DegreeEncoder, GraphormerLayer, PathEncoder, SpatialEncoder
from functools import partial



class Graphormer(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        # dropout = nn.Dropout(p=config["dropout"])
        dropout = config["dropout"]
        feat_dim = config["feat_dim"]
        self.num_heads = num_heads = config["num_heads"]
        self.encoding = config["encoding"]
        activation_fn = config["activation"]
        pre_layernorm = config["pre_layernorm"]
        ffn_embedding_dim = config["num_hidden"]
        out_dim = config["embed_dim"]
        edge_dim = config["edge_dim"]
        max_degree = config["max_degree"]
        num_spatial = config["num_spatial"]
        multi_hop_max_dist = config["multi_hop_max_dist"]
        num_layers = config["num_layers"]

        # self.atom_encoder = nn.Embedding(
        #     num_atoms + 1, embedding_dim, padding_idx=0
        # )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, feat_dim))
        self.graph_token = nn.Embedding(1, feat_dim)

        self.degree_encoder = DegreeEncoder(
            max_degree=max_degree, embedding_dim=feat_dim
        )

        self.path_encoder = PathEncoder(
            max_len=multi_hop_max_dist,
            feat_dim=edge_dim,
            num_heads=num_heads,
        )

        self.spatial_encoder = SpatialEncoder(
            max_dist=num_spatial, num_heads=num_heads
        )
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        self.emb_layer_norm = nn.LayerNorm(feat_dim)

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                GraphormerLayer(
                    feat_size=feat_dim,
                    hidden_size=ffn_embedding_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    activation=create_activation(activation_fn),
                    norm_first=pre_layernorm,
                )
                for _ in range(num_layers)
            ]
        )

        # map graph_rep to num_classes
        self.lm_head_transform_weight = nn.Linear(
            feat_dim, feat_dim
        )
        self.layer_norm = nn.LayerNorm(feat_dim)
        self.activation_fn = create_activation(activation_fn)
        self.embed_out = nn.Linear(feat_dim, out_dim, bias=False)
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(out_dim))

        ### add a fc layer to map the graph representation to the output (from embedding_dim to 256)
        self.fc = nn.Linear(feat_dim, out_dim, bias=False)

    def reset_output_layer_parameters(self):
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
        self.embed_out.reset_parameters()

    def forward(
        self,
        batched_data,
        mask=None,
    ):  
        (attn_mask,node_feat,in_degree,out_degree,path_data,dist) = batched_data
        num_graphs, max_num_nodes, _ = node_feat.shape
        deg_emb = self.degree_encoder(torch.stack((in_degree, out_degree)))

        if self.encoding:
            node_feat_comb = node_feat + deg_emb
            graph_token_feat = self.graph_token.weight.unsqueeze(0).repeat(
                num_graphs, 1, 1
            )
            x = torch.cat([graph_token_feat, node_feat_comb], dim=1)

        attn_bias = torch.zeros(
            num_graphs,
            max_num_nodes + 1,
            max_num_nodes + 1,
            self.num_heads,
            device=dist.device,
        )
        path_encoding = self.path_encoder(dist, path_data)
        spatial_encoding = self.spatial_encoder(dist)
        attn_bias[:, 1:, 1:, :] = path_encoding + spatial_encoding

        t = self.graph_token_virtual_distance.weight.reshape(
            1, 1, self.num_heads
        )

        attn_bias[:, 1:, 0, :] = attn_bias[:, 1:, 0, :] + t
        attn_bias[:, 0, :, :] = attn_bias[:, 0, :, :] + t

        if (mask is not None) and self.encoding:
            x = x[:, ~mask]  # [n graph, n non-masked nodes, n hidden]
            attn_bias = attn_bias[:, ~mask, :, :][:, :, ~mask, :] # [n graph, n non-masked nodes, n non-masked nodes, n heads,]
            attn_mask = attn_mask[:, ~mask, :][:, :, ~mask]

        x = self.emb_layer_norm(x)

        for layer in self.layers:
            x = layer(
                x,
                attn_mask=attn_mask,
                attn_bias=attn_bias,
            )

        graph_rep = x[:, 0, :]
        node_rep = x[:, 1:, :]
        
        node_rep = self.embed_out(node_rep)
        graph_rep = self.embed_out(graph_rep) + self.lm_output_learned_bias

        return node_rep, graph_rep

def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")

def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "graphnorm":
        return partial(NormLayer, norm_type="groupnorm")
    else:
        return nn.Identity

class NormLayer(nn.Module):
    def __init__(self, hidden_dim, norm_type):
        super().__init__()
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm_type == "graphnorm":
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
        else:
            raise NotImplementedError
        
    def forward(self, graph, x):
        tensor = x
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor

        batch_list = graph.batch_num_nodes
        batch_size = len(batch_list)
        batch_list = torch.Tensor(batch_list).long().to(tensor.device)
        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean * self.mean_scale

        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias


class MLP_dec(nn.Module):
    """MLP with linear output"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, activation="relu", norm="batchnorm"):
        super(MLP_dec, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.norms = torch.nn.ModuleList()
            self.activations = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.norms.append(create_norm(norm)(hidden_dim))
                self.activations.append(create_activation(activation))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = self.norms[i](self.linears[i](h))
                h = self.activations[i](h)
            return self.linears[-1](h)
        
class MLP_regression(nn.Module):
    """MLP with linear output"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, activation="relu", norm="batchnorm"):
        super(MLP_regression, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.norms = torch.nn.ModuleList()
            self.activations = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.norms.append(create_norm(norm)(hidden_dim))
                self.activations.append(create_activation(activation))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = self.norms[i](self.linears[i](h))
                h = self.activations[i](h)
            return self.linears[-1](h)