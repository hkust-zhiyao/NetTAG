import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv, AvgPooling, SAGEConv
from dgl import mean_nodes

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

class GCN_dgl(nn.Module):
    def __init__(
        self,
        config,
    ):
        super(GCN_dgl, self).__init__()
        self.num_layers = config['num_layers']
        in_dim = config['feat_dim']
        num_hidden = config['num_hidden']
        norm = config['norm']
        activation = config['activation']
        out_dim = config['embed_dim']
        self.gcn_layers = nn.ModuleList()

        self.gcn_layers.append(GraphConv(
            in_dim, num_hidden, norm=norm, activation=create_activation(activation), allow_zero_in_degree=True))
        # hidden layers
        for l in range(1, self.num_layers - 1):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gcn_layers.append(GraphConv(
                num_hidden, num_hidden,  norm=norm, activation=create_activation(activation), allow_zero_in_degree=True))
        # output projection
        self.gcn_layers.append(GraphConv(
            num_hidden, out_dim, activation=create_activation(activation), norm=norm, allow_zero_in_degree=True))


    def forward(self, g, feature, return_hidden=False):
        h = feature
        # h = inputs
        hidden_list = []
        for l in range(3):
            h = self.gcn_layers[l](g, h)
            hidden_list.append(h)


        ### get graph-level embedding ###
        avgpool = AvgPooling()
        hg = avgpool(g, h)

        if return_hidden:
            return h, hidden_list, hg
        else:
            return h, hg
