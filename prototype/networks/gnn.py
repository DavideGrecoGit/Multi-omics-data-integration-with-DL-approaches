import math
from torch.nn.parameter import Parameter
import torch
import torch.nn.functional as F
from torch import nn
from base_models import BaseModel
from torch_geometric.nn import GCNConv, GATv2Conv
import pandas as pd


class GraphConvolution(nn.Module):
    def __init__(self, infeas, outfeas, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = infeas
        self.out_features = outfeas
        self.weight = Parameter(torch.FloatTensor(infeas, outfeas))
        if bias:
            self.bias = Parameter(torch.FloatTensor(outfeas))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        """
        for name, param in GraphConvolution.named_parameters(self):
            if 'weight' in name:
                #torch.nn.init.constant_(param, val=0.1)
                torch.nn.init.normal_(param, mean=0, std=0.1)
            if 'bias' in name:
                torch.nn.init.constant_(param, val=0)
        """

    def forward(self, x, adj):
        x1 = torch.mm(x, self.weight)
        output = torch.mm(adj, x1)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN(BaseModel):
    def __init__(self, n_in, n_hid=64, n_out=4, dropout=0, name="MoGCN_GCN"):
        super().__init__(name)

        self.gc1 = GraphConvolution(n_in, n_hid)
        self.gc2 = GraphConvolution(n_hid, n_hid)
        self.dp1 = nn.Dropout(dropout)
        self.dp2 = nn.Dropout(dropout)
        self.fc = nn.Linear(n_hid, n_out)
        self.dropout = dropout

    def forward(self, input, adj):
        x = self.gc1(input, adj)
        x = F.elu(x)
        x = self.dp1(x)
        x = self.gc2(x, adj)
        x = F.elu(x)
        x = self.dp2(x)

        prediction = self.fc(x)

        return x, prediction

    def forward_pass(self, latent, adj_matrix, gt_classes):
        latent = latent.to(self.device)
        adj_matrix = adj_matrix.to(self.device)
        gt_classes = gt_classes.to(self.device)

        latent, prediction = self.forward(latent, adj_matrix)

        loss = F.cross_entropy(prediction, gt_classes)

        return prediction, loss

    def get_latent_space(self, dataset):
        latent = dataset.latent_space.to(self.device)
        adj_matrix = dataset.adj_matrix.to(self.device)

        latent, prediction = self.forward(latent, adj_matrix)

        return latent.detach().cpu().numpy()


class GAT(torch.nn.Module):
    """Graph Attention Network"""

    def __init__(self, dim_in, dim_h, dim_out=4, d_p=0.2, heads=8):
        super().__init__()
        self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
        self.gat2 = GATv2Conv(dim_h * heads, dim_out, heads=1)
        self.d_p = d_p

    def forward(self, x, edge_index):
        h = F.dropout(x, p=self.d_p, training=self.training)
        h = self.gat1(h, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=self.d_p, training=self.training)
        h = self.gat2(h, edge_index)
        return h, F.log_softmax(h, dim=1)

    def forward_pass(self, node_features, edge_index, gt_classes):
        df = pd.DataFrame(gt_classes, columns=["gt_classes"])
        class_weights = len(df["gt_classes"]) / df["gt_classes"].value_counts()
        class_weights = torch.tensor(class_weights.to_list(), dtype=torch.float)

        h, z = self.forward(node_features, edge_index)

        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = criterion(z, gt_classes)

        return h, loss
