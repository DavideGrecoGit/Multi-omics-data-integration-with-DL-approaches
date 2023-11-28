import math
from torch.nn.parameter import Parameter
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, GATv2Conv
import pandas as pd


class GAT(torch.nn.Module):
    """Graph Attention Network"""

    def __init__(self, dim_in, dim_h, dim_out=2, d_p=0.2, heads=8):
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
        self.train()

        df = pd.DataFrame(gt_classes, columns=["gt_classes"])
        class_weights = len(df["gt_classes"]) / df["gt_classes"].value_counts()
        class_weights = torch.tensor(class_weights.to_list(), dtype=torch.float)

        h, z = self.forward(node_features, edge_index)

        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = criterion(z, gt_classes)

        return h, loss

    def infer(self, node_features, edge_index):
        self.eval()
        with torch.no_grad():
            h, z = self.forward(node_features, edge_index)

        return z
