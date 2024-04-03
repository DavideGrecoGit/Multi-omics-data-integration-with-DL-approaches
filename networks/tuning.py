import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sksurv.metrics import concordance_index_ipcw, integrated_brier_score
from torch import nn
from torch.nn import BatchNorm1d
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.logging import log
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, GraphNorm

from networks.basemodel import ClsBaseModel
from utils.utils import ACT_FN, EarlyStopper

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomBlock(nn.Module):

    def __init__(
        self, in_dim, out_dim, normalisation, act_fn, dropout, layer_type, heads=1
    ):
        super().__init__()

        self.normalisation = normalisation
        self.out_dim = out_dim
        self.act_fn = act_fn
        self.dropout = dropout
        self.layer_type = layer_type

        if layer_type == "MLP":
            self.ds_layer = nn.Linear(in_dim, out_dim)
            self.norm_layer = BatchNorm1d(out_dim)
        else:
            match layer_type:
                case "GCN":
                    self.ds_layer = GCNConv(in_dim, out_dim)
                case "GAT":
                    self.ds_layer = GATConv(
                        in_dim, out_dim, heads=heads, dropout=dropout, concat=False
                    )

                case "GATv2":
                    self.ds_layer = GATv2Conv(
                        in_dim, out_dim, heads=heads, dropout=dropout, concat=False
                    )

            self.norm_layer = GraphNorm(out_dim)

    def forward(self, x, edge_index=None, edge_weight=None):
        if self.dropout != 0:
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.layer_type == "MLP":
            x = self.ds_layer(x)
        else:
            x = self.ds_layer(x, edge_index, edge_weight)

        if self.normalisation and x.size()[0] != 1:
            x = self.norm_layer(x)
        if self.act_fn is not None:
            x = self.act_fn(x)

        return x


class CustomClsTuning(ClsBaseModel):

    def __init__(self, config):
        super().__init__(config)

        self.act_fn = ACT_FN[config["act_fn"]]
        self.normalisation = config["norm"]
        self.dropout = config["dp"]
        self.layer_type = config["net_type"]

        self.layer_1 = CustomBlock(
            config["input_dim"],
            config["ds"],
            config["norm"],
            self.act_fn,
            config["dp"],
            config["net_type"],
            config.get("n_heads", 1),
        )

        self.layer_2 = CustomBlock(
            config["ds"],
            config["ls"],
            config["norm"],
            self.act_fn,
            config["dp"],
            config["net_type"],
            config.get("n_heads", 1),
        )

        self.layer_cls = CustomBlock(
            config["ls"],
            config["n_classes"],
            False,
            None,
            config["dp"],
            config["cls_type"],
            config.get("n_heads", 1),
        )

    def forward(self, x, edge_index=None, edge_weight=None):

        x = self.layer_1(x, edge_index, edge_weight)
        x = self.layer_2(x, edge_index, edge_weight)
        pred_cls = self.layer_cls(x, edge_index, edge_weight)

        return pred_cls, x
