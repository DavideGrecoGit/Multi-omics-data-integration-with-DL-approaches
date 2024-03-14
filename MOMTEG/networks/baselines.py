import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from networks.basemodels import BaseModel
from networks.losses import MTLR_loss, cross_entropy_loss, l2_loss
from sklearn.metrics import accuracy_score, f1_score
from sksurv.metrics import concordance_index_censored
from torch import nn
from torch.nn import BatchNorm1d
from torch_geometric.logging import log
from torch_geometric.nn import GATv2Conv, GCNConv, GraphNorm
from utils.data import get_tri_matrix
from utils.utils import ACT_FN, EarlyStopper


class MoGCN_GCN(BaseModel):
    def __init__(self, config, hidden_dim=64, dp=0.5):
        super().__init__(config)
        self.config = config
        self.dp = dp

        self.conv1 = GCNConv(config["input_dim"], hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, config["n_classes"])

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.elu(x)
        x = F.dropout(x, p=self.dp, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.elu(x)
        y = F.dropout(x, p=self.dp, training=self.training)
        y = self.fc(y)
        return [y, None], x
