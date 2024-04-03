import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.logging import log
from torch_geometric.nn import GATv2Conv, GCNConv

from networks.basemodel import ClsBaseModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MoGCN(ClsBaseModel):
    def __init__(self, config, hidden_dim=64, dp=0):
        super().__init__(config)
        self.config = config
        self.dp = dp

        self.conv1 = GCNConv(config["input_dim"], hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.cls = nn.Linear(hidden_dim, config["n_classes"])

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.elu(x)
        x = F.dropout(x, p=self.dp, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.elu(x)
        y = F.dropout(x, p=self.dp, training=self.training)
        y = self.cls(y)
        return y, x
