import torch
import torch.nn.functional as F
from networks.basemodels import BaseModel
from torch import nn
from torch.nn import BatchNorm1d
from torch_geometric.logging import log
from torch_geometric.nn import GATv2Conv, GCNConv, GraphNorm
from utils.utils import ACT_FN, EarlyStopper

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomNet(nn.Module):

    def __init__(
        self,
        in_dim,
        ds_dim,
        out_dim,
        normalisation,
        act_fn,
        dropout,
        layer_type,
        is_trunk,
        heads=1,
    ):
        super().__init__()

        self.normalisation = normalisation
        self.ds_dim = ds_dim
        self.act_fn = act_fn
        self.dropout = dropout
        self.layer_type = layer_type
        self.is_trunk = is_trunk

        match layer_type:
            case "MLP":
                if ds_dim != 0:
                    self.ds_layer = nn.Linear(in_dim, ds_dim)
                    self.norm_layer = BatchNorm1d(ds_dim)
                    self.out_layer = nn.Linear(ds_dim, out_dim)
                else:
                    self.out_layer = nn.Linear(in_dim, out_dim)

                if is_trunk:
                    self.norm_layer_2 = BatchNorm1d(out_dim)
            case "GCN":
                if ds_dim != 0:
                    self.ds_layer = GCNConv(in_dim, ds_dim)
                    self.norm_layer = GraphNorm(ds_dim)
                    self.out_layer = GCNConv(ds_dim, out_dim)
                else:
                    self.out_layer = GCNConv(in_dim, out_dim)

                if is_trunk:
                    self.norm_layer_2 = GraphNorm(out_dim)

            case "GAT":
                if ds_dim != 0:
                    self.ds_layer = GATv2Conv(
                        in_dim, ds_dim, heads=heads, dropout=dropout
                    )
                    self.norm_layer = GraphNorm(ds_dim * heads)
                    self.out_layer = GATv2Conv(
                        ds_dim * heads, out_dim, heads=1, dropout=dropout, concat=False
                    )
                else:
                    self.out_layer = GATv2Conv(
                        in_dim, out_dim, heads=1, dropout=dropout, concat=False
                    )

                if is_trunk:
                    self.norm_layer_2 = GraphNorm(out_dim)

    def forward(self, x, edge_index=None, edge_weight=None):
        if self.dropout != 0:
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.layer_type == "MLP":
            if self.ds_dim != 0:
                x = self.ds_layer(x)
                if self.normalisation and x.size()[0] != 1:
                    x = self.norm_layer(x)
                x = self.act_fn(x)

            if self.dropout != 0 and self.is_trunk:
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.out_layer(x)

        else:
            if self.ds_dim != 0:
                x = self.ds_layer(x, edge_index, edge_weight)
                if self.normalisation and x.size()[0] != 1:
                    x = self.norm_layer(x)
                x = self.act_fn(x)

            if self.dropout != 0 and self.is_trunk:
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.out_layer(x, edge_index, edge_weight)

        if self.is_trunk:
            if self.normalisation and x.size()[0] != 1:
                x = self.norm_layer_2(x)
            x = self.act_fn(x)

        return x


class CustomTuning(BaseModel):

    def __init__(self, config):
        super().__init__(config)

        self.act_fn = ACT_FN[config["act_fn"]]

        self.trunk = CustomNet(
            config["input_dim"],
            0,
            config["trunk_ls"],
            self.config["norm"],
            self.act_fn,
            config["dp"],
            config["net_type"],
            True,
            config["n_heads"],
        )

        if self.config["cls_loss_weight"] != 0:
            self.cls_net = CustomNet(
                config["trunk_ls"],
                config["cls_ds"],
                config["n_classes"],
                self.config["norm"],
                self.act_fn,
                config["dp"],
                config["net_type"],
                False,
                config["n_heads"],
            )

        if self.config["cls_loss_weight"] != 1:
            self.surv_net = CustomNet(
                config["trunk_ls"],
                config["surv_ds"],
                config["n_buckets"],
                self.config["norm"],
                self.act_fn,
                config["dp"],
                config["net_type"],
                False,
                config["n_heads"],
            )

    def forward(self, x, edge_index=None, edge_weight=None):

        x = self.trunk.forward(x, edge_index, edge_weight)

        pred_cls = None
        if self.config["cls_loss_weight"] != 0:
            pred_cls = self.cls_net.forward(x, edge_index, edge_weight)

        pred_surv = None
        if self.config["cls_loss_weight"] != 1:
            pred_surv = self.surv_net.forward(x, edge_index, edge_weight)

        return [pred_cls, pred_surv], x
