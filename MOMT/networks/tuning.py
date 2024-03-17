import torch
import torch.nn.functional as F
from networks.basemodels import BaseModel
from torch import nn
from torch.nn import BatchNorm1d
from torch_geometric.nn import GATv2Conv, GCNConv, GraphNorm
from utils.utils import ACT_FN

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
        heads=1,
    ):
        super().__init__()

        self.normalisation = normalisation
        self.ds_dim = ds_dim
        self.act_fn = act_fn
        self.dropout = dropout
        self.layer_type = layer_type

        match layer_type:
            case "MLP":
                if ds_dim != 0:
                    self.ds_layer = nn.Linear(in_dim, ds_dim)
                    self.norm_layer = BatchNorm1d(ds_dim)
                    self.out_layer = nn.Linear(ds_dim, out_dim)
                else:
                    self.out_layer = nn.Linear(in_dim, out_dim)
            case "GCN":
                if ds_dim != 0:
                    self.ds_layer = GCNConv(in_dim, ds_dim)
                    self.norm_layer = GraphNorm(ds_dim)
                    self.out_layer = GCNConv(ds_dim, out_dim)
                else:
                    self.out_layer = GCNConv(in_dim, out_dim)

            case "GAT":
                if ds_dim != 0:
                    self.ds_layer = GATv2Conv(
                        in_dim, ds_dim, heads=heads, dropout=dropout
                    )
                    self.norm_layer = GraphNorm(ds_dim * heads)
                    self.out_layer = GATv2Conv(
                        ds_dim * heads,
                        out_dim,
                        heads=1,
                        dropout=dropout,
                        concat=False,
                    )
                else:
                    self.out_layer = GATv2Conv(
                        in_dim, out_dim, heads=1, dropout=dropout, concat=False
                    )

    def forward(self, x, edge_index=None, edge_weight=None):
        if self.dropout != 0:
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.layer_type == "MLP":
            if self.ds_dim != 0:
                x = self.ds_layer(x)
                if self.normalisation and x.size()[0] != 1:
                    x = self.norm_layer(x)
                x = self.act_fn(x)

            x = self.out_layer(x)

        else:
            if self.ds_dim != 0:
                x = self.ds_layer(x, edge_index, edge_weight)
                if self.normalisation and x.size()[0] != 1:
                    x = self.norm_layer(x)
                x = self.act_fn(x)

            x = self.out_layer(x, edge_index, edge_weight)
        return x


"""


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
        heads=1,
    ):
        super().__init__()

        self.normalisation = normalisation
        self.ds_dim = ds_dim
        self.act_fn = act_fn
        self.dropout = dropout
        self.layer_type = layer_type

        if ds_dim != 0:
            match layer_type:
                case "MLP":
                    self.ds_layer = nn.Linear(in_dim, ds_dim)
                    self.norm_layer = BatchNorm1d(ds_dim)
                case "GCN":
                    self.ds_layer = GCNConv(in_dim, ds_dim)
                    self.norm_layer = GraphNorm(ds_dim)
                case "GAT":
                    self.ds_layer = GATv2Conv(
                        in_dim, ds_dim, heads=heads, dropout=dropout, concat=False
                    )
                    self.norm_layer = GraphNorm(ds_dim)

            self.out_layer = nn.Linear(ds_dim, out_dim)
        else:
            self.out_layer = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index=None, edge_weight=None):
        if self.ds_dim != 0:
            if self.layer_type == "MLP":
                x = self.ds_layer(x)
                if self.normalisation and x.size()[0] != 1:
                    x = self.norm_layer(x)
                x = self.act_fn(x)
            else:
                x = self.ds_layer(x, edge_index, edge_weight)
                if self.normalisation and x.size()[0] != 1:
                    x = self.norm_layer(x)
                x = self.act_fn(x)

        x = self.out_layer(x)
        return x

"""


class CustomTuning(BaseModel):

    def __init__(self, config):
        super().__init__(config)

        self.act_fn = ACT_FN[config["act_fn"]]
        self.normalisation = config["norm"]
        self.dropout = config["dp"]
        self.layer_type = config["net_type"]

        match config["net_type"]:
            case "MLP":
                self.trunk_layer = nn.Linear(config["input_dim"], config["trunk_ls"])
                self.norm_layer = BatchNorm1d(config["trunk_ls"])
            case "GCN":
                self.trunk_layer = GCNConv(config["input_dim"], config["trunk_ls"])
                self.norm_layer = GraphNorm(config["trunk_ls"])
            case "GAT":
                self.trunk_layer = GATv2Conv(
                    config["input_dim"],
                    config["trunk_ls"],
                    heads=config["n_heads"],
                    dropout=config["dp"],
                    concat=False,
                )
                self.norm_layer = GraphNorm(config["trunk_ls"])

        if self.config["cls_loss_weight"] != 0:
            self.cls_net = CustomNet(
                config["trunk_ls"],
                config["cls_ds"],
                config["n_classes"],
                config["norm"],
                self.act_fn,
                config["dp"],
                config["net_type"],
                config["n_heads"],
            )

        if self.config["cls_loss_weight"] != 1:
            self.surv_net = CustomNet(
                config["trunk_ls"],
                config["surv_ds"],
                config["n_buckets"],
                config["norm"],
                self.act_fn,
                config["dp"],
                config["net_type"],
                config["n_heads"],
            )

    def forward(self, x, edge_index=None, edge_weight=None):

        if self.dropout != 0:
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.layer_type == "MLP":
            x = self.trunk_layer(x)
            if self.normalisation and x.size()[0] != 1:
                x = self.norm_layer(x)
            x = self.act_fn(x)
        else:
            x = self.trunk_layer(x, edge_index, edge_weight)
            if self.normalisation and x.size()[0] != 1:
                x = self.norm_layer(x)
            x = self.act_fn(x)

        if self.dropout != 0:
            x = F.dropout(x, p=self.dropout, training=self.training)

        pred_cls = None
        if self.config["cls_loss_weight"] != 0:
            pred_cls = self.cls_net.forward(x, edge_index, edge_weight)

        pred_surv = None
        if self.config["cls_loss_weight"] != 1:
            pred_surv = self.surv_net.forward(x, edge_index, edge_weight)

        return [pred_cls, pred_surv], x
