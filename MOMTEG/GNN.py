#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/8/8 16:20
# @Author  : Li Xiao

import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch_geometric.logging import log
from torch_geometric.nn import GATv2Conv, GCNConv


class GNN(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    def train_loop(self, data):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config["gnn_lr"],
            weight_decay=self.config["gnn_wd"],
        )

        for epoch in range(1, self.config["epochs"] + 1):
            self.train()

            pred, _ = self.forward(data.x, data.edge_index, data.edge_attr)
            loss = F.cross_entropy(pred[data.train_mask], data.y[data.train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_acc, train_f1 = self.validate(data, data.train_mask)
            val_acc, val_f1 = self.validate(data, data.val_mask)

            log(
                Epoch=epoch,
                Loss=loss,
                Train_Acc=train_acc,
                Val_Acc=val_acc,
                Train_f1=train_f1,
                Val_f1=val_f1,
            )

        return val_acc, val_f1

    @torch.no_grad()
    def validate(self, data, mask):
        self.eval()
        pred, _ = self.forward(data.x, data.edge_index, data.edge_attr)
        pred = pred.argmax(dim=-1)

        f1 = f1_score(
            data.y[mask].cpu().numpy(),
            pred[mask].cpu().numpy(),
            average="macro",
        )
        acc = accuracy_score(data.y[mask].cpu().numpy(), pred[mask].cpu().numpy())

        return acc, f1

    @torch.no_grad()
    def get_latent_space(self, data, save_path=None):
        self.eval()
        pred, latent = self.forward(data.x, data.edge_index)
        pred = pred.argmax(dim=-1)

        latent = latent.cpu().numpy()

        if save_path:
            np.savetxt(
                os.path.join(save_path),
                latent,
                delimiter=",",
            )

        return latent

    @torch.no_grad()
    def get_predictions(self, data, save_dir=None, mask=None):
        self.eval()
        pred, _ = self.forward(data.x, data.edge_index)
        pred = pred.argmax(dim=-1)
        pred = pred.cpu().numpy()

        if mask is not None:
            pred = pred[mask]

        if save_dir is not None:
            data_to_save = {"GT": data.y.cpu().numpy(), "Pred": pred}
            df = pd.DataFrame.from_dict(data_to_save)
            df.to_csv(os.path.join(save_dir, "predictions.csv"), index=False)

        return pred


class GCN(GNN):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.conv1 = GCNConv(config["gnn_input_dim"], config["gnn_latent_dim"])
        self.conv2 = GCNConv(config["gnn_latent_dim"], config["gnn_latent_dim"])
        self.fc = nn.Linear(config["gnn_latent_dim"], config["n_classes"])

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.elu(x)
        x = F.dropout(x, p=self.config["gnn_dp"], training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.elu(x)
        y = F.dropout(x, p=self.config["gnn_dp"], training=self.training)
        y = self.fc(y)
        return y, x


class GAT(GNN):
    def __init__(
        self, in_channels, hidden_channels, out_channels, dropout=0.5, n_heads=8
    ):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=n_heads)
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels, heads=n_heads)
        self.fc = GATv2Conv(hidden_channels, out_channels, heads=n_heads)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.elu(x)
        y = F.dropout(x, p=self.dropout, training=self.training)
        # y = self.fc(y)
        y = self.fc(y, edge_index, edge_weight)

        return y, x
