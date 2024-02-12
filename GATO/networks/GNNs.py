import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.logging import log
from torch_geometric.nn import GATv2Conv, GraphNorm
from sklearn.metrics import accuracy_score, f1_score
import os
import numpy as np
from torch_geometric.nn import GCNConv


class GAT(torch.nn.Module):
    def __init__(self, params):
        super().__init__()

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.activation = params.activation_fn
        self.dense_dim = params.dense_dim
        self.d_p = params.d_p
        # self.norm = GraphNorm(params.input_dim)

        if params.dense_dim > 0:
            self.conv_dense = GATv2Conv(
                params.input_dim,
                params.dense_dim,
                heads=params.heads,
                concat=False,
                dropout=params.d_p,
                edge_dim=1,
            )

            self.conv_latent = GATv2Conv(
                params.dense_dim,
                params.latent_dim,
                heads=params.heads,
                concat=False,
                dropout=params.d_p,
                edge_dim=1,
            )
        else:
            self.conv_latent = GATv2Conv(
                params.input_dim,
                params.latent_dim,
                heads=params.heads,
                dropout=params.d_p,
                edge_dim=1,
            )

        # self.norm_cls = GraphNorm(params.latent_dim * params.heads)
        self.conv_cls = GATv2Conv(
            params.latent_dim,
            params.n_classes,
            heads=1,
            concat=False,
            dropout=params.d_p,
            edge_dim=1,
        )

        # self.fc = nn.Linear(params.latent_dim * params.heads, params.n_classes)

    def forward(self, x, edge_index, edge_attr=None):
        # x = self.norms[i](x)
        if self.dense_dim > 0:
            x = F.dropout(x, p=self.d_p, training=self.training)
            x = self.conv_dense(x=x, edge_index=edge_index, edge_attr=edge_attr)
            x = self.activation(x)

        # x = self.norm(x)
        x = F.dropout(x, p=self.d_p, training=self.training)
        x = self.conv_latent(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = self.activation(x)

        # x = self.norm_cls(x)
        x = F.dropout(x, p=self.d_p, training=self.training)
        y = self.conv_cls(x=x, edge_index=edge_index, edge_attr=edge_attr)
        # y = self.fc(x)

        return y, x

    def train_loop(self, data, optimizer, epochs):
        for epoch in range(1, epochs + 1):
            self.train()
            out, _ = self.forward(data.x, data.edge_index, data.edge_attr)
            loss = F.cross_entropy(
                out[data.train_mask],
                data.y[data.train_mask],
                weight=data.class_weights,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_acc, train_f1 = self.validate(data, data.train_mask)
            val_acc, val_f1 = self.validate(data, data.test_mask)
            # if epoch % 20 == 0:
            log(
                Epoch=epoch,
                Loss=loss,
                Train_Acc=train_acc,
                Val_Acc=val_acc,
                Train_f1=train_f1,
                Val_f1=val_f1,
            )

    @torch.no_grad()
    def validate(self, data, mask):
        self.eval()
        pred, _ = self.forward(data.x, data.edge_index)
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
    def get_predictions(self, data, mask=None):
        self.eval()
        pred, _ = self.forward(data.x, data.edge_index)
        pred = pred.argmax(dim=-1)

        if mask is not None:
            return pred[mask].cpu().numpy()

        return pred.cpu().numpy()


class Params_GNN:
    def __init__(
        self,
        input_dim,
        dense_dim,
        latent_dim,
        n_edges=7000,
        n_classes=5,
        heads=8,
        lr=0.001,
        weight_decay=0.0001,
        epochs=150,
        loss_fn=nn.MSELoss(),
        d_p=0.2,
        activation_fn=nn.ELU(),
        remove_unknown=True,
        edge_dim=1,
        edge_attr=False,
    ):
        self.remove_unknown = remove_unknown
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.input_dim = input_dim
        self.dense_dim = dense_dim
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.d_p = d_p
        self.activation_fn = activation_fn
        self.heads = heads
        self.n_edges = n_edges
        self.edge_dim = edge_dim
        self.edge_attr = edge_attr

    def save_parameters(self, save_dir):
        with open(os.path.join(save_dir, "parameters.txt"), "w") as f:
            f.write(f"Remove_unknown = {self.remove_unknown}\n")
            f.write(f"N_edges = {self.n_edges}\n")
            f.write(f"Edge_attributes = {self.edge_attr}\n")
            f.write(f"Edge_dim = {self.edge_dim}\n")
            f.write(f"Input_dim = {self.input_dim}\n")
            f.write(f"Dense_dim = {self.dense_dim}\n")
            f.write(f"Latent_dim = {self.latent_dim}\n")
            f.write(f"Heads = {self.heads}\n")
            f.write(f"Epochs = {self.epochs}\n")
            f.write(f"Loss_fn = {self.loss_fn}\n")
            f.write(f"Activation_fn = {self.activation_fn}\n")
            f.write(f"Dropout = {self.d_p}\n")
            f.write(f"Learning_rate = {self.lr}\n")
            f.write(f"Weight_decay = {self.weight_decay}\n")
