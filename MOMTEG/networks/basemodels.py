import os

import numpy as np
import pandas as pd
import torch
from networks.losses import MTLR_loss, cross_entropy_loss
from sklearn.metrics import accuracy_score, f1_score
from sksurv.metrics import concordance_index_censored
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.logging import log
from utils.data import get_tri_matrix
from utils.utils import EarlyStopper

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        if self.config["cls_loss_weight"] != 1:
            self.tri_matrix_1 = get_tri_matrix(
                config["n_buckets"], dimension_type=1
            ).to(DEVICE)
            self.tri_matrix_2 = get_tri_matrix(
                config["n_buckets"], dimension_type=2
            ).to(DEVICE)

    def calculate_loss(self, pred, cls_y, surv_y, E, mask=None, reduction="mean"):
        cls_pred, surv_pred = pred
        cls_loss = 0
        if self.config["cls_loss_weight"] > 0:
            if mask is not None:
                cls_pred = cls_pred[mask]
                cls_y = cls_y[mask]

            cls_loss = cross_entropy_loss(cls_pred, cls_y, reduction=reduction)

        surv_loss = 0
        if self.config["surv_loss_weight"] > 0:
            if mask is not None:
                surv_pred = surv_pred[mask]
                surv_y = surv_y[mask]
                E = E[mask].to_numpy()

            surv_loss = MTLR_loss(
                surv_pred,
                surv_y,
                E,
                self.tri_matrix_1,
                reduction=reduction,
            )

        loss = (
            self.config["cls_loss_weight"] * cls_loss
            + self.config["surv_loss_weight"] * surv_loss
        )

        return loss

    def update_step(self, data, optimizer):
        if self.config["net_type"] == "MLP":

            if self.config["cls_loss_weight"] == 1:
                train_data = TensorDataset(
                    data.x[data.train_mask], data.cls_y[data.train_mask]
                )
            else:
                train_data = TensorDataset(
                    data.x[data.train_mask],
                    data.cls_y[data.train_mask],
                    data.y[data.train_mask],
                    torch.tensor(data.E[data.train_mask].values, dtype=torch.bool),
                )

            train_loader = DataLoader(
                train_data,
                batch_size=self.config["batch_size"],
                shuffle=False,
            )
            loss = 0

            for batch_idx, batch_data in enumerate(train_loader):

                if self.config["cls_loss_weight"] == 1:
                    x, cls_y = batch_data
                    surv_y, E = None, None
                else:
                    x, cls_y, surv_y, E = batch_data
                    E = E.detach().cpu().numpy()
                x = x.to(DEVICE)
                pred, _ = self.forward(x)

                batch_loss = self.calculate_loss(
                    pred, cls_y, surv_y, E, reduction="mean"
                )

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                loss += batch_loss.item()

        else:
            pred, _ = self.forward(data.x, data.edge_index, data.edge_attr)

            loss = self.calculate_loss(
                pred,
                data.cls_y,
                data.y,
                data.E,
                data.train_mask,
                reduction="mean",
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.item()

        val_loss = loss
        if len(data.val_mask) > 0:
            pred, _ = self.forward(data.x, data.edge_index, data.edge_attr)
            cls_pred, surv_pred = pred
            val_loss = self.calculate_loss(
                pred,
                data.cls_y,
                data.y,
                data.E,
                mask=data.val_mask,
                reduction="mean",
            )
            val_loss = val_loss.item()
        return loss, val_loss

    def train_loop(self, data, verbose=False):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config["wd"],
        )

        # earlyStop = EarlyStopper(
        #     min_epochs=self.config["min_epochs"], patience=self.config["patience"]
        # )

        for epoch in range(1, self.config["epochs"] + 1):
            self.train()

            loss, val_loss = self.update_step(data, optimizer)

            train_acc, train_f1 = None, None
            if self.config["cls_loss_weight"] != 0:
                train_acc, train_f1 = self.evaluate_cls(data, data.train_mask)
            train_c = None
            if self.config["cls_loss_weight"] != 1:
                train_c = self.evaluate_surv(data, data.train_mask)

            val_acc, val_f1, val_c = None, None, None
            if len(data.val_mask) > 0:
                if self.config["cls_loss_weight"] != 0:
                    val_acc, val_f1 = self.evaluate_cls(data, data.val_mask)
                if self.config["cls_loss_weight"] != 1:
                    val_c = self.evaluate_surv(data, data.val_mask)

            if verbose:
                log(
                    Epoch=epoch,
                    Loss=loss,
                    Val_Loss=val_loss,
                    Train_Acc=train_acc,
                    Val_Acc=val_acc,
                    Train_f1=train_f1,
                    Val_f1=val_f1,
                    Train_C=train_c,
                    Val_C=val_c,
                )

            # if (len(data.val_mask) > 0) and earlyStop.check(
            #     val_loss, epoch, [val_acc, val_f1, val_c]
            # ):
            #     if verbose:
            #         print(
            #             f"Early stopped! Best metrics {earlyStop.best_metrics} at epoch {earlyStop.best_epoch}"
            #         )
            #     val_acc, val_f1, val_c = earlyStop.best_metrics
            #     break

        # return val_acc, val_f1, val_c, earlyStop.best_epoch
        return val_acc, val_f1, val_c

    @torch.no_grad()
    def get_latent_space(self, data, save_path=None):
        self.eval()
        _, latent = self.forward(data.x, data.edge_index, data.edge_attr)
        latent = latent.cpu().numpy()

        if save_path:
            np.savetxt(
                os.path.join(save_path),
                latent,
                delimiter=",",
            )

        return latent

    @torch.no_grad()
    def predict_cls(self, data, mask=None):
        self.eval()
        pred, _ = self.forward(data.x, data.edge_index, data.edge_attr)
        cls_pred, _ = pred
        cls_pred = cls_pred.argmax(dim=-1)
        cls_pred = cls_pred.cpu().numpy()

        if mask is not None:
            cls_pred = cls_pred[mask]

        return cls_pred

    @torch.no_grad()
    def predict_risk(self, data, mask):
        """
        Predict the density, survival and hazard function, as well as the risk score
        """
        self.eval()

        pred, _ = self.forward(data.x, data.edge_index, data.edge_attr)
        _, surv_pred = pred

        phi = torch.exp(torch.mm(surv_pred[mask], self.tri_matrix_1))
        div = torch.repeat_interleave(
            torch.sum(phi, 1).reshape(-1, 1), phi.shape[1], dim=1
        )

        density = phi / div
        survival = torch.mm(density, self.tri_matrix_2)
        hazard = density[:, :-1] / survival[:, 1:]

        cumulative_hazard = torch.cumsum(hazard, dim=1)
        risk = torch.sum(cumulative_hazard, 1)

        return {
            "density": density,
            "survival": survival,
            "hazard": hazard,
            "risk": risk,
        }

    @torch.no_grad()
    def evaluate_cls(self, data, mask):
        self.eval()
        pred, _ = self.forward(data.x, data.edge_index, data.edge_attr)
        cls_pred, _ = pred
        cls_pred = cls_pred.argmax(dim=-1)

        f1 = f1_score(
            data.cls_y[mask].cpu().numpy(),
            cls_pred[mask].cpu().numpy(),
            average="macro",
        )
        acc = accuracy_score(
            data.cls_y[mask].cpu().numpy(), cls_pred[mask].cpu().numpy()
        )

        return acc, f1

    @torch.no_grad()
    def evaluate_surv(self, data, mask):
        self.eval()

        out_pred = self.predict_risk(data, mask)

        c_index = concordance_index_censored(
            data.E[mask], data.T[mask], out_pred["risk"].detach().cpu().numpy()
        )[0]
        return c_index
