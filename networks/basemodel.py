import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.logging import log

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cross_entropy_loss(y_pred, y_true, weight=None, reduction="mean"):
    criterion = nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    return criterion(y_pred, y_true)


class EarlyStopper:
    def __init__(self, min_epochs, patience=5, minimise=True):
        self.min_epochs = min_epochs
        self.patience = patience
        self.best_value = 0
        self.best_epoch = 0
        self.best_metrics = None
        self.best_model = None

        if minimise:
            self.best_value = float("inf")
        self.minimise = minimise

    def check(self, value, epoch, metrics, model):
        if self.min_epochs >= epoch:
            return False

        if self.minimise:
            if value < self.best_value:
                self.best_value = value
                self.best_epoch = epoch
                self.best_metrics = metrics
                self.best_model = model.state_dict()
        else:
            if value > self.best_value:
                self.best_value = value
                self.best_epoch = epoch
                self.best_metrics = metrics
                self.best_model = model.state_dict()

        if (epoch - self.best_epoch) >= self.patience:
            return True

        return False


class ClsBaseModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    def calculate_loss(self, cls_pred, cls_y, mask=None, weight=None, reduction="mean"):
        if mask is not None:
            cls_pred = cls_pred[mask]
            cls_y = cls_y[mask]

        cls_loss = cross_entropy_loss(
            cls_pred, cls_y, reduction=reduction, weight=weight
        )

        return cls_loss

    def update_step(self, data, optimizer):
        if self.config["net_type"] == "MLP":
            train_data = TensorDataset(data.x[data.train_mask], data.y[data.train_mask])

            train_loader = DataLoader(
                train_data,
                batch_size=self.config["batch_size"],
                shuffle=False,
            )
            loss = 0

            for batch_idx, batch_data in enumerate(train_loader):

                x, cls_y = batch_data

                x = x.to(DEVICE)
                pred, _ = self.forward(x)

                batch_loss = self.calculate_loss(
                    pred, cls_y, weight=data.class_weights, reduction="mean"
                )

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                loss += batch_loss.item()

        else:
            pred, _ = self.forward(data.x, data.edge_index, data.edge_attr)

            loss = self.calculate_loss(
                pred,
                data.y,
                data.train_mask,
                data.class_weights,
                reduction="mean",
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.item()

        return loss

    def validation_step(self, data):
        if len(data.val_mask) > 0:
            pred, _ = self.forward(data.x, data.edge_index, data.edge_attr)
            val_loss = self.calculate_loss(
                pred,
                data.y,
                mask=data.val_mask,
                weight=data.class_weights,
                reduction="mean",
            )
            val_loss = val_loss.item()

            val_acc, val_f1 = self.evaluate_cls(data, data.val_mask)

            return val_loss, val_acc, val_f1

        return None, None, None

    def train_loop(self, data, verbose=False):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config["wd"],
        )

        earlyStop = EarlyStopper(
            min_epochs=self.config["min_epochs"], patience=self.config["patience"]
        )

        best_model = None

        for epoch in range(1, self.config["epochs"] + 1):
            self.train()

            loss = self.update_step(data, optimizer)
            train_acc, train_f1 = self.evaluate_cls(data, data.train_mask)

            val_loss, val_acc, val_f1 = self.validation_step(data)

            if verbose:
                log(
                    Epoch=epoch,
                    Loss=loss,
                    Val_Loss=val_loss,
                    Train_Acc=train_acc,
                    Val_Acc=val_acc,
                    Train_f1=train_f1,
                    Val_f1=val_f1,
                )

            if (len(data.val_mask) > 0) and earlyStop.check(
                val_loss, epoch, [val_acc, val_f1], self
            ):
                if verbose:
                    print(
                        f"Early stopped! Best metrics {earlyStop.best_metrics} at epoch {earlyStop.best_epoch}"
                    )
                val_acc, val_f1 = earlyStop.best_metrics
                best_model = earlyStop.best_model

                break

        return best_model, val_acc, val_f1, earlyStop.best_epoch

    @torch.no_grad()
    def get_latent_space(self, data, save_path=None):
        self.eval()
        latent, _ = self.forward(data.x, data.edge_index, data.edge_attr)
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
        cls_pred, _ = self.forward(data.x, data.edge_index, data.edge_attr)
        cls_pred = cls_pred.argmax(dim=-1)
        cls_pred = cls_pred.cpu().numpy()

        if mask is not None:
            cls_pred = cls_pred[mask]

        return cls_pred

    @torch.no_grad()
    def evaluate_cls(self, data, mask):
        self.eval()
        cls_pred, _ = self.forward(data.x, data.edge_index, data.edge_attr)
        cls_pred = cls_pred.argmax(dim=-1)

        f1 = f1_score(
            data.y[mask].cpu().numpy(),
            cls_pred[mask].cpu().numpy(),
            average="macro",
        )
        acc = accuracy_score(data.y[mask].cpu().numpy(), cls_pred[mask].cpu().numpy())

        return acc, f1
