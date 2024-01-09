from sklearn.metrics import f1_score
import torch.nn.functional as F
import torch
import numpy as np
import random
from typing import Literal
import copy
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


def setup_seed(seed=SEED):
    """
    setup seed to make the experiments deterministic

    Parameters:
        seed(int) -- the random seed

    @source https://github.com/zhangxiaoyu11/OmiEmbed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    return seed


from utils.data import plot_confusion_matrix, plot_latent_space


def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out, _ = model(data.x, data.edge_index)
    loss = F.cross_entropy(
        out[data.mask_train], data.y[data.mask_train], weight=data.class_weights
    )
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def validate(model, data):
    model.eval()
    pred, _ = model(data.x, data.edge_index)
    pred = pred.argmax(dim=-1)

    f1s = []
    for mask in [data.mask_train, data.mask_val]:
        f1s.append(
            f1_score(
                data.y[mask].cpu().numpy(), pred[mask].cpu().numpy(), average="macro"
            )
        )

    return f1s


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    pred, h = model(data.x, data.edge_index)
    pred = pred.argmax(dim=-1)

    f1 = f1_score(
        data.y[data.mask_test].cpu().numpy(),
        pred[data.mask_test].cpu().numpy(),
        average="macro",
    )
    plot_latent_space(h.cpu().numpy(), data.pam50)
    plot_confusion_matrix(
        data.y[data.mask_test].cpu().numpy(),
        pred[data.mask_test].cpu().numpy(),
        data.pam50_labels,
        normalize=None,
    )
    plot_confusion_matrix(
        data.y[data.mask_test].cpu().numpy(),
        pred[data.mask_test].cpu().numpy(),
        data.pam50_labels,
        normalize="true",
    )
    return f1


_MODES = Literal["minimize", "maximize"]


class Early_Stopping:
    def __init__(self, tolerance=5, mode: _MODES = "minimize"):
        self.mode = mode

        match self.mode:
            case "minimize":
                self.best_loss = float("inf")
            case "maximize":
                self.best_loss = -float("inf")

        self.best_model = None
        self.counter = 0
        self.tolerance = tolerance

    def check(self, new_loss):
        self.counter += 1

        match self.mode:
            case "minimize":
                if new_loss < self.best_loss:
                    self.best_loss = new_loss
                    self.counter = 0

                    return False

            case "maximize":
                if new_loss > self.best_loss:
                    self.best_loss = new_loss
                    self.counter = 0

                    return False

        if self.counter == self.tolerance:
            return True

        return False
