import json
import random

import numpy as np
import torch
import torch.nn as nn

ACT_FN = {
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "elu": nn.ELU(),
    "leakyrelu": nn.LeakyReLU(),
}


def setup_seed(seed=42):
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


def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def save_config(config, save_path):
    with open(save_path, "w") as f:
        json.dump(config, f)


class EarlyStopper:
    def __init__(self, min_epochs, patience=5, minimise=True):
        self.min_epochs = min_epochs
        self.patience = patience
        self.best_value = 0
        self.best_epoch = 0
        self.best_metrics = None
        if minimise:
            self.best_value = float("inf")
        self.minimise = minimise

    def check(self, value, epoch, metrics):
        if self.min_epochs >= epoch:
            return False

        if self.minimise:
            if value < self.best_value:
                self.best_value = value
                self.best_epoch = epoch
                self.best_metrics = metrics
        else:
            if value > self.best_value:
                self.best_value = value
                self.best_epoch = epoch
                self.best_metrics = metrics

        if (epoch - self.best_epoch) >= self.patience:
            return True

        return False
