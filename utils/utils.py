import json
import random

import numpy as np
import torch
import torch.nn as nn


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
