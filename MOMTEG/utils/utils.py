import random

import numpy as np
import torch


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
