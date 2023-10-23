import torch
from torch import nn
from abc import ABC, abstractmethod


class BaseModel(ABC, nn.Module):
    def __init__(self, name):
        super(BaseModel, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = name

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def forward_pass(self, x_omics, loss_fn):
        pass

    @abstractmethod
    def get_latent_space(self, latent_data):
        pass
