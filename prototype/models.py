import torch
import torch.nn.functional as F
from torch import nn
from matplotlib import pyplot as plt
import functools
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


class MoGCN_AE(BaseModel):
    """
    Implementation of the paper MoGCN: A Multi-Omics Integration Method Based on
    Graph Convolutional Network for Cancer Subtype Analysis

    @doi: 10.3389/fgene.2022.806842
    """

    def __init__(
        self,
        input_dims,
        activation_fn=nn.Sigmoid(),
        dropout_p=0,
        latent_dim=100,
        loss_fn=[nn.MSELoss(reduction="mean")] * 3,
        name="MoGCN_AE",
    ):
        super().__init__(name)

        self.input_dims = input_dims
        self.latent_dim = latent_dim
        self.loss_fn = loss_fn
        self.a = 0.4
        self.b = 0.3
        self.c = 0.3

        # encoders, multi channel input
        self.encoder_omics_1 = nn.Sequential(
            nn.Linear(self.input_dims[0], self.latent_dim),
            nn.BatchNorm1d(self.latent_dim),
            nn.Dropout(p=dropout_p),
            activation_fn,
        )
        self.encoder_omics_2 = nn.Sequential(
            nn.Linear(self.input_dims[1], self.latent_dim),
            nn.BatchNorm1d(self.latent_dim),
            nn.Dropout(p=dropout_p),
            activation_fn,
        )
        self.encoder_omics_3 = nn.Sequential(
            nn.Linear(self.input_dims[2], self.latent_dim),
            nn.BatchNorm1d(self.latent_dim),
            nn.Dropout(p=dropout_p),
            activation_fn,
        )

        # decoders
        self.decoder_omics_1 = nn.Sequential(
            nn.Linear(self.latent_dim, self.input_dims[0])
        )
        self.decoder_omics_2 = nn.Sequential(
            nn.Linear(self.latent_dim, self.input_dims[1])
        )
        self.decoder_omics_3 = nn.Sequential(
            nn.Linear(self.latent_dim, self.input_dims[2])
        )

        # Variable initialization
        for name, param in MoGCN_AE.named_parameters(self):
            if "weight" in name:
                torch.nn.init.normal_(param, mean=0, std=0.1)
            if "bias" in name:
                torch.nn.init.constant_(param, val=0)

    def forward(self, omics_1, omics_2, omics_3):
        """
        :param omics_1: omics data 1
        :param omics_2: omics data 2
        :param omics_3: omics data 3
        """
        encoded_omics_1 = self.encoder_omics_1(omics_1)
        encoded_omics_2 = self.encoder_omics_2(omics_2)
        encoded_omics_3 = self.encoder_omics_3(omics_3)
        latent_data = (
            torch.mul(encoded_omics_1, self.a)
            + torch.mul(encoded_omics_2, self.b)
            + torch.mul(encoded_omics_3, self.c)
        )
        decoded_omics_1 = self.decoder_omics_1(latent_data)
        decoded_omics_2 = self.decoder_omics_2(latent_data)
        decoded_omics_3 = self.decoder_omics_3(latent_data)

        return latent_data, decoded_omics_1, decoded_omics_2, decoded_omics_3

    def forward_pass(self, x_omics):
        omics_1 = x_omics[0].to(self.device)
        omics_2 = x_omics[1].to(self.device)
        omics_3 = x_omics[2].to(self.device)

        latent_data, decoded_omics_1, decoded_omics_2, decoded_omics_3 = self.forward(
            omics_1, omics_2, omics_3
        )

        loss = (
            self.a * self.loss_fn[0](decoded_omics_1, omics_1)
            + self.b * self.loss_fn[1](decoded_omics_2, omics_2)
            + self.c * self.loss_fn[2](decoded_omics_3, omics_3)
        )

        return latent_data, loss

    # def get_latent_space(self, latent_data):
    #     return latent_data.detach().cpu().numpy()

    def get_latent_space(self, dataset):
        x_omics = dataset.omics_data

        with torch.no_grad():
            omics_1 = x_omics[0].to(self.device)
            omics_2 = x_omics[1].to(self.device)
            omics_3 = x_omics[2].to(self.device)

            (
                latent_data,
                decoded_omics_1,
                decoded_omics_2,
                decoded_omics_3,
            ) = self.forward(omics_1, omics_2, omics_3)

            return latent_data.detach().cpu().numpy()


class MoGCN_VAE(BaseModel):
    """
    Implementation of the paper MoGCN: A Multi-Omics Integration Method Based on
    Graph Convolutional Network for Cancer Subtype Analysis

    @doi: 10.3389/fgene.2022.806842
    """

    def __init__(
        self,
        input_dims,
        activation_fn,
        dropout_p,
        latent_dim=100,
        loss_fn=[nn.MSELoss(reduction="mean")] * 3,
        beta=0.000020,
        name="MoGCN_VAE",
    ):
        super().__init__(name)

        self.activation_fn = activation_fn
        self.loss_fn = loss_fn
        self.input_dims = input_dims
        self.latent_dim = latent_dim
        self.beta = beta

        # encoders, multi channel input
        self.encoder_omics_1 = nn.Sequential(
            nn.Linear(self.input_dims[0], self.latent_dim),
            nn.BatchNorm1d(self.latent_dim),
            nn.Dropout(p=dropout_p),
            self.activation_fn,
        )
        self.encoder_omics_2 = nn.Sequential(
            nn.Linear(self.input_dims[1], self.latent_dim),
            nn.BatchNorm1d(self.latent_dim),
            nn.Dropout(p=dropout_p),
            self.activation_fn,
        )
        self.encoder_omics_3 = nn.Sequential(
            nn.Linear(self.input_dims[2], self.latent_dim),
            nn.BatchNorm1d(self.latent_dim),
            nn.Dropout(p=dropout_p),
            self.activation_fn,
        )

        self.encoder_mean = nn.Linear(self.latent_dim * 3, self.latent_dim)
        self.encoder_log_var = nn.Linear(self.latent_dim * 3, self.latent_dim)

        # decoders
        self.decoder_omics_1 = nn.Sequential(
            nn.Linear(self.latent_dim, self.input_dims[0])
        )
        self.decoder_omics_2 = nn.Sequential(
            nn.Linear(self.latent_dim, self.input_dims[1])
        )
        self.decoder_omics_3 = nn.Sequential(
            nn.Linear(self.latent_dim, self.input_dims[2])
        )

        # Variable initialization
        for name, param in MoGCN_AE.named_parameters(self):
            if "weight" in name:
                torch.nn.init.normal_(param, mean=0, std=0.1)
            if "bias" in name:
                torch.nn.init.constant_(param, val=0)

    def reparameterize(self, mean, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mean)

        return mean

    def forward(self, omics_1, omics_2, omics_3):
        """
        :param omics_1: omics data 1
        :param omics_2: omics data 2
        :param omics_3: omics data 3
        """
        # Encoder
        encoded_omics_1 = self.encoder_omics_1(omics_1)
        encoded_omics_2 = self.encoder_omics_2(omics_2)
        encoded_omics_3 = self.encoder_omics_3(omics_3)

        # Latent
        latent_data = torch.cat((encoded_omics_1, encoded_omics_2, encoded_omics_3), 1)

        latent_mean = self.encoder_mean(latent_data)
        latent_log_var = self.encoder_log_var(latent_data)

        z = self.reparameterize(latent_mean, latent_log_var)

        # Decoder
        decoded_omics_1 = self.decoder_omics_1(z)
        decoded_omics_2 = self.decoder_omics_2(z)
        decoded_omics_3 = self.decoder_omics_3(z)

        return (
            [latent_data, latent_mean, latent_log_var, z],
            decoded_omics_1,
            decoded_omics_2,
            decoded_omics_3,
        )

    def forward_pass(self, x_omics):
        omics_1 = x_omics[0].to(self.device)
        omics_2 = x_omics[1].to(self.device)
        omics_3 = x_omics[2].to(self.device)

        latent, decoded_omics_1, decoded_omics_2, decoded_omics_3 = self.forward(
            omics_1, omics_2, omics_3
        )

        latent_data, latent_mean, latent_log_var, z = latent

        # Loss
        reconstruction_loss = (
            self.loss_fn[0](decoded_omics_1, omics_1)
            + self.loss_fn[1](decoded_omics_2, omics_2)
            + self.loss_fn[2](decoded_omics_3, omics_3)
        )
        KLD = -0.5 * torch.mean(
            1 + latent_log_var - latent_mean.pow(2) - latent_log_var.exp()
        )

        loss = reconstruction_loss + self.beta * KLD

        return z, loss

    def get_latent_space(self, dataset):
        x_omics = dataset.omics_data

        with torch.no_grad():
            omics_1 = x_omics[0].to(self.device)
            omics_2 = x_omics[1].to(self.device)
            omics_3 = x_omics[2].to(self.device)

            (
                latent_data,
                decoded_omics_1,
                decoded_omics_2,
                decoded_omics_3,
            ) = self.forward(omics_1, omics_2, omics_3)

            return latent_data[-1].detach().cpu().numpy()


import math
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
    def __init__(self, infeas, outfeas, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = infeas
        self.out_features = outfeas
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))

        if bias:
            self.bias = Parameter(torch.FloatTensor(self.out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        """
        for name, param in GraphConvolution.named_parameters(self):
            if 'weight' in name:
                #torch.nn.init.constant_(param, val=0.1)
                torch.nn.init.normal_(param, mean=0, std=0.1)
            if 'bias' in name:
                torch.nn.init.constant_(param, val=0)
        """

    def forward(self, x, adj):
        x1 = torch.mm(x, self.weight)
        output = torch.mm(adj, x1)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN(BaseModel):
    def __init__(self, n_in, n_hid=64, n_out=4, dropout=0, name="MoGCN_GCN"):
        super().__init__(name)

        self.gc1 = GraphConvolution(n_in, n_hid)
        self.gc2 = GraphConvolution(n_hid, n_hid)
        self.dp1 = nn.Dropout(dropout)
        self.dp2 = nn.Dropout(dropout)
        self.fc = nn.Linear(n_hid, n_out)
        self.dropout = dropout

    def forward(self, input, adj):
        x = self.gc1(input, adj)
        x = F.elu(x)
        x = self.dp1(x)
        x = self.gc2(x, adj)
        x = F.elu(x)
        x = self.dp2(x)

        prediction = self.fc(x)

        return x, prediction

    def forward_pass(self, latent, adj_matrix, gt_classes):
        latent = latent.to(self.device)
        adj_matrix = adj_matrix.to(self.device)
        gt_classes = gt_classes.to(self.device)

        latent, prediction = self.forward(latent, adj_matrix)

        loss = F.cross_entropy(prediction, gt_classes)

        return prediction, loss

    def get_latent_space(self, dataset):
        latent = dataset.latent_space.to(self.device)
        adj_matrix = dataset.adj_matrix.to(self.device)

        latent, prediction = self.forward(latent, adj_matrix)

        return latent.detach().cpu().numpy()
