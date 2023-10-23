import torch
from torch import nn
from networks.base_models import BaseModel


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

        return latent_data, [decoded_omics_1, decoded_omics_2, decoded_omics_3]

    def forward_pass(self, x_omics):
        omics_1 = x_omics[0].to(self.device)
        omics_2 = x_omics[1].to(self.device)
        omics_3 = x_omics[2].to(self.device)

        latent_data, decoded_omics = self.forward(omics_1, omics_2, omics_3)

        loss = (
            self.a * self.loss_fn[0](decoded_omics[0], omics_1)
            + self.b * self.loss_fn[1](decoded_omics[1], omics_2)
            + self.c * self.loss_fn[2](decoded_omics[2], omics_3)
        )

        return latent_data, loss

    def get_latent_space(self, dataset):
        x_omics = dataset.omics_data

        with torch.no_grad():
            omics_1 = x_omics[0].to(self.device)
            omics_2 = x_omics[1].to(self.device)
            omics_3 = x_omics[2].to(self.device)

            latent_data, decoded_omics = self.forward(omics_1, omics_2, omics_3)

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
        activation_fn=nn.Sigmoid(),
        dropout_p=0,
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
            [decoded_omics_1, decoded_omics_2, decoded_omics_3],
        )

    def forward_pass(self, x_omics):
        omics_1 = x_omics[0].to(self.device)
        omics_2 = x_omics[1].to(self.device)
        omics_3 = x_omics[2].to(self.device)

        latent, decoded_omics = self.forward(omics_1, omics_2, omics_3)

        latent_data, latent_mean, latent_log_var, z = latent

        # Loss
        reconstruction_loss = (
            self.loss_fn[0](decoded_omics[0], omics_1)
            + self.loss_fn[1](decoded_omics[1], omics_2)
            + self.loss_fn[2](decoded_omics[2], omics_3)
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

            latent, decoded_omics = self.forward(omics_1, omics_2, omics_3)
            return latent[-1].detach().cpu().numpy()
