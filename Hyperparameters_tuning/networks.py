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


class VAE(BaseModel):
    def __init__(
        self,
        input_dims,
        name,
        activation_fn=nn.Sigmoid(),
        dropout_p=0,
        hidden_dim=100,
        latent_dim=100,
        loss_fn=nn.MSELoss(reduction="mean"),
        beta=0.000020,
    ):
        super().__init__(name)

        self.activation_fn = activation_fn
        self.loss_fn = loss_fn
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.beta = beta

        # encoders, multi channel input
        self.encoder_omics = nn.Sequential(
            nn.Linear(self.input_dims, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Dropout(p=dropout_p),
            self.activation_fn,
        )

        self.encoder_mean = nn.Linear(self.hidden_dim, self.latent_dim)
        self.encoder_var = nn.Linear(self.hidden_dim, self.latent_dim)

        # decoders
        self.decoder_z = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Dropout(p=dropout_p),
            self.activation_fn,
        )

        self.decoder_omics = nn.Sequential(
            nn.Linear(self.hidden_dim, self.input_dims),
            nn.BatchNorm1d(self.input_dims),
            nn.Dropout(p=dropout_p),
            self.activation_fn,
        )

        # Variable initialization
        for name, param in VAE.named_parameters(self):
            if "weight" in name:
                torch.nn.init.normal_(param, mean=0, std=0.1)
            if "bias" in name:
                torch.nn.init.constant_(param, val=0)

    def reparameterize(self, mean, var):
        if self.training:
            std = torch.exp(0.5 * var)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mean)

        return mean

    def forward(self, omics):
        """
        :param omics_1: omics data 1
        :param omics_2: omics data 2
        :param omics_3: omics data 3
        """
        # omics_1 to z_1
        encoded_omics = self.encoder_omics(omics)

        latent_mean = self.encoder_mean(encoded_omics)
        latent_var = self.encoder_var(encoded_omics)

        z = self.reparameterize(latent_mean, latent_var)

        # z_1 to omics_1
        decoded_z = self.decoder_z(z)
        decoded_omics = self.decoder_omics(decoded_z)

        return (
            [
                z,
                latent_mean,
                latent_var,
            ],
            decoded_omics,
        )

    def forward_pass(self, x_omics):
        omics = x_omics.to(self.device)

        latent, decoded_omics = self.forward(omics)
        z, mean, var = latent

        reconstruction_loss = self.loss_fn(decoded_omics, omics)
        KLD = -0.5 * torch.mean(1 + var - mean.pow(2) - var.exp())
        loss = reconstruction_loss + self.beta * KLD

        return decoded_omics, loss

    def get_latent_space(self, x_omics):
        with torch.no_grad():
            omics = x_omics.to(self.device)

            latent, decoded_omics = self.forward(omics)
            return latent[0].detach().cpu().numpy()
