import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.logging import log
from utils.train_val_test import Early_Stopping
from abc import ABC, abstractmethod

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
_REGULARISATION = Literal["mmd", "kld"]


class Base_VAE(ABC, nn.Module):
    def __init__(
        self,
        beta,
        loss_fn,
        regularisation: _REGULARISATION = "mmd",
        early_stopping=None,
    ):
        super().__init__()

        self.beta = beta
        self.loss_fn = loss_fn
        self.early_stopping = early_stopping
        self.regularisation = regularisation

    def _init_weights(self, module):
        """
        Xavier uniform model initialisation
        @source https://stackoverflow.com/questions/49433936/how-do-i-initialize-weights-in-pytorch
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    def reparameterize(self, mean, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mean)

        return mean

    def compute_kernel(self, x, y):
        """
        source @https://github.com/napsternxg/pytorch-practice/blob/master/Pytorch%20-%20MMD%20VAE.ipynb
        """
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        x = x.unsqueeze(1)  # (x_size, 1, dim)
        y = y.unsqueeze(0)  # (1, y_size, dim)
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
        return torch.exp(-kernel_input)  # (x_size, y_size)

    def compute_mmd(self, x, y):
        """
        source @https://github.com/napsternxg/pytorch-practice/blob/master/Pytorch%20-%20MMD%20VAE.ipynb
        """
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
        return mmd

    def compute_kld(self, latent_mean, latent_log_var):
        return -0.5 * torch.mean(
            1 + latent_log_var - latent_mean.pow(2) - latent_log_var.exp()
        )

    def compute_loss(self, x, reconstructed, latent_mean, latent_log_var):
        reconstruction_loss = self.loss_fn(reconstructed, x)

        match self.regularisation:
            case "mmd":
                regularisation_loss = self.compute_mmd(reconstructed, x)
            case "kld":
                regularisation_loss = self.compute_kld(latent_mean, latent_log_var)
            case _:
                regularisation_loss = 0

        return reconstruction_loss + self.beta * regularisation_loss

    @abstractmethod
    def forward(self, x):
        pass

    def forward_pass(self, x):
        x = x.to(DEVICE)

        reconstructed, latent_mean, latent_log_var, z = self.forward(x)
        loss = self.compute_loss(x, reconstructed, latent_mean, latent_log_var)

        return loss, z

    def train_loop(self, dataloader, optimizer, epochs):
        for epoch in range(0, epochs):
            self.train()
            loss_sum = 0.0

            for batch_idx, x in enumerate(dataloader):
                loss, _ = self.forward_pass(x)
                loss_sum += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss = loss_sum / len(dataloader)

            val_loss = self.validate(dataloader)

            if epoch % 20 == 0:
                log(
                    Epoch=epoch,
                    Train=train_loss,
                    Val=val_loss,
                )

            if self.early_stopping.check(val_loss):
                print(
                    f"Early stopped at epoch: {epoch}, Best Val: {self.early_stopping.best_loss:.4f}"
                )

                # torch.save(model.state_dict(), "rnaVAE.pth")
                break

    @torch.no_grad()
    def validate(self, dataloader):
        self.eval()
        loss_sum = 0.0

        for batch_idx, x in enumerate(dataloader):
            loss, _ = self.forward_pass(x)
            loss_sum += loss.item()

        return loss_sum / len(dataloader)

    @torch.no_grad()
    def get_latent_space(self, dataloader):
        latent_space = None

        with torch.no_grad():
            self.eval()
            for batch_idx, x in enumerate(dataloader):
                _, z = self.forward_pass(x)
                if latent_space is not None:
                    latent_space = torch.cat((latent_space, z), dim=0)
                else:
                    latent_space = z

        return latent_space.cpu().numpy()


class VAE(Base_VAE):
    def __init__(
        self,
        layers_dim,
        loss_fn,
        params,
        tolerance=10,
    ):
        super().__init__(
            params["beta"],
            loss_fn,
            params["regularisation"],
            Early_Stopping(tolerance),
        )

        input_dim, dense_dim, latent_dim = layers_dim

        self.encoder_dense = nn.Sequential(
            # nn.BatchNorm1d(input_dim),
            # nn.Dropout(p=params["dropout"]),
            nn.Linear(input_dim, dense_dim),
            nn.BatchNorm1d(dense_dim),
            params["activation_fn"],
        )

        self.encoder_mean = nn.Linear(dense_dim, latent_dim)
        self.encoder_log_var = nn.Linear(dense_dim, latent_dim)

        self.decoder_dense = nn.Sequential(
            # nn.BatchNorm1d(latent_dim),
            nn.Linear(latent_dim, dense_dim),
            nn.BatchNorm1d(dense_dim),
            params["activation_fn"],
        )
        self.decoder_output = nn.Sequential(
            # nn.BatchNorm1d(dense_dim),
            nn.Linear(dense_dim, input_dim),
            # nn.Sigmoid(),
        )

        self.apply(self._init_weights)

    def forward(self, x):
        x = self.encoder_dense(x)

        latent_mean = self.encoder_mean(x)
        latent_log_var = self.encoder_log_var(x)

        z = self.reparameterize(latent_mean, latent_log_var)

        reconstructed = self.decoder_dense(z)
        reconstructed = self.decoder_output(reconstructed)

        return reconstructed, latent_mean, latent_log_var, z


class H_VAE(Base_VAE):
    def __init__(
        self,
        rnaVAE,
        cnaVAE,
        layers_dim,
        loss_fn,
        params,
        tolerance=10,
    ):
        super().__init__(
            params["beta"],
            loss_fn,
            params["regularisation"],
            Early_Stopping(tolerance),
        )

        dense_dim, latent_dim = layers_dim

        self.rnaVAE = rnaVAE
        self.cnaVAE = cnaVAE

        dense_dim = params["dense_dim"]
        latent_dim = params["latent_dim"]

        self.encoder_dense = nn.Sequential(
            # nn.BatchNorm1d(dense_dim),
            # nn.Dropout(p=params["dropout"]),
            nn.Linear(dense_dim, dense_dim),
            nn.BatchNorm1d(dense_dim),
            params["activation_fn"],
        )

        self.encoder_mean = nn.Linear(dense_dim, latent_dim)
        self.encoder_log_var = nn.Linear(dense_dim, latent_dim)

        self.decoder_dense = nn.Sequential(
            # nn.BatchNorm1d(latent_dim),
            nn.Linear(latent_dim, dense_dim),
            nn.BatchNorm1d(dense_dim),
            params["activation_fn"],
        )

        self.decoder_output = nn.Sequential(
            # nn.BatchNorm1d(dense_dim),
            nn.Linear(dense_dim, dense_dim),
            # nn.Sigmoid()
        )

        self.apply(self._init_weights)

    def forward(self, rna, cna):
        with torch.no_grad():
            _, _, _, rna = self.rnaVAE(rna)
            _, _, _, cna = self.cnaVAE(cna)

        latent_x = torch.cat((cna, rna), dim=1)
        x = self.encoder_dense(latent_x)

        latent_mean = self.encoder_mean(x)
        latent_log_var = self.encoder_log_var(x)

        z = self.reparameterize(latent_mean, latent_log_var)

        reconstructed = self.decoder_dense(z)
        reconstructed = self.decoder_output(reconstructed)

        return latent_x, reconstructed, latent_mean, latent_log_var, z

    def forward_pass(self, x):
        # print('MEM {:.3f}MB'.format(torch.cuda.memory_allocated()/1024**2))

        rna, cna = x

        rna = rna.to(DEVICE)
        cna = cna.to(DEVICE)

        latent_x, reconstructed, latent_mean, latent_log_var, z = self.forward(rna, cna)
        loss = self.compute_loss(latent_x, reconstructed, latent_mean, latent_log_var)

        return loss, z
