import torch
import torch.nn as nn
from typing import Literal
from torch_geometric.logging import log
from losses import compute_vae_loss
import numpy as np
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
_REGULARISATION = Literal["mmd", "kld", "_"]
_MODES = Literal["CNA", "RNA", "CLI"]


class VAE(nn.Module):
    def __init__(self, params, omics_index=None):
        super().__init__()

        self.beta = params.beta
        self.regularisation = params.regularisation
        self.loss_fn = params.loss_fn
        self.latent_dim = params.latent_dim
        self.batch_size = params.batch_size
        self.omics_index = omics_index

        self.encoder_dense = nn.Sequential(
            nn.Linear(params.input_dim, params.dense_dim),
            nn.BatchNorm1d(params.dense_dim),
            # nn.Dropout(params.d_p),
            nn.ELU(),
        )

        self.encoder_mean = nn.Linear(params.dense_dim, params.latent_dim)
        self.encoder_log_var = nn.Linear(params.dense_dim, params.latent_dim)

        self.decoder_dense = nn.Sequential(
            nn.Linear(params.latent_dim, params.dense_dim),
            nn.BatchNorm1d(params.dense_dim),
            nn.Dropout(params.d_p),
            nn.ELU(),
        )

        self.decoder_output = nn.Linear(params.dense_dim, params.input_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def handle_input(self, x):
        if self.omics_index is not None:
            return x[self.omics_index]
        return x

    def forward(self, x):
        original_x = self.handle_input(x).to(DEVICE)

        x = self.encoder_dense(original_x)

        latent_mean = self.encoder_mean(x)
        latent_log_var = self.encoder_log_var(x)

        z = self.reparameterize(latent_mean, latent_log_var)

        reconstructed_x = self.decoder_dense(z)
        reconstructed_x = self.decoder_output(reconstructed_x)

        return original_x, reconstructed_x, latent_mean, latent_log_var, z

    def train_loop(self, dataloader, val_dataloader, optimizer, epochs):
        for epoch in range(0, epochs):
            self.train()
            loss_sum = 0.0

            for batch_idx, x in enumerate(dataloader):
                (
                    original_x,
                    reconstructed_x,
                    latent_mean,
                    latent_log_var,
                    z,
                ) = self.forward(x)

                loss = compute_vae_loss(
                    self.loss_fn,
                    self.regularisation,
                    self.beta,
                    original_x,
                    reconstructed_x,
                    latent_mean,
                    latent_log_var,
                    z,
                    self.latent_dim,
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_sum += loss.item()

            train_loss = loss_sum / len(dataloader)

            val_loss = self.validate(val_dataloader)

            if epoch % 20 == 0:
                log(
                    Epoch=epoch,
                    Train=train_loss,
                    Val=val_loss,
                )

    def validate(self, val_dataloader):
        self.eval()
        loss_sum = 0.0

        for batch_idx, x in enumerate(val_dataloader):
            original_x, reconstructed_x, latent_mean, latent_log_var, z = self.forward(
                x
            )

            loss = compute_vae_loss(
                self.loss_fn,
                self.regularisation,
                self.beta,
                original_x,
                reconstructed_x,
                latent_mean,
                latent_log_var,
                z,
                self.latent_dim,
            )

            loss_sum += loss.item()

        return loss_sum / len(val_dataloader)

    def get_latent_space(self, dataloader, save_path=None):
        self.eval()
        latent_space = None

        with torch.no_grad():
            for batch_idx, x in enumerate(dataloader):
                return_values = self.forward(x)
                z = return_values[-1]
                if latent_space is not None:
                    latent_space = torch.cat((latent_space, z), dim=0)
                else:
                    latent_space = z

        latent_space = latent_space.cpu().numpy()

        if save_path:
            np.savetxt(
                os.path.join(save_path),
                latent_space,
                delimiter=",",
            )

        return latent_space


class CNC_VAE(VAE):
    def __init__(self, params):
        super().__init__(params)

    def handle_input(self, x):
        return torch.cat(x, dim=1)


class H_VAE(VAE):
    def __init__(self, params, omics_names, omics_dims):
        super().__init__(params)

        input_VAEs = []

        for i in range(len(omics_names)):
            temp_params = Params_VAE(
                omics_dims[i],
                params.dense_dim,
                params.dense_dim // 2,
                epochs=params.epochs,
                beta=params.beta,
                regularisation=params.regularisation,
                weight_decay=params.weight_decay,
            )
            if omics_names[i] == "CNA" or omics_names[i] == "CLI":
                temp_params.loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
            if omics_names[i] == "RNA":
                temp_params.loss_fn = nn.MSELoss(reduction="mean")

            input_VAEs.append(VAE(temp_params, omics_index=i))

        self.input_VAEs = input_VAEs
        self.params = params

    def forward(self, x):
        latent_omics = []

        with torch.no_grad():
            for i in range(len(x)):
                self.input_VAEs[i].eval()
                latent_data = self.input_VAEs[i].forward(x)
                z = latent_data[-1]
                latent_omics.append(z)

        latent_x = torch.cat(latent_omics, dim=1)

        x, reconstructed, latent_mean, latent_log_var, z = super().forward(latent_x)

        return latent_x, reconstructed, latent_mean, latent_log_var, z

    def train_loop(self, train_dataloader, test_dataloader, optimizer, epochs):
        for i in range(len(self.input_VAEs)):
            print(f"Training VAE {i+1}")

            input_optimizer = torch.optim.Adam(
                self.input_VAEs[i].parameters(),
                lr=self.params.lr,
                weight_decay=self.params.weight_decay,
            )

            self.input_VAEs[i] = self.input_VAEs[i].to(DEVICE)
            self.input_VAEs[i].train_loop(
                train_dataloader, test_dataloader, input_optimizer, epochs
            )

        print("Training H_VAE")
        super().train_loop(train_dataloader, test_dataloader, optimizer, epochs)


class Params_VAE:
    def __init__(
        self,
        input_dim,
        dense_dim,
        latent_dim,
        lr=0.001,
        weight_decay=0.0001,
        batch_size=64,
        epochs=150,
        loss_fn=nn.MSELoss(),
        d_p=0.2,
        activation_fn=nn.ELU(),
        beta=50,
        regularisation: _REGULARISATION = "mmd",
    ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.input_dim = input_dim
        self.dense_dim = dense_dim
        self.latent_dim = latent_dim
        self.d_p = d_p
        self.activation_fn = activation_fn
        self.beta = beta
        self.regularisation = regularisation

    def get_list_parameters(self):
        return [
            self.regularisation,
            self.beta,
            self.dense_dim,
            self.latent_dim,
            self.epochs,
        ]

    def save_parameters(self, save_dir):
        with open(os.path.join(save_dir, "parameters.txt"), "w") as f:
            f.write(f"Input_dim = {self.input_dim}\n")
            f.write(f"Dense_dim = {self.dense_dim}\n")
            f.write(f"Latent_dim = {self.latent_dim}\n")
            f.write(f"Regularisation = {self.regularisation}\n")
            f.write(f"Beta = {self.beta}\n")
            f.write(f"Epochs = {self.epochs}\n")
            f.write(f"Loss_fn = {self.loss_fn}\n")
            f.write(f"Activation_fn = {self.activation_fn}\n")
            f.write(f"Dropout = {self.d_p}\n")
            f.write(f"Batch_size = {self.batch_size}\n")
            f.write(f"Learning_rate = {self.lr}\n")
            f.write(f"Weight_decay = {self.weight_decay}\n")
