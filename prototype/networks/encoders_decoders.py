import torch
from torch import nn
from networks.base_models import BaseModel
from utils.train_val_test import Early_Stopping, save_checkpoint


class FCBlock(nn.Module):
    """
    Linear => Norm1D => LeakyReLU
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        norm_layer=nn.BatchNorm1d,
        leaky_slope=0.2,
        dropout_p=0,
        activation=True,
        normalization=True,
        activation_name="LeakyReLU",
    ):
        """
        Construct a fully-connected block
        Parameters:
            input_dim (int)         -- the dimension of the input tensor
            output_dim (int)        -- the dimension of the output tensor
            norm_layer              -- normalization layer
            leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
            dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
            activation (bool)       -- need activation or not
            normalization (bool)    -- need normalization or not
            activation_name (str)   -- name of the activation function used in the FC block
        """
        super(FCBlock, self).__init__()
        # Linear
        self.fc_block = [nn.Linear(input_dim, output_dim)]
        # Norm
        if normalization:
            norm_layer = nn.BatchNorm1d
            self.fc_block.append(norm_layer(output_dim))
        # Dropout
        if 0 < dropout_p <= 1:
            self.fc_block.append(nn.Dropout(p=dropout_p))
        # LeakyReLU
        if activation:
            if activation_name.lower() == "leakyrelu":
                self.fc_block.append(
                    nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)
                )
            elif activation_name.lower() == "tanh":
                self.fc_block.append(nn.Tanh())
            else:
                raise NotImplementedError(
                    "Activation function [%s] is not implemented" % activation_name
                )

        self.fc_block = nn.Sequential(*self.fc_block)

    def forward(self, x):
        y = self.fc_block(x)
        return y


class FcVaeA(nn.Module):
    """
    Defines a fully-connected variational autoencoder for gene expression dataset
    """

    def __init__(
        self,
        omics_dims,
        norm_layer=nn.BatchNorm1d,
        leaky_slope=0.2,
        dropout_p=0,
        dim_1A=1024,
        dim_2A=1024,
        dim_3=512,
        latent_dim=256,
    ):
        """
        Construct a fully-connected variational autoencoder
        Parameters:
            omics_dims (list)       -- the list of input omics dimensions
            norm_layer              -- normalization layer
            leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
            dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
            latent_dim (int)        -- the dimensionality of the latent space
        """

        super(FcVaeA, self).__init__()

        self.A_dim = omics_dims[0]

        self.device = torch.device("cuda")
        self.loss = nn.MSELoss(reduction="mean")

        # ENCODER
        # Layer 1
        self.encode_fc_1A = FCBlock(
            self.A_dim,
            dim_1A,
            norm_layer=norm_layer,
            leaky_slope=leaky_slope,
            dropout_p=dropout_p,
            activation=True,
        )
        # Layer 2
        self.encode_fc_2A = FCBlock(
            dim_1A,
            dim_2A,
            norm_layer=norm_layer,
            leaky_slope=leaky_slope,
            dropout_p=dropout_p,
            activation=True,
        )
        # Layer 3
        self.encode_fc_3 = FCBlock(
            dim_2A,
            dim_3,
            norm_layer=norm_layer,
            leaky_slope=leaky_slope,
            dropout_p=dropout_p,
            activation=True,
        )
        # Layer 4
        self.encode_fc_mean = FCBlock(
            dim_3,
            latent_dim,
            norm_layer=norm_layer,
            leaky_slope=leaky_slope,
            dropout_p=0,
            activation=False,
            normalization=False,
        )
        self.encode_fc_log_var = FCBlock(
            dim_3,
            latent_dim,
            norm_layer=norm_layer,
            leaky_slope=leaky_slope,
            dropout_p=0,
            activation=False,
            normalization=False,
        )

        # DECODER
        # Layer 1
        self.decode_fc_z = FCBlock(
            latent_dim,
            dim_3,
            norm_layer=norm_layer,
            leaky_slope=leaky_slope,
            dropout_p=dropout_p,
            activation=True,
        )
        # Layer 2
        self.decode_fc_2 = FCBlock(
            dim_3,
            dim_2A,
            norm_layer=norm_layer,
            leaky_slope=leaky_slope,
            dropout_p=dropout_p,
            activation=True,
        )
        # Layer 3
        self.decode_fc_3A = FCBlock(
            dim_2A,
            dim_1A,
            norm_layer=norm_layer,
            leaky_slope=leaky_slope,
            dropout_p=dropout_p,
            activation=True,
        )
        # Layer 4
        self.decode_fc_4A = FCBlock(
            dim_1A,
            self.A_dim,
            norm_layer=norm_layer,
            leaky_slope=leaky_slope,
            dropout_p=0,
            activation=False,
            normalization=False,
        )

    def encode(self, x):
        level_2_A = self.encode_fc_1A(x[0])

        level_3_A = self.encode_fc_2A(level_2_A)

        level_4 = self.encode_fc_3(level_3_A)

        latent_mean = self.encode_fc_mean(level_4)
        latent_log_var = self.encode_fc_log_var(level_4)

        return latent_mean, latent_log_var

    def reparameterize(self, mean, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mean)

        return mean

    def decode(self, z):
        level_1 = self.decode_fc_z(z)

        level_2 = self.decode_fc_2(level_1)

        level_3_A = self.decode_fc_3A(level_2)

        recon_A = self.decode_fc_4A(level_3_A)

        return [recon_A]

    def get_last_encode_layer(self):
        return self.encode_fc_mean

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        recon_x = self.decode(z)
        return z, recon_x, mean, log_var

    def forward_pass(self, omics):
        x = omics.to(self.device)
        z, recon_x, mean, log_var = self.forward(x)

        # Loss
        reconstruction_loss = self.loss(recon_x, x)
        KLD = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())

        loss = reconstruction_loss + 0.01 * KLD

        return z, loss

    def get_latent_space(self, dataset):
        x_omics = dataset.omics_data

        with torch.no_grad():
            omics_1 = x_omics.to(self.device)

            z, recon_x, mean, log_var = self.forward(omics_1)
            return z.detach().cpu().numpy()


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


# class H_VAE(BaseModel):
#     """
#     Implementation of the paper MoGCN: A Multi-Omics Integration Method Based on
#     Graph Convolutional Network for Cancer Subtype Analysis

#     @doi: 10.3389/fgene.2022.806842
#     """

#     def __init__(
#         self,
#         input_dims,
#         activation_fn=nn.Sigmoid(),
#         dropout_p=0,
#         per_omics_dims=[100, 100, 100],
#         latent_dim=100,
#         loss_fn=[nn.MSELoss(reduction="mean")] * 3,
#         beta=[0.000020, 0.000020, 0.000020],
#         name="MoGCN_VAE",
#     ):
#         super().__init__(name)

#         self.activation_fn = activation_fn
#         self.loss_fn = loss_fn
#         self.input_dims = input_dims
#         self.per_omics_dims = per_omics_dims
#         self.latent_dim = latent_dim
#         self.beta = beta

#         # encoders, multi channel input
#         self.encoder_omics_1 = nn.Sequential(
#             nn.Linear(self.input_dims[0], self.per_omics_dims[0]),
#             nn.BatchNorm1d(self.per_omics_dims[0]),
#             nn.Dropout(p=dropout_p),
#             self.activation_fn,
#         )
#         self.encoder_omics_2 = nn.Sequential(
#             nn.Linear(self.input_dims[1], self.per_omics_dims[1]),
#             nn.BatchNorm1d(self.per_omics_dims[1]),
#             nn.Dropout(p=dropout_p),
#             self.activation_fn,
#         )
#         self.encoder_omics_3 = nn.Sequential(
#             nn.Linear(self.input_dims[2], self.per_omics_dims[2]),
#             nn.BatchNorm1d(self.per_omics_dims[2]),
#             nn.Dropout(p=dropout_p),
#             self.activation_fn,
#         )

#         self.encoder_mean_1 = nn.Linear(self.per_omics_dims[0], self.latent_dim)
#         self.encoder_log_var_1 = nn.Linear(self.per_omics_dims[0], self.latent_dim)

#         self.encoder_mean_2 = nn.Linear(self.per_omics_dims[1], self.latent_dim)
#         self.encoder_log_var_2 = nn.Linear(self.per_omics_dims[1], self.latent_dim)

#         self.encoder_mean_3 = nn.Linear(self.per_omics_dims[2], self.latent_dim)
#         self.encoder_log_var_3 = nn.Linear(self.per_omics_dims[2], self.latent_dim)

#         # decoders
#         self.decoder_z_1 = nn.Sequential(
#             nn.Linear(self.latent_dim, self.per_omics_dims[0]),
#             nn.BatchNorm1d(self.per_omics_dims[0]),
#             nn.Dropout(p=dropout_p),
#             self.activation_fn,
#         )

#         self.decoder_z_2 = nn.Sequential(
#             nn.Linear(self.latent_dim, self.per_omics_dims[1]),
#             nn.BatchNorm1d(self.per_omics_dims[1]),
#             nn.Dropout(p=dropout_p),
#             self.activation_fn,
#         )

#         self.decoder_z_3 = nn.Sequential(
#             nn.Linear(self.latent_dim, self.per_omics_dims[2]),
#             nn.BatchNorm1d(self.per_omics_dims[2]),
#             nn.Dropout(p=dropout_p),
#             self.activation_fn,
#         )

#         self.decoder_omics_1 = nn.Sequential(
#             nn.Linear(self.per_omics_dims[0], self.input_dims[0]),
#             nn.BatchNorm1d(self.per_omics_dims[0]),
#             nn.Dropout(p=dropout_p),
#             self.activation_fn,
#         )
#         self.decoder_omics_2 = nn.Sequential(
#             nn.Linear(self.per_omics_dims[1], self.input_dims[1]),
#             nn.BatchNorm1d(self.per_omics_dims[1]),
#             nn.Dropout(p=dropout_p),
#             self.activation_fn,
#         )
#         self.decoder_omics_3 = nn.Sequential(
#             nn.Linear(self.per_omics_dims[2], self.input_dims[2]),
#             nn.BatchNorm1d(self.per_omics_dims[2]),
#             nn.Dropout(p=dropout_p),
#             self.activation_fn,
#         )

#         # Variable initialization
#         for name, param in MoGCN_AE.named_parameters(self):
#             if "weight" in name:
#                 torch.nn.init.normal_(param, mean=0, std=0.1)
#             if "bias" in name:
#                 torch.nn.init.constant_(param, val=0)

#     def reparameterize(self, mean, log_var):
#         if self.training:
#             std = torch.exp(0.5 * log_var)
#             eps = torch.randn_like(std)
#             return eps.mul(std).add_(mean)

#         return mean

#     def forward(self, omics_1, omics_2, omics_3):
#         """
#         :param omics_1: omics data 1
#         :param omics_2: omics data 2
#         :param omics_3: omics data 3
#         """
#         # omics_1 to z_1
#         encoded_omics_1 = self.encoder_omics_1(omics_1)

#         latent_mean_1 = self.encoder_mean_1(encoded_omics_1)
#         latent_log_var_1 = self.encoder_log_var_1(encoded_omics_1)

#         z_1 = self.reparameterize(latent_mean_1, latent_log_var_1)

#         # omics_2 to z_2
#         encoded_omics_2 = self.encoder_omics_2(omics_2)

#         latent_mean_2 = self.encoder_mean_2(encoded_omics_2)
#         latent_log_var_2 = self.encoder_log_var_2(encoded_omics_2)

#         z_2 = self.reparameterize(latent_mean_2, latent_log_var_2)

#         # omics_3 to z_3
#         encoded_omics_3 = self.encoder_omics_3(omics_3)

#         latent_mean_3 = self.encoder_mean_3(encoded_omics_3)
#         latent_log_var_3 = self.encoder_log_var_3(encoded_omics_3)

#         z_3 = self.reparameterize(latent_mean_3, latent_log_var_3)

#         # z_1 to omics_1
#         decoded_z_1 = self.decoder_z_1(z_1)
#         decoded_omics_1 = self.decoder_omics_1(decoded_z_1)

#         # z_2 to omics_2
#         decoded_z_2 = self.decoder_z_2(z_2)
#         decoded_omics_2 = self.decoder_omics_2(decoded_z_2)

#         # z_3 to omics_3
#         decoded_z_3 = self.decoder_z_3(z_3)
#         decoded_omics_3 = self.decoder_omics_3(decoded_z_3)

#         return (
#             [
#                 [z_1, z_2, z_3],
#                 [latent_mean_1, latent_mean_2, latent_mean_3],
#                 [latent_log_var_1, latent_log_var_2, latent_log_var_3],
#             ],
#             [decoded_omics_1, decoded_omics_2, decoded_omics_3],
#         )

#     def get_loss(self, loss_fn, decoded_omics, omics, log_var, mean, beta):
#         reconstruction_loss = loss_fn(decoded_omics, omics)

#         KLD_1 = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())

#         return reconstruction_loss + beta * KLD_1

#     def forward_pass(self, x_omics):
#         omics_1 = x_omics[0].to(self.device)
#         omics_2 = x_omics[1].to(self.device)
#         omics_3 = x_omics[2].to(self.device)

#         latent, decoded_omics = self.forward(omics_1, omics_2, omics_3)

#         zs, means, log_vars = latent
#         x_omics = [omics_1, omics_2, omics_3]

#         losses = [
#             self.get_loss(
#                 self.loss_fn[i],
#                 decoded_omics[i],
#                 x_omics[i],
#                 log_vars[i],
#                 means[i],
#                 self.beta[i],
#             )
#             for i in range(len(x_omics))
#         ]

#         return decoded_omics, losses

#     def get_latent_space(self, dataset):
#         x_omics = dataset.omics_data

#         with torch.no_grad():
#             omics_1 = x_omics[0].to(self.device)
#             omics_2 = x_omics[1].to(self.device)
#             omics_3 = x_omics[2].to(self.device)

#             latent, decoded_omics = self.forward(omics_1, omics_2, omics_3)
#             return [z.detach().cpu().numpy() for z in latent[0]]

#     def train_loop(
#         self,
#         train_data,
#         optimizers,
#         output_path=None,
#         num_epochs=100,
#         val_data=None,
#         early_stopping_mode="train_loss",
#         tolerance=5,
#     ):
#         train_loss_ls = []
#         val_loss_ls = []
#         train_loss = 0.0
#         val_loss = 0

#         early_stopping = Early_Stopping(tolerance)

#         for epoch in range(num_epochs):
#             self.train()

#             train_loss_sums = [0.0 * len(optimizers)]
#             for batch_idx, (x_omics, _) in enumerate(train_data):
#                 outputs, losses = self.forward_pass(x_omics)

#                 for i in range(len(optimizers)):
#                     # Backpropagation
#                     optimizers[i].zero_grad()
#                     losses[i].backward()
#                     optimizers[i].step()

#                     train_loss_sums[i] += losses[i].item()

#             train_losses = [loss_sum / len(train_data) for loss_sum in train_loss_sums]

#             train_loss_ls.append(train_losses)

#             if val_data:
#                 val_losses = self.val_loop(val_data)
#                 val_loss_ls.append(val_losses)

#             if epoch == 0 or (epoch + 1) % 5 == 0:
#                 print(f"epoch: {epoch + 1} | train loss: {train_loss:.4f}", end="")
#                 if val_data:
#                     for val_loss in val_losses:
#                         print(f" | val loss: {val_loss:.4f}")

#             match early_stopping_mode:
#                 case "train_loss":
#                     new_loss = train_loss
#                 case "val_loss":
#                     new_loss = val_loss

#             if early_stopping.check(self, new_loss, epoch + 1):
#                 break

#         print(
#             f"\nBest epoch: {early_stopping.best_epoch} | {early_stopping_mode} {early_stopping.best_loss:.4f}"
#         )

#         if output_path:
#             save_checkpoint(early_stopping.best_model, output_path)

#         if val_data:
#             return {"train": train_loss_ls, "val": val_loss_ls}
#         return {"train": train_loss_ls}


class H_Low_VAE(BaseModel):
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
        for name, param in MoGCN_AE.named_parameters(self):
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


class H_VAE(BaseModel):
    def __init__(
        self,
        input_dims,
        name="H_VAE",
        activation_fn=nn.Sigmoid(),
        dropout_p=0,
        hidden_dim=100,
        latent_dim=100,
        loss_fn=nn.MSELoss(reduction="mean"),
        beta=0.000020,
    ):
        super().__init__(name)

        self.VAE_omics_1 = H_Low_VAE(input_dims[0], "H_low_VAE_omics_1")
        self.VAE_omics_2 = H_Low_VAE(input_dims[1], "H_low_VAE_omics_2")
        self.VAE_omics_3 = H_Low_VAE(input_dims[2], "H_low_VAE_omics_3")
