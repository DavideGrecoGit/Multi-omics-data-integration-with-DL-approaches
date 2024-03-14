import torch
import torch.nn as nn


class MoGCN_AE(nn.Module):
    def __init__(self, config, a=0.4, b=0.3, c=0.3):
        """
        :param in_feas_dim: a list, input dims of omics data
        :param latent_dim: dim of latent layer
        :param a: weight of omics data type 1
        :param b: weight of omics data type 2
        :param c: weight of omics data type 3
        """
        super().__init__()
        self.a = 0.6
        self.b = 0.1
        self.c = 0.3
        self.config = config
        self.in_feas = config["ae_input_dim"]
        self.latent = config["ae_latent_dim"]

        # encoders, multi channel input
        self.encoder_omics_1 = nn.Sequential(
            nn.Linear(self.in_feas[0], self.latent),
            nn.BatchNorm1d(self.latent),
            nn.Sigmoid(),
        )
        self.encoder_omics_2 = nn.Sequential(
            nn.Linear(self.in_feas[1], self.latent),
            nn.BatchNorm1d(self.latent),
            nn.Sigmoid(),
        )
        self.encoder_omics_3 = nn.Sequential(
            nn.Linear(self.in_feas[2], self.latent),
            nn.BatchNorm1d(self.latent),
            nn.Sigmoid(),
        )
        # decoders
        self.decoder_omics_1 = nn.Sequential(nn.Linear(self.latent, self.in_feas[0]))
        self.decoder_omics_2 = nn.Sequential(nn.Linear(self.latent, self.in_feas[1]))
        self.decoder_omics_3 = nn.Sequential(nn.Linear(self.latent, self.in_feas[2]))

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

    def train_loop(self, train_loader, val_loader):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["ae_lr"])
        loss_fn = nn.MSELoss()

        for epoch in range(1, self.config["epochs"] + 1):
            self.train()

            for batch_ids, x in enumerate(train_loader):
                pass
