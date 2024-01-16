import argparse
import time
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from networks import Params_VAE, VAE
from data import Omics
import numpy as np
import pandas as pd
from classifiers import Benchmark_Classifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FOLDS = 5


class CNC_VAE(VAE):
    def __init__(self, params):
        super().__init__(params)

    def handle_input(self, x):
        return torch.cat(x, dim=1)


class H_VAE(VAE):
    def __init__(self, params, omics_names):
        super().__init__(params)

        input_VAEs = []

        for i in range(len(omics_names)):
            temp_params = Params_VAE(
                train_omics.get_input_dims(omics_names[i]),
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-omics",
        help="Type of integration CLI+RNA, CNA+RNA or CLI+CNA",
        type=str,
        default="CLI+RNA",
    )

    parser.add_argument(
        "-ds", help="The intermediate dense layers size", type=int, default=256
    )

    parser.add_argument("-ls", help="The latent layer size", type=int, default=64)
    parser.add_argument("-e", help="Number of epochs", type=int, default=150)
    parser.add_argument("-b", help="Beta value", type=int, default=50)
    parser.add_argument("-r", help="Regularisation function", type=str, default="mmd")
    parser.add_argument("-w_d", help="Weight decay", type=int, default=0.00001)
    parser.add_argument("-d_p", help="Dropout probability", type=int, default=0)

    parser.add_argument("-m", help="Model name", type=str, required=False)

    args = parser.parse_args()

    metabric_path = "./data/MBdata_33CLINwMiss_1KfGE_1KfCNA.csv"
    fold_dir = "./data/5-fold_pam50stratified/"
    file_name = "MBdata_33CLINwMiss_1KfGE_1KfCNA"
    id = time.strftime("%m%d%H%M%S", time.gmtime())

    omics_combinations = args.omics.split(",")

    for omics_types in omics_combinations:
        omics_names = omics_types.split("+")
        print(f"\n>>> {omics_names} >>>\n")
        save_dir = os.path.join(
            "results",
            f"{args.m}_{omics_types}",
            id,
        )
        os.makedirs(save_dir)

        vae_params = Params_VAE(
            None,
            args.ds,
            args.ls,
            epochs=args.e,
            beta=args.b,
            regularisation=args.r,
            weight_decay=args.w_d,
        )
        vae_params.save_parameters(save_dir)

        acc_scores = []

        for k in range(1, N_FOLDS + 1):
            print(f"=== FOLD {k} ===")

            # Data loading
            train_data_path = os.path.join(
                fold_dir, f"fold{k}", file_name + "_train.csv"
            )
            test_data_path = os.path.join(fold_dir, f"fold{k}", file_name + "_test.csv")

            train_omics = Omics(train_data_path, metabric_path, omics_names)
            test_omics = Omics(test_data_path, metabric_path, omics_names)

            train_dataloader = DataLoader(
                train_omics, batch_size=vae_params.batch_size, shuffle=False
            )
            test_dataloader = DataLoader(
                test_omics, batch_size=vae_params.batch_size, shuffle=False
            )

            match args.m:
                case "CNC-VAE":
                    vae_params.input_dim = train_omics.get_input_dims()
                    if omics_types == "CLI+CNA":
                        vae_params.loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
                    model = CNC_VAE(vae_params)
                case "H-VAE":
                    vae_params.input_dim = args.ds
                    model = H_VAE(vae_params, omics_names)
                case "VAE":
                    vae_params.input_dim = train_omics.get_input_dims(omics_types)
                    if omics_types == "CLI" or omics_types == "CNA":
                        vae_params.loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
                    model = VAE(vae_params, omics_index=0)

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=vae_params.lr,
                weight_decay=vae_params.weight_decay,
            )
            model.to(DEVICE)
            model.train_loop(
                train_dataloader, test_dataloader, optimizer, vae_params.epochs
            )

            save_path = os.path.join(save_dir, f"fold_{k}")
            os.makedirs(save_path)

            torch.save(model.state_dict(), os.path.join(save_path, f"{args.m}.pth"))

            # Embeddings
            train_save_path = os.path.join(save_path, "train_latent.csv")
            test_save_path = os.path.join(save_path, "test_latent.csv")

            train_embed = model.get_latent_space(train_dataloader, train_save_path)
            test_embed = model.get_latent_space(test_dataloader, test_save_path)
