import argparse
import time
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from networks import Params_GNN, GAT, Params_VAE, VAE
from data import Omics, get_data, get_pam50_labels
import pandas as pd
import numpy as np
from torch_geometric.data import Data
import snf
from torch_geometric.utils import to_edge_index


def get_edge_index(snf_path, threshold):
    X = np.genfromtxt(snf_path, delimiter=",")
    X[X < threshold] = 0
    return to_edge_index((torch.tensor(X, dtype=torch.long).to_sparse()))


def get_fold_mask(fold_path, metabric, remove_unknown=True):
    kfold = pd.read_csv(fold_path, index_col=None, header=0, low_memory=False)
    if remove_unknown:
        # Remove unknown classes
        kfold = kfold.drop(kfold[kfold["Pam50Subtype"] == "?"].index)

    return np.array(metabric["METABRIC_ID"].isin(kfold["METABRIC_ID"]))


SEED = 42
N_FOLDS = 5
K = 5
FOLD_DIR = "./data/5-fold_pam50stratified/"
FILE_NAME = "MBdata_33CLINwMiss_1KfGE_1KfCNA"
METABRIC_PATH = "./data/MBdata_33CLINwMiss_1KfGE_1KfCNA.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 30
CLASSES = 5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-omics",
        help="Type of integration CLI+RNA, CNA+RNA, CLI+CNA or CLI+CNA+RNA",
        type=str,
        default="CLI+RNA",
    )

    parser.add_argument(
        "-ds", help="The intermediate dense layers size", type=int, default=256
    )

    parser.add_argument("-ls", help="The latent layer size", type=int, default=64)
    parser.add_argument("-e", help="Number of epochs", type=int, default=50)
    parser.add_argument("-w_d", help="Weight decay", type=float, default=0.0001)
    parser.add_argument("-d_p", help="Dropout probability", type=float, default=0)
    parser.add_argument(
        "-remove_unknown",
        help="Remove samples with unkown Ground Truth class",
        type=bool,
        default=True,
    )

    parser.add_argument("-m", help="Model name", type=str, default="GAT")

    args = parser.parse_args()

    metabric_path = "./data/MBdata_33CLINwMiss_1KfGE_1KfCNA.csv"
    fold_dir = "./data/5-fold_pam50stratified/"
    file_name = "MBdata_33CLINwMiss_1KfGE_1KfCNA"
    id = time.strftime("%m%d%H%M%S", time.gmtime())

    metabric = get_data(METABRIC_PATH)
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

        snf_path = f"./results/SNF_{'+'.join(omics_names)}/0122140808/SNF_{'_'.join(omics_names)}.csv"

        gnn_params = Params_GNN(
            None,
            args.ds,
            args.ls,
            epochs=args.e,
            weight_decay=args.w_d,
            remove_unknown=args.remove_unknown,
            lr=0.01,
        )

        acc_scores = []

        for k in range(1, N_FOLDS + 1):
            print(f"=== FOLD {k} ===")

            # Get pre-processed data
            train_mask = get_fold_mask(
                os.path.join(FOLD_DIR, f"fold{k}", FILE_NAME + "_train.csv"), metabric
            )
            test_mask = get_fold_mask(
                os.path.join(FOLD_DIR, f"fold{k}", FILE_NAME + "_test.csv"), metabric
            )

            latents = []
            for name in omics_names:
                input_dim = 1000

                if name == "CLI":
                    input_dim = 350
                dense_dim = 256
                latent_dim = 128

                vae_path = (
                    f"../IntegrativeVAE/results/VAE_{name}/0122141739/fold_{k}/VAE.pth"
                )

                vae_params = Params_VAE(input_dim, dense_dim, latent_dim)
                vae = VAE(vae_params)
                vae.load_state_dict(torch.load(vae_path))
                vae.eval()
                vae.to(DEVICE)

                _, _, _, _, z = vae.forward(
                    torch.tensor(metabric[name], dtype=torch.float32)
                )
                latents.append(z.detach().cpu().numpy())

            edge_index, _ = get_edge_index(snf_path, 0.25)
            x = np.hstack(latents)
            print(x.shape)
            y = metabric["pam50np"]

            dataset = Data(
                x=torch.tensor(x, dtype=torch.float32),
                edge_index=edge_index,
                y=torch.tensor(y, dtype=torch.long),
            )

            dataset.pam50_labels = get_pam50_labels(metabric["pam50"])
            dataset.pam50 = metabric["pam50"]

            dataset.train_mask = torch.tensor(train_mask, dtype=torch.bool)
            dataset.test_mask = torch.tensor(test_mask, dtype=torch.bool)

            df = pd.DataFrame(dataset.y[dataset.train_mask], columns=["gt_classes"])
            class_weights = len(df["gt_classes"]) / df["gt_classes"].value_counts()
            dataset.class_weights = torch.tensor(
                class_weights.to_list(), dtype=torch.float
            )

            gnn_params.input_dim = latent_dim * len(latents)
            gnn_params.dense_dim = gnn_params.input_dim
            model = GAT(gnn_params).to(DEVICE)

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=gnn_params.lr,
                weight_decay=gnn_params.weight_decay,
            )
            data = dataset.to(DEVICE)

            model.train_loop(data, optimizer, EPOCHS)
            accTest, f1Test = model.validate(data, data.test_mask)
            acc_scores.append([accTest, f1Test])

metrics = np.array(acc_scores).mean(axis=0)
print(f"Mean Test metrics: {metrics}\n")
