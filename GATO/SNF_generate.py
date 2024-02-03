import argparse
import time
import os
import torch
import torch.nn as nn
from networks.VAEs import Params_VAE, VAE
from networks.GNNs import Params_GNN, GAT
from data import get_data, get_pam50_labels, plot_latent_space, plot_confusion_matrix
import pandas as pd
import numpy as np
from torch_geometric.data import Data
import snf
from torch_geometric.utils import to_edge_index
import gower as gw

# def get_edge_index(snf_path, threshold):
#     X = np.genfromtxt(snf_path, delimiter=",")
#     X[X < threshold] = 0
#     return to_edge_index((torch.tensor(X, dtype=torch.float).to_sparse()))


def get_edge_index(snf_path, threshold=None, N_largest=None):
    X = np.genfromtxt(snf_path, delimiter=",")
    if N_largest:
        idx = np.argpartition(X.ravel(), -N_largest * 2)[-N_largest * 2 :]
        threshold = X.ravel()[idx].min()

    if threshold:
        X[X < threshold] = 0

    return to_edge_index((torch.tensor(X, dtype=torch.float).to_sparse()))


def calc_edge_index(X, threshold=None, N_largest=None):
    if N_largest:
        idx = np.argpartition(X.ravel(), -N_largest * 2)[-N_largest * 2 :]
        threshold = X.ravel()[idx].min()

    if threshold:
        X[X < threshold] = 0

    return to_edge_index((torch.tensor(X, dtype=torch.float).to_sparse()))


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
        "-net_type",
        help="Generate graph from VAEs latent spaces",
        type=str,
        default="latent",
    )
    parser.add_argument(
        "-remove_unknown",
        help="Remove samples with unkown Ground Truth class",
        type=bool,
        default=True,
    )

    args = parser.parse_args()

    metabric_path = "./data/MBdata_33CLINwMiss_1KfGE_1KfCNA.csv"
    fold_dir = "./data/5-fold_pam50stratified/"
    file_name = "MBdata_33CLINwMiss_1KfGE_1KfCNA"

    metabric = get_data(METABRIC_PATH)
    omics_combinations = args.omics.split(",")

    for omics_types in omics_combinations:
        omics_names = omics_types.split("+")
        print(f"\n>>> {omics_names} >>>\n")

        if args.net_type == "latent":
            print(args.net_type)
            for k in range(1, N_FOLDS + 1):
                print(f"=== FOLD {k} ===")

                adj_all = []

                for name in omics_names:
                    input_dim = 1000
                    if name == "CLI":
                        input_dim = 350

                    dense_dim = 256
                    latent_dim = 128

                    vae_path = f"/home/davide/Desktop/Projects/Multi-omics-data-integration-with-DL-approaches/IntegrativeVAE/results/VAE_{name}/0122141739/fold_{k}/VAE.pth"

                    vae_params = Params_VAE(input_dim, dense_dim, latent_dim)
                    vae = VAE(vae_params)
                    vae.load_state_dict(torch.load(vae_path))
                    vae.eval()
                    vae.to(DEVICE)

                    _, _, _, _, z = vae.forward(
                        torch.tensor(metabric[name], dtype=torch.float32)
                    )
                    z = z.detach().cpu().numpy()
                    adj_all.append(snf.make_affinity(z, metric="correlation"))

                if len(adj_all) > 1:
                    snf_all = snf.compute.snf(adj_all)

                else:
                    snf_all = adj_all[0]

                np.fill_diagonal(snf_all, 0)
                latent_path = os.path.join(
                    "/home/davide/Desktop/Projects/Multi-omics-data-integration-with-DL-approaches/GATO/data/SNF",
                    f"fold_{k}",
                )
                os.makedirs(latent_path, exist_ok=True)

                np.savetxt(
                    os.path.join(
                        latent_path,
                        f"snf_{('_').join(omics_names)}_latent.csv",
                    ),
                    snf_all,
                    delimiter=",",
                )
        else:
            adj_all = []
            for name in omics_names:
                match name:
                    case "CLI":
                        adj_all.append(gw.gower_matrix(metabric["CLI"]))
                    case _:
                        adj_all.append(
                            snf.make_affinity(metabric[name], metric="correlation")
                        )
            if len(adj_all) > 1:
                snf_all = snf.compute.snf(adj_all)
            else:
                snf_all = adj_all[0]

            np.fill_diagonal(snf_all, 0)
            snf_path = "/home/davide/Desktop/Projects/Multi-omics-data-integration-with-DL-approaches/GATO/data/SNF"
            os.makedirs(snf_path, exist_ok=True)

            np.savetxt(
                os.path.join(
                    snf_path,
                    f"snf_{('_').join(omics_names)}.csv",
                ),
                snf_all,
                delimiter=",",
            )
