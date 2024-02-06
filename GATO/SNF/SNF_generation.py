import os
import sys

DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, DIR)

import argparse
import os
import torch
from networks.VAEs import Params_VAE, VAE
from utils.data import (
    get_data,
    get_pam50_labels,
)
import pandas as pd
import numpy as np
import snf
from torch_geometric.utils import to_edge_index
import gower as gw


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


SEED = 42
N_FOLDS = 5
METABRIC_PATH = "../data/MBdata_33CLINwMiss_1KfGE_1KfCNA.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-omics",
        help="Type of integration CLI+RNA, CNA+RNA, CLI+CNA or CLI+CNA+RNA",
        type=str,
        default="CLI+RNA",
    )
    parser.add_argument(
        "-from_latent",
        help="Generate graph from VAEs latent spaces",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "-remove_unknown",
        help="Remove samples with unkown Ground Truth class",
        type=bool,
        default=True,
    )

    args = parser.parse_args()

    metabric = get_data(METABRIC_PATH, complete_metabric_path=METABRIC_PATH)
    omics_combinations = args.omics.split(",")

    for omics_types in omics_combinations:
        omics_names = omics_types.split("+")
        print(f"\n>>> {omics_names} >>>\n")

        if args.from_latent:
            for k in range(1, N_FOLDS + 1):
                print(f"=== FOLD {k} ===")

                adj_all = []

                for name in omics_names:
                    input_dim = 1000
                    if name == "CLI":
                        input_dim = 350

                    dense_dim = 256
                    latent_dim = 128

                    vae_path = f"../../IntegrativeVAE/results/VAE_{name}/0122141739/fold_{k}/VAE.pth"

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
                    "../data/SNF",
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
            snf_path = "../data/SNF"
            os.makedirs(snf_path, exist_ok=True)

            np.savetxt(
                os.path.join(
                    snf_path,
                    f"snf_{('_').join(omics_names)}.csv",
                ),
                snf_all,
                delimiter=",",
            )
