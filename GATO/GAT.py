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


def get_edge_index(snf_path, threshold=None, N_largest=None):
    X = np.genfromtxt(snf_path, delimiter=",")
    if N_largest:
        idx = np.argpartition(X.ravel(), -N_largest * 2)[-N_largest * 2 :]
        threshold = X.ravel()[idx].min()

    if threshold:
        X[X < threshold] = 0

    return to_edge_index((torch.tensor(X, dtype=torch.float).to_sparse()))


def merge_edge_indexes(omics_names, edge_number, k):
    edge_indices = []
    edge_attributes = []
    for name in omics_names:
        aff_path = f"/home/davide/Desktop/Projects/Multi-omics-data-integration-with-DL-approaches/GATO/data/SNF/fold_{k}/snf_{'_'.join(omics_names)}_latent.csv"
        e_i, e_a = get_edge_index(aff_path, N_largest=edge_number)
        edge_indices.append(e_i)
        edge_attributes.append(e_a)

    edge_index = [[], []]
    edge_attr = []
    for i in range(len(edge_indices[0][0])):
        if (edge_indices[0][0][i] == edge_indices[1][0][i]) & (
            edge_indices[0][1][i] == edge_indices[1][1][i]
        ):
            print("a")
        else:
            for n in range(2):
                edge_index[0].append(edge_indices[n][0][i])
                edge_index[1].append(edge_indices[n][1][i])
                # edge_attr.append(edge_attributes[n][i])
                # edge_attr.append(n)
                edge_attr.append([n, edge_attributes[n][i]])

    edge_index = torch.tensor(edge_index)
    edge_attr = torch.tensor(edge_attr)


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
        "-ds", help="The intermediate dense layers size", type=int, default=512
    )

    parser.add_argument("-ls", help="The latent layer size", type=int, default=64)
    parser.add_argument(
        "-edge_number", help="Number of edges to consider", type=int, default=14000
    )
    parser.add_argument("-e", help="Number of epochs", type=int, default=200)
    parser.add_argument("-wd", help="Weight decay", type=float, default=0.00001)
    parser.add_argument("-lr", help="Learning rate", type=float, default=0.0005)
    parser.add_argument("-dp", help="Dropout probability", type=float, default=0.2)
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

        gnn_params = Params_GNN(
            None,
            args.ds,
            args.ls,
            epochs=args.e,
            weight_decay=args.wd,
            remove_unknown=args.remove_unknown,
            lr=args.lr,
            n_edges=args.edge_number,
            d_p=args.dp,
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

            if args.m == "GATO":
                dense_dim = 256
                latent_dim = 128

                latents = []
                for name in omics_names:
                    input_dim = 1000
                    if name == "CLI":
                        input_dim = 350

                    vae_path = f"../IntegrativeVAE/results/VAE_{name}/0122141739/fold_{k}/VAE.pth"

                    vae_params = Params_VAE(input_dim, dense_dim, latent_dim)
                    vae = VAE(vae_params)
                    vae.load_state_dict(torch.load(vae_path))
                    vae.eval()
                    vae.to(DEVICE)

                    _, _, _, _, z = vae.forward(
                        torch.tensor(metabric[name], dtype=torch.float32)
                    )
                    z = z.detach().cpu().numpy()
                    latents.append(z)

                snf_path = f"/home/davide/Desktop/Projects/Multi-omics-data-integration-with-DL-approaches/GATO/data/SNF/fold_{k}/snf_{'_'.join(omics_names)}_latent.csv"

                edge_index, edge_attr = get_edge_index(
                    snf_path, N_largest=args.edge_number
                )

                gnn_params.input_dim = latent_dim * len(omics_names)
                x = np.hstack(latents)
            else:
                input_dim = 0
                for name in omics_names:
                    if name == "CLI":
                        input_dim = input_dim + 350
                    else:
                        input_dim = input_dim + 1000

                gnn_params.input_dim = input_dim
                x = np.hstack([metabric[name] for name in omics_names])

                snf_path = f"/home/davide/Desktop/Projects/Multi-omics-data-integration-with-DL-approaches/GATO/data/SNF/snf_{'_'.join(omics_names)}.csv"
                edge_index, edge_attr = get_edge_index(
                    snf_path, N_largest=args.edge_number
                )

            if args.edge_number == 0:
                edge_index = torch.tensor([[], []], dtype=torch.long)
                edge_attr = None

            y = metabric["pam50np"]

            dataset = Data(
                x=torch.tensor(x, dtype=torch.float32),
                edge_index=edge_index,
                # edge_attr=edge_attr,
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

            model = GAT(gnn_params).to(DEVICE)

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=gnn_params.lr,
                weight_decay=gnn_params.weight_decay,
            )
            data = dataset.to(DEVICE)

            model.train_loop(data, optimizer, args.e)
            accTest, f1Test = model.validate(data, data.test_mask)
            acc_scores.append([accTest, f1Test])

            save_path = os.path.join(save_dir, f"fold_{k}")
            os.makedirs(save_path)

            plot_latent_space(
                model.get_latent_space(data),
                data.y.cpu().numpy(),
                os.path.join(save_path, f"{args.m}_latent.jpg"),
            )

            plot_confusion_matrix(
                data.y[data.test_mask].cpu().numpy(),
                model.get_predictions(data, data.test_mask),
                os.path.join(save_path, f"{args.m}_cm.jpg"),
            )

            # torch.save(model.state_dict(), os.path.join(save_path, f"{args.m}.pth"))


means = np.array(acc_scores).mean(axis=0)
stds = np.array(acc_scores).std(axis=0)
print(f"Mean Test metrics: {means}, SD: {stds}\n")
columns = ["mean_acc", "sd_acc", "mean_f1", "sd_f1"]

df = pd.DataFrame(
    [[means[0], stds[0], means[1], stds[1]]],
    columns=columns,
)
df.to_csv(os.path.join(save_dir, "metrics.csv"), index=False)
gnn_params.save_parameters(save_dir)
