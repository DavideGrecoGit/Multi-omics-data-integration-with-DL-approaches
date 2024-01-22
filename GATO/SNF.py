from data import get_data, get_pam50_labels
import os
import snf
import torch
from torch_geometric.utils import to_edge_index
import pandas as pd
import numpy as np
from VAEs import Params_VAE, VAE
from sklearn.preprocessing import minmax_scale
from torch_geometric.data import Data
from classifiers import GAT
import torch.nn as nn


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
EPOCHS = 50
CLASSES = 5

metabric = get_data(METABRIC_PATH)
omics_names = ["CLI", "RNA"]
snf_path = (
    f"./results/SNF_{'+'.join(omics_names)}/0122140808/SNF_{'_'.join(omics_names)}.csv"
)


acc_scores = []

for j in range(1, N_FOLDS + 1):
    print(f"=== FOLD {j} ===")

    train_mask = get_fold_mask(
        os.path.join(FOLD_DIR, f"fold{j}", FILE_NAME + "_train.csv"), metabric
    )
    test_mask = get_fold_mask(
        os.path.join(FOLD_DIR, f"fold{j}", FILE_NAME + "_test.csv"), metabric
    )

    latents = []
    for name in omics_names:
        input_dim = 1000

        if name == "CLI":
            input_dim = 350
        dense_dim = 256
        latent_dim = 128

        vae_path = f"../IntegrativeVAE/results/VAE_{name}/0115195808/fold_{j}/VAE.pth"

        vae_params = Params_VAE(input_dim, dense_dim, latent_dim)
        vae = VAE(vae_params)
        vae.load_state_dict(torch.load(vae_path))
        vae.eval()
        vae.to(DEVICE)

        _, _, _, _, z = vae.forward(torch.tensor(metabric[name], dtype=torch.float32))
        latents.append(z.detach().cpu().numpy())

    edge_index, _ = get_edge_index(snf_path, 0.25)

    print(len(edge_index[0]))
    x = np.hstack(latents)
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
    dataset.class_weights = torch.tensor(class_weights.to_list(), dtype=torch.float)

    # print(len(dataset.y[dataset.train_mask]))

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    i_dim = latent_dim * len(latents)
    layers = [(i_dim, 24)]

    model = GAT(Params_GNN).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    data = dataset.to(DEVICE)

    model.train_loop(data, optimizer, EPOCHS)
    accTest, f1Test = model.validate(data, data.test_mask)
    acc_scores.append([accTest, f1Test])

metrics = np.array(acc_scores).mean(axis=0)
print(f"Mean Test metrics: {metrics}\n")
