import os

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_edge_index

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_fold_masks(path, n_fold=5):
    train = [
        np.genfromtxt(
            os.path.join(path, f"train_fold_{k+1}.csv"),
            delimiter=",",
            dtype=bool,
        )
        for k in range(n_fold)
    ]
    val = [
        np.genfromtxt(
            os.path.join(path, f"val_fold_{k+1}.csv"),
            delimiter=",",
            dtype=bool,
        )
        for k in range(n_fold)
    ]

    return train, val


def get_edge_index(X, threshold=0.005, N_largest=None):
    if N_largest:
        # idx = np.argpartition(X.ravel(), -N_largest * 2)[-N_largest * 2 :]
        # threshold = X.ravel()[idx].min()

        threshold = np.sort(np.triu(X).ravel())[-N_largest:].min()

    X[X < threshold] = 0

    return to_edge_index((torch.tensor(X, dtype=torch.float).to_sparse()))


def get_tri_matrix(n_buckets, dimension_type=1):
    """
    Omiembed: Get tensor of the triangular matrix
    """
    if dimension_type == 1:
        ones_matrix = torch.ones(n_buckets, n_buckets + 1)
    else:
        ones_matrix = torch.ones(n_buckets + 1, n_buckets + 1)
    tri_matrix = torch.tril(ones_matrix)
    return tri_matrix


def get_omics(data_path):
    df = pd.read_csv(data_path, header=0, index_col=None)
    df.sort_values(by="Sample_ID", ascending=True, inplace=True)
    return df.iloc[:, 1:].values.astype(np.float64)


def get_ordered_class_lables(
    gt_df,
):
    counts = gt_df[["Pam50 Subtype", "class"]].value_counts()
    counts = counts.reset_index().sort_values(["class"])
    return counts["Pam50 Subtype"].to_list()


def get_discrete_time(surv_df, n_buckets):
    buckets = np.linspace(0, surv_df["Survival_in_days"].max(), n_buckets + 1)

    y_true = []

    for i, (t, e) in enumerate(zip(surv_df["Survival_in_days"], surv_df["Status"])):
        y = np.zeros(n_buckets + 1)
        min_abs_value = [abs(b - t) for b in buckets]
        index = np.argmin(min_abs_value)

        if e:  # death occurs at t=2 -> y = [0,0,1,0,0,0]
            y[index] = 1.0

        else:
            y[(index):] = 1.0  # censor occurs at t=2 -> y = [0,0,1,1,1,1]
        y_true.append(y)

    # y_true = torch.tensor(y_true, dtype=torch.int)
    return np.array(y_true)


def get_dataset(config, gt_df=None, latent_df=None, adj=None):
    if gt_df is None:
        # get surv_data
        gt_df = pd.read_csv(config["gt_path"])

    if latent_df is None:
        # load latent
        latent_df = pd.read_csv(config["latent_path"])
    data_x = latent_df.iloc[:, 1:].values

    # load snf
    if adj is None:
        adj = pd.read_csv(config["snf_path"], header=None).values
    np.fill_diagonal(adj, 0)
    edge_index, edge_attr = get_edge_index(adj, N_largest=config["n_edges"])

    y = []
    if config["cls_loss_weight"] != 1:
        y = get_discrete_time(gt_df, config["n_buckets"])

    # dataset
    dataset = Data(
        x=torch.tensor(data_x, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(y, dtype=torch.int),
    )

    dataset.cls_y = torch.tensor(gt_df["class"].to_list(), dtype=torch.long)

    # dataset.surv_mask = surv_mask
    dataset.E = gt_df["Status"]
    dataset.T = gt_df["Survival_in_days"]

    dataset.class_labels = get_ordered_class_lables(gt_df)
    dataset.gt_labels = gt_df["Pam50 Subtype"]

    return dataset


def get_bool_mask(df_gt, df_ids):
    return np.array(df_gt["Sample_ID"].isin(df_ids))
