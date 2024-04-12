import os

import numpy as np
import pandas as pd
import torch
from sksurv.compare import compare_survival
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


def get_edge_index(X, N_largest=None):
    if N_largest == 0:
        X = np.eye(X.shape[0], dtype=int)
        return to_edge_index((torch.tensor(X, dtype=torch.float).to_sparse()))

    if N_largest is not None:
        threshold = np.sort(np.triu(X).ravel())[-N_largest:].min()
    else:
        threshold = 0.005

    X[X < threshold] = 0

    return to_edge_index((torch.tensor(X, dtype=torch.float).to_sparse()))


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


def get_class_weight(cls_y, n_classes=4):
    df = pd.DataFrame(cls_y, columns=["gt_classes"])
    class_weights = len(df["gt_classes"]) / (
        n_classes * df["gt_classes"].value_counts()
    )
    return torch.tensor(class_weights.sort_index().to_list(), dtype=torch.float)


def get_chisq(surv_data, groups, mask=None):

    if len(np.unique(groups)) == 1:
        return 0, 1

    if mask is not None:
        surv_data = surv_data[mask]
        groups = groups[mask]

    chisq, pvalue, stats, covar = compare_survival(surv_data, groups, return_stats=True)

    return chisq, pvalue


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

    # dataset
    dataset = Data(
        x=torch.tensor(data_x, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(gt_df["class"].to_list(), dtype=torch.long),
    )

    dataset.E = gt_df["Status"]
    dataset.T = gt_df["Survival_in_days"]
    dataset.surv_data = gt_df[["Status", "Survival_in_days"]].to_records(index=False)

    dataset.class_labels = get_ordered_class_lables(gt_df)
    dataset.gt_labels = gt_df["Pam50 Subtype"]

    return dataset


def get_bool_mask(df_gt, df_ids):
    return np.array(df_gt["Sample_ID"].isin(df_ids))
