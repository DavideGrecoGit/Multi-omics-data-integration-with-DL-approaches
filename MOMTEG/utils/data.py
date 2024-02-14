import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_edge_index


def get_edge_index(X, threshold=0.005):
    if threshold:
        X[X < threshold] = 0

    return to_edge_index((torch.tensor(X, dtype=torch.float).to_sparse()))


def get_ordered_class_lables(
    pam50_df,
):
    counts = pam50_df[["Pam50 Subtype", "class"]].value_counts()
    counts = counts.reset_index().sort_values(["class"])
    return counts["Pam50 Subtype"].to_list()


def get_dataset(config, train_index, val_index):

    # load latent
    latent_df = pd.read_csv(config["latent_path"])
    x = latent_df.iloc[:, 1:].values

    # load gt
    pam50_df = pd.read_csv(config["pam50_path"])
    y = pam50_df["class"].to_list()

    # load snf
    df_snf = pd.read_csv(config["snf_path"])
    adj = df_snf.iloc[:, 1:].values
    np.fill_diagonal(adj, 0)
    edge_index, edge_attr = get_edge_index(adj)

    dataset = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(y, dtype=torch.long),
    )

    dataset.class_labels = get_ordered_class_lables(pam50_df)
    dataset.gt_labels = pam50_df["Pam50 Subtype"]

    # load masks
    test_samples = pd.read_csv(config["test_path"])
    dataset.test_mask = get_bool_mask(pam50_df, test_samples["Sample_ID"])

    dataset.train_mask = get_bool_mask(pam50_df, pam50_df["Sample_ID"][train_index])

    dataset.val_mask = get_bool_mask(pam50_df, pam50_df["Sample_ID"][val_index])

    return dataset


def get_bool_mask(df_gt, df_ids):
    return np.array(df_gt["Sample_ID"].isin(df_ids))
