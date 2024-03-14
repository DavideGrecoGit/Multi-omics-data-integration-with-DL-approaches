import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from networks.GNNs import MTLR_GCN
from sklearn.model_selection import StratifiedKFold
from sksurv.metrics import concordance_index_censored
from torch_geometric.data import Data
from utils.data import get_bool_mask, get_edge_index
from utils.plots import get_chisq, save_mean_metrics
from utils.utils import load_config, save_config, setup_seed

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")


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


def get_surv_dataset(config):
    # get surv_data
    gt_df = pd.read_csv(config["gt_path"])
    # gt_df["Survival_in_months"] = gt_df["Survival_in_days"] // 30

    # load latent
    latent_df = pd.read_csv(config["latent_path"])
    data_x = latent_df.iloc[:, 1:].values

    # load snf
    adj = pd.read_csv(config["snf_path"], header=None).values
    # adj = df_snf.iloc[:, 1:].values
    np.fill_diagonal(adj, 0)
    edge_index, edge_attr = get_edge_index(adj)

    # gen y
    y_true = get_discrete_time(gt_df, config["n_buckets"])

    # dataset
    dataset = Data(
        x=torch.tensor(data_x, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(y_true, dtype=torch.int),
    )

    # dataset.surv_mask = surv_mask
    dataset.e = gt_df["Status"]
    dataset.t = gt_df["Survival_in_days"]

    return dataset


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_id",
        help="Experiment id",
        default=time.strftime("%m%d%H%M%S", time.gmtime()),
    )
    parser.add_argument(
        "--config_path",
        help="Path to JSON config file",
        default="./config.json",
    )
    args = parser.parse_args()

    # setup
    config = load_config(args.config_path)
    setup_seed(config["seed"])

    save_dir = os.path.join(
        "results",
        f"surv_{config['n_kfolds']}_fold",
        args.exp_id,
    )
    os.makedirs(save_dir)
    metrics = []

    # split mask
    gt_df = pd.read_csv(config["gt_path"])
    train_mask = get_bool_mask(gt_df, pd.read_csv(config["train_path"])["Sample_ID"])
    train_y = gt_df["class"][train_mask]
    train_ids = gt_df["Sample_ID"][train_mask]

    # dataset
    dataset = get_surv_dataset(config)

    # k-fold
    skf = StratifiedKFold(n_splits=config["n_kfolds"], shuffle=False)

    for k, (train_index, val_index) in enumerate(skf.split(train_ids, train_y)):

        # assign masks
        dataset.train_mask = get_bool_mask(gt_df, gt_df["Sample_ID"][train_index])
        dataset.val_mask = get_bool_mask(gt_df, gt_df["Sample_ID"][val_index])

        # Instanziate model
        config["gnn_input_dim"] = dataset.x.shape[1]

        model = MTLR_GCN(config)

        model.to(DEVICE)
        data = dataset.to(DEVICE)

        # Train & val
        model.train_loop(data)

        pred = model.predict_risk(data, data.val_mask)

        c_index = concordance_index_censored(
            dataset.e[data.val_mask],
            dataset.t[data.val_mask],
            pred["risk"].detach().cpu().numpy(),
        )[0]
        print(c_index)
        metrics.append(c_index)

        # Save predictions
        # save_fold_dir = os.path.join(save_dir, f"fold_{k+1}")
        # os.makedirs(save_fold_dir, exist_ok=True)

    #     pred = model.get_predictions(data, save_dir=save_fold_dir)

    means = np.array(metrics).mean(axis=0)
    stds = np.array(metrics).std(axis=0)

    print(f"Mean K-fold metrics: {means}, SD: {stds}\n")
    save_mean_metrics(metrics, save_dir, columns=["mean_c_index", "sd_c_index"])
    # save_config(config, os.path.join(save_dir, "config.json"))
