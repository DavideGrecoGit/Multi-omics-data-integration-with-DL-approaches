import argparse
import os
import time

import pandas as pd
import torch
from networks.GNNs import MoGCN_GCN
from sklearn.model_selection import StratifiedKFold
from utils.data import get_bool_mask, get_dataset
from utils.plots import get_chisq, save_mean_metrics
from utils.utils import load_config, save_config, setup_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        f"cls_{config['n_kfolds']}_fold",
        args.exp_id,
    )
    os.makedirs(save_dir)
    metrics = []

    # get train ids
    gt_df = pd.read_csv(config["gt_path"])

    train_mask = get_bool_mask(gt_df, pd.read_csv(config["train_path"])["Sample_ID"])
    train_y = gt_df["class"][train_mask]
    train_ids = gt_df["Sample_ID"][train_mask]

    # k-fold
    skf = StratifiedKFold(n_splits=config["n_kfolds"], shuffle=False)

    for k, (train_index, val_index) in enumerate(skf.split(train_ids, train_y)):

        # load dataset
        train_mask = get_bool_mask(gt_df, gt_df["Sample_ID"][train_index])
        val_mask = get_bool_mask(gt_df, gt_df["Sample_ID"][val_index])
        dataset = get_dataset(config, train_mask, val_mask=val_mask)

        # Instanziate model
        config["gnn_input_dim"] = dataset.x.shape[1]
        config["n_classes"] = len(dataset.class_labels)

        model = MoGCN_GCN(config)

        model.to(DEVICE)
        data = dataset.to(DEVICE)

        # Train & val
        val_acc, val_f1 = model.train_loop(data)

        # Log-rank test
        chi, p_value = get_chisq(gt_df, model.get_predictions(data))
        metrics.append([val_acc, val_f1, chi, p_value])

        # Save predictions
        save_fold_dir = os.path.join(save_dir, f"fold_{k+1}")
        os.makedirs(save_fold_dir, exist_ok=True)

        pred = model.get_predictions(data, save_dir=save_fold_dir)

    save_mean_metrics(metrics, save_dir)
    save_config(config, os.path.join(save_dir, "config.json"))
