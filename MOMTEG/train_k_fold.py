import argparse
import json
import os
import time

import pandas as pd
import torch
from GNN import GCN
from sklearn.model_selection import StratifiedKFold
from utils.data import get_bool_mask, get_dataset
from utils.plots import get_chisq, save_mean_metrics
from utils.utils import setup_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e_id",
        help="Experiment id",
        default=time.strftime("%m%d%H%M%S", time.gmtime()),
    )
    args = parser.parse_args()

    # load config
    with open("config.json", "r") as f:
        config = json.load(f)

    # setup
    setup_seed(config["seed"])

    save_dir = os.path.join(
        "results",
        f"{config['n_kfolds']}_fold",
        args.e_id,
    )
    os.makedirs(save_dir)
    metrics = []

    # get train ids
    pam50_df = pd.read_csv(config["pam50_path"])
    train_mask = get_bool_mask(pam50_df, pd.read_csv(config["train_path"])["Sample_ID"])
    train_y = pam50_df["class"][train_mask]
    train_ids = pam50_df["Sample_ID"][train_mask]

    # get surv_data
    surv_df = pd.read_csv(config["surv_path"])
    # mask is necessary since not all training samples are included into survival data
    surv_mask = get_bool_mask(pam50_df, surv_df["Sample_ID"])

    # k-fold
    skf = StratifiedKFold(n_splits=config["n_kfolds"], shuffle=False)

    for k, (train_index, val_index) in enumerate(skf.split(train_ids, train_y)):

        # load dataset
        dataset = get_dataset(config, train_index, val_index)

        # Instanziate model
        config["gnn_input_dim"] = dataset.x.shape[1]
        config["n_classes"] = len(dataset.class_labels)

        model = GCN(config)

        model.to(DEVICE)
        data = dataset.to(DEVICE)

        # Train & val
        val_acc, val_f1 = model.train_loop(data)

        # Log-rank test
        chi, p_value = get_chisq(surv_df, model.get_predictions(data, mask=surv_mask))
        metrics.append([val_acc, val_f1, chi, p_value])

        # Save predictions
        save_fold_dir = os.path.join(save_dir, f"fold_{k+1}")
        os.makedirs(save_fold_dir, exist_ok=True)

        pred = model.get_predictions(data, save_dir=save_fold_dir)

    save_mean_metrics(metrics, save_dir)

    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f)
