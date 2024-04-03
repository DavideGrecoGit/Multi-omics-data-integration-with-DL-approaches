import argparse
import os
import time

import pandas as pd
import torch

from networks.classifiers import MoGCN
from networks.tuning import CustomClsTuning
from utils.data import get_bool_mask, get_class_weight, get_dataset, get_fold_masks
from utils.plots import (
    get_chisq,
    plot_confusion_matrix,
    plot_km,
    plot_pam50_latent_space,
    save_mean_metrics,
    save_metrics,
)
from utils.utils import load_config, save_config, setup_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = {"MoGCN": MoGCN}


def k_fold_experiment(config, args):

    # dataset
    gt_df = pd.read_csv(config["gt_path"])
    dataset = get_dataset(config)
    config["input_dim"] = dataset.x.shape[1]
    config["n_classes"] = len(dataset.class_labels)

    metrics = []
    fold_train_masks, fold_val_masks = get_fold_masks(
        config["k_fold_path"], config["n_kfolds"]
    )

    y = dataset.y

    for k in range(config["n_kfolds"]):

        print(f"\n=== Fold {k+1} ===\n")

        # assign masks
        dataset.train_mask = fold_train_masks[k]
        dataset.val_mask = fold_val_masks[k]

        dataset.class_weights = get_class_weight(y[dataset.train_mask])

        model = CustomClsTuning(config)
        model.to(DEVICE)
        data = dataset.to(DEVICE)

        # Train & val
        val_acc, val_f1, best_epoch = model.train_loop(data, args.verbose)

        # Log-rank test
        chi, p_value = get_chisq(gt_df, model.predict_cls(data), args.verbose)

        results = [val_acc, val_f1, chi, p_value, best_epoch]
        metrics.append(results)
        print(results)

    save_mean_metrics(
        metrics,
        save_dir,
        columns=[
            "mean_acc",
            "sd_acc",
            "mean_f1",
            "sd_f1",
            "mean_chisqr",
            "sd_chsqr",
            "mean_p_value",
            "sd_p_value",
            "mean_best_epoch",
            "sd_best_epoch",
        ],
    )

    return metrics


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-exp_id",
        help="Experiment id",
        default=time.strftime("%m%d%H%M%S", time.gmtime()),
    )

    parser.add_argument(
        "-verbose",
        help="Whether or not to print metrics and losses",
        type=bool,
        default=False,
    )

    parser.add_argument(
        "-config",
        help="Path to JSON config file",
        default="./config.json",
    )

    args = parser.parse_args()

    # setup
    config = load_config(args.config)
    setup_seed(config["seed"])

    save_dir = os.path.join(
        "results",
        args.exp_id,
        f"{config['n_kfolds']}_fold",
    )
    os.makedirs(save_dir)

    k_fold_experiment(config, args)

    save_config(config, os.path.join(save_dir, "config.json"))
