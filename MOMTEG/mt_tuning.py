import argparse
import copy
import os
import time

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from networks.tuning import CustomTuning
from optuna.samplers import GridSampler, TPESampler
from sklearn.model_selection import StratifiedKFold
from torch_geometric.nn import GATv2Conv, GCNConv
from utils.data import get_bool_mask, get_dataset, get_fold_masks
from utils.plots import get_chisq
from utils.utils import load_config, setup_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PARAMS = {}


def objective(trial):

    # define hyperparameters
    for key, value in config.items():
        if isinstance(value, list):
            # if key != "omics_labels":
            PARAMS[key] = trial.suggest_categorical(key, value)
    PARAMS["surv_loss_weight"] = 1 - PARAMS["cls_loss_weight"]

    # dataset
    dataset = get_dataset(PARAMS, gt_df, latent_df, adj)
    PARAMS["input_dim"] = dataset.x.shape[1]
    PARAMS["n_classes"] = len(dataset.class_labels)

    # metrics lists
    metrics = []
    custom_metrics = []

    try:
        # skf = StratifiedKFold(
        #     n_splits=config["n_kfolds"], random_state=config["seed"], shuffle=True
        # )

        # for k in range(args.k_fold):

        #     # print(f"=== {k+1} fold ===")
        #     # assign masks
        #     dataset.train_mask = get_bool_mask(gt_df, gt_df["Sample_ID"][train_index])

        #     dataset.val_mask = get_bool_mask(gt_df, gt_df["Sample_ID"][val_index])

        for k in range(config["n_kfolds"]):

            # assign masks
            dataset.train_mask = fold_train_masks[k]
            dataset.val_mask = fold_val_masks[k]

            # Instanziate model
            model = CustomTuning(PARAMS)

            model.to(DEVICE)
            data = dataset.to(DEVICE)

            # Train & val
            # val_acc, val_f1, val_c, best_epoch = model.train_loop(data)
            val_acc, val_f1, val_c = model.train_loop(data)

            # custom_metrics.append(best_epoch)

            if PARAMS["cls_loss_weight"] == 0:
                metrics.append(val_c)
            if PARAMS["cls_loss_weight"] == 1:
                metrics.append(val_f1)
            if PARAMS["cls_loss_weight"] != 0 and PARAMS["cls_loss_weight"] != 1:
                metrics.append([val_f1, val_c])

        if PARAMS["cls_loss_weight"] == 0 or PARAMS["cls_loss_weight"] == 1:
            std = np.array(metrics).std()
            mean = np.array(metrics).mean()

        else:
            std = np.array(metrics).std(axis=0).tolist()
            mean = np.array(metrics).mean(axis=0).tolist()

        # trial.set_user_attr("avg_best_epoch", np.array(custom_metrics).mean())
        trial.set_user_attr("avg_std", std)

        return mean
    except Exception as e:
        print(e)
        if PARAMS["cls_loss_weight"] == 0 or PARAMS["cls_loss_weight"] == 1:
            return [0]
        else:
            return [0, 0]


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-study_name",
        help="Name of the optuna study to run",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-n_trials", help="Number of optuna trials to run", type=int, default=100
    )
    parser.add_argument(
        "-timeout", help="Max seach running time (in minutes)", type=int, default=30
    )
    parser.add_argument(
        "-net_type",
        help="Type of the network type (MLP, GCN, GAT)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-gridsearch",
        help="Specifies whether to perform gridsearch or not",
        type=bool,
        required=False,
    )
    parser.add_argument(
        "-config",
        help="Path to JSON config file",
        required=True,
    )
    args = parser.parse_args()

    # setup
    config = load_config(args.config)
    PARAMS = copy.deepcopy(config)
    PARAMS["net_type"] = args.net_type

    setup_seed(config["seed"])
    n_trials = args.n_trials
    timeout = args.timeout * 60  # in seconds

    study_name = args.study_name
    storage_path = f"sqlite:///tuning/{study_name}_study.db"

    if args.gridsearch:
        search_space = {}
        for key, value in config.items():
            if isinstance(value, list):
                # if key != "omics_labels":
                search_space[key] = value
        sampler = GridSampler(search_space)
    else:
        sampler = TPESampler(seed=config["seed"])

    # get data
    gt_df = pd.read_csv(config["gt_path"])
    latent_df = pd.read_csv(config["latent_path"])
    adj = pd.read_csv(config["snf_path"], header=None).values

    fold_train_masks, fold_val_masks = get_fold_masks(
        config["k_fold_path"], config["n_kfolds"]
    )

    # Select study: single or multiple values
    if PARAMS["cls_loss_weight"] == 0 or PARAMS["cls_loss_weight"] == 1:

        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=storage_path,
            sampler=sampler,
            load_if_exists=True,
            setup_seed=config["seed"],
        )
    else:
        study = optuna.create_study(
            directions=["maximize", "maximize"],
            study_name=study_name,
            storage=storage_path,
            sampler=sampler,
            load_if_exists=True,
            setup_seed=config["seed"],
        )

    # optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    # output
    pruned_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    ]
    complete_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    print("Study statistics: ")
    print("\tN finished trials: ", len(study.trials))
    print("\tN pruned trials: ", len(pruned_trials))
    print("\tN completed trials: ", len(complete_trials))

    print("Best trials:")

    for trial in study.best_trials:
        print("\tBest mean values: ", trial.values)
        print("\tCustom metrics: ", trial.user_attrs)

        print("\tParams: ")
        for key, value in trial.params.items():
            print(f"\t\t{key}: {value}")
