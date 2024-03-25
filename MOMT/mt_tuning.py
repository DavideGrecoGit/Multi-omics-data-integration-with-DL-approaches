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


def objective(trial):

    # define hyperparameters
    for key, value in TUNING.items():
        if isinstance(value, list):
            if key == "n_edges" and config["net_type"] == "MLP":
                continue
            if key == "batch_size" and config["net_type"] != "MLP":
                continue
            config[key] = trial.suggest_categorical(key, value)

    config["surv_loss_weight"] = 1 - config["cls_loss_weight"]

    # dataset
    dataset = get_dataset(config, gt_df, latent_df, adj)
    config["input_dim"] = dataset.x.shape[1]
    config["n_classes"] = len(dataset.class_labels)

    # metrics lists
    metrics = []
    epochs = []
    c_index = []
    try:

        for k in range(config["n_kfolds"]):

            # assign masks
            dataset.train_mask = fold_train_masks[k]
            dataset.val_mask = fold_val_masks[k]

            # Instanziate model
            model = CustomTuning(config)

            model.to(DEVICE)
            data = dataset.to(DEVICE)

            # Train & val
            val_acc, val_f1, val_c, val_ibs, best_epoch = model.train_loop(data)
            # Log-rank test
            chi, p_value = get_chisq(gt_df, model.predict_cls(data), args.verbose)

            epochs.append(best_epoch)

            if config["cls_loss_weight"] == 0:
                metrics.append([val_c, val_ibs])
            if config["cls_loss_weight"] == 1:
                metrics.append(val_f1)
            if config["cls_loss_weight"] != 0 and config["cls_loss_weight"] != 1:
                # metrics.append([val_f1, val_c, val_ibs])
                metrics.append([p_value, val_ibs])

        if config["cls_loss_weight"] == 1:
            std = np.array(metrics).std()
            mean = np.array(metrics).mean()

        else:
            std = np.array(metrics).std(axis=0).tolist()
            mean = np.array(metrics).mean(axis=0).tolist()

        trial.set_user_attr("avg_best_epoch", np.array(epochs).mean())
        trial.set_user_attr("avg_std", std)

        return mean
    except Exception as e:
        print(e)
        if config["cls_loss_weight"] == 1:
            return [0, 0]
        elif config["cls_loss_weight"] == 0:
            return [0, 1]
        else:
            return [1, 1]
            # return [0, 0, 1]


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
        default="./config.json",
    )
    parser.add_argument(
        "-config_tuning",
        help="Path to JSON config file for tuning",
        required=True,
    )
    args = parser.parse_args()

    # params setup
    config = load_config(args.config)
    TUNING = load_config(args.config_tuning)

    # Override config params with TUNING params that are not lists -> custom parameters for this tuning modle but will not be searched by optuna
    for key, value in TUNING.items():
        if not isinstance(value, list):
            config[key] = value
        if key == "cls_loss_weight" and isinstance(value, list):
            # override with a value != than 0 or 1 to allow multi-task studies
            config[key] = 0.5

    config["net_type"] = args.net_type
    setup_seed(config["seed"])

    # get data
    gt_df = pd.read_csv(config["gt_path"])
    latent_df = pd.read_csv(config["latent_path"])
    adj = pd.read_csv(config["snf_path"], header=None).values

    fold_train_masks, fold_val_masks = get_fold_masks(
        config["k_fold_path"], config["n_kfolds"]
    )

    # Study setup
    n_trials = args.n_trials
    timeout = args.timeout * 60  # in seconds

    study_name = args.study_name
    storage_path = f"sqlite:///tuning/{study_name}_study.db"

    if args.gridsearch:
        search_space = {}
        for key, value in TUNING.items():
            if isinstance(value, list):
                if key == "n_edges" and config["net_type"] == "MLP":
                    continue
                if key == "batch_size" and config["net_type"] != "MLP":
                    continue

                search_space[key] = value
        sampler = GridSampler(search_space, seed=config["seed"])
    else:
        sampler = TPESampler(seed=config["seed"])

    # Select study: single or multiple values
    if config["cls_loss_weight"] == 1:
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=storage_path,
            sampler=sampler,
            load_if_exists=True,
        )
    elif config["cls_loss_weight"] == 0:
        study = optuna.create_study(
            directions=["maximize", "minimize"],
            study_name=study_name,
            storage=storage_path,
            sampler=sampler,
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(
            directions=["minimize", "minimize"],
            # directions=["maximize", "maximize", "minimize"],
            study_name=study_name,
            storage=storage_path,
            sampler=sampler,
            load_if_exists=True,
        )

    # optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, gc_after_trial=True)

    # output
    complete_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    print("Study statistics: ")
    print("\tN finished trials: ", len(study.trials))
    print("\tN completed trials: ", len(complete_trials))

    print("Best trials:")

    for trial in study.best_trials:
        print("\tBest mean values: ", trial.values)
        print("\tCustom metrics: ", trial.user_attrs)

        print("\tParams: ")
        for key, value in trial.params.items():
            print(f"\t\t{key}: {value}")
