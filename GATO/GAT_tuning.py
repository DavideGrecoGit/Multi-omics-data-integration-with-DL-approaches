import argparse
import os

import numpy as np
import optuna
import pandas as pd
import snf
import torch
import torch.nn as nn
from networks.GNNs import GAT, Params_GNN
from optuna.samplers import TPESampler
from torch_geometric.data import Data
from torch_geometric.utils import to_edge_index
from utils.data import get_data, get_pam50_labels
from utils.settings import DEVICE, FILE_NAME, FOLD_DIR, METABRIC_PATH, N_FOLDS, SEED


def get_edge_index(snf_path, threshold=None, N_largest=None):
    X = np.genfromtxt(snf_path, delimiter=",")
    if N_largest:
        idx = np.argpartition(X.ravel(), -N_largest * 2)[-N_largest * 2 :]
        threshold = X.ravel()[idx].min()

    if threshold:
        X[X < threshold] = 0

    return to_edge_index((torch.tensor(X, dtype=torch.float).to_sparse()))


def calc_edge_index(X, threshold=None, N_largest=None):
    if N_largest:
        idx = np.argpartition(X.ravel(), -N_largest * 2)[-N_largest * 2 :]
        threshold = X.ravel()[idx].min()

    if threshold:
        X[X < threshold] = 0

    return to_edge_index((torch.tensor(X, dtype=torch.float).to_sparse()))


def get_fold_mask(fold_path, metabric, remove_unknown=True):
    kfold = pd.read_csv(fold_path, index_col=None, header=0, low_memory=False)
    if remove_unknown:
        # Remove unknown classes
        kfold = kfold.drop(kfold[kfold["Pam50Subtype"] == "?"].index)

    return np.array(metabric["METABRIC_ID"].isin(kfold["METABRIC_ID"]))


def get_activation_fn(trial, activation_functions=None):
    if not activation_functions:
        activation_functions = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "elu": nn.ELU(),
            "leakyrelu": nn.LeakyReLU(),
        }

    activation = trial.suggest_categorical(
        "activation", list(activation_functions.keys())
    )
    return activation_functions[activation]


def define_hyperparams(trial):
    # lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    lr = trial.suggest_categorical("lr", [0.01, 0.005, 0.001, 0.0005])
    wd = trial.suggest_categorical("weight_decay", [0.0001, 0.00001, 0.000001])

    d_p = trial.suggest_int("d_p", 0, 8)
    d_p = d_p / 10

    activation_fn = get_activation_fn(trial)

    # ds = trial.suggest_categorical("ds", [0, 64, 128, 256])
    ds = trial.suggest_categorical("ds", [0, 256, 512, 758, 1024])
    # ls = trial.suggest_categorical("ls", [16, 32, 64])
    ls = trial.suggest_categorical("ls", [32, 64, 128, 256])

    n_edges = trial.suggest_categorical("n_edges", [0, 4500, 10000, 25000, 50000])

    use_edge_attr = trial.suggest_categorical("edge_attr", [True, False])

    loss_fn = nn.MSELoss(reduction="mean")

    return lr, wd, d_p, activation_fn, ds, ls, loss_fn, n_edges, use_edge_attr


def objective(trial):
    try:
        (
            lr,
            wd,
            d_p,
            activation_fn,
            ds,
            ls,
            loss_fn,
            n_edges,
            use_edge_attr,
        ) = define_hyperparams(trial)

        gnn_params = Params_GNN(
            None,
            ds,
            ls,
            n_edges,
            epochs=50,
            lr=lr,
            weight_decay=wd,
            loss_fn=loss_fn,
            d_p=d_p,
            activation_fn=activation_fn,
        )

        f1_scores = []

        for k in range(1, N_FOLDS + 1):
            print(f"=== FOLD {k} ===")

            # Get pre-processed data
            train_mask = get_fold_mask(
                os.path.join(FOLD_DIR, f"fold{k}", FILE_NAME + "_train.csv"), metabric
            )
            test_mask = get_fold_mask(
                os.path.join(FOLD_DIR, f"fold{k}", FILE_NAME + "_test.csv"), metabric
            )

            x = np.hstack([metabric[name] for name in OMICS_NAMES])

            snf_path = f"/home/davide/Desktop/Projects/Multi-omics-data-integration-with-DL-approaches/GATO/data/SNF/snf_{'_'.join(OMICS_NAMES)}.csv"
            edge_index, edge_attr = get_edge_index(snf_path, N_largest=args.edge_number)

            if n_edges == 0:
                edge_index = torch.tensor([[], []], dtype=torch.long)
                edge_attr = None

            if not use_edge_attr:
                edge_attr = None

            y = metabric["pam50np"]

            dataset = Data(
                x=torch.tensor(x, dtype=torch.float32),
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor(y, dtype=torch.long),
            )

            dataset.pam50_labels = get_pam50_labels(metabric["pam50"])
            dataset.pam50 = metabric["pam50"]

            dataset.train_mask = torch.tensor(train_mask, dtype=torch.bool)
            dataset.test_mask = torch.tensor(test_mask, dtype=torch.bool)

            df = pd.DataFrame(dataset.y[dataset.train_mask], columns=["gt_classes"])
            class_weights = len(df["gt_classes"]) / df["gt_classes"].value_counts()
            dataset.class_weights = torch.tensor(
                class_weights.to_list(), dtype=torch.float
            )

            gnn_params.input_dim = dataset.x.shape[1]
            gnn_params.n_classes = torch.unique(dataset.y).shape[0]
            model = GAT(gnn_params).to(DEVICE)

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=gnn_params.lr,
                weight_decay=gnn_params.weight_decay,
            )
            data = dataset.to(DEVICE)

            model.train_loop(data, optimizer, 50)
            accTest, f1Test = model.validate(data, data.test_mask)
            f1 = np.mean(f1Test)

            print(f"Fold: {k}, F1 score: {f1}")

            f1_scores.append(f1)

            trial.report(f1, k - 1)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return np.mean(np.array(f1_scores))

    except:
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-omics",
        help="Type of integration CLI, CNA or RNA",
        type=str,
        default="CLI+RNA",
    )
    parser.add_argument(
        "-study_name",
        help="Name of the optuna study to run",
        type=str,
        required=True,
    )
    parser.add_argument("-epochs", help="Number of epochs", type=int, default=20)
    parser.add_argument(
        "-n_trials", help="Number of optuna trials to run", type=int, default=100
    )
    parser.add_argument(
        "-timeout", help="Max seach running time (in minutes)", type=int, default=30
    )

    metabric = get_data(METABRIC_PATH)
    args = parser.parse_args()
    # omics_type = args.omics
    OMICS_NAMES = args.omics.split("+")
    epochs = args.epochs
    n_trials = args.n_trials
    timeout = args.timeout * 60  # in seconds

    study_name = args.study_name
    storage_path = f"sqlite:///{'_'.join(OMICS_NAMES)}_{study_name}_study.db"

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage_path,
        sampler=TPESampler(seed=SEED),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

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

    print("Best trial:")
    trial = study.best_trial
    print("\tMean F1 score: ", trial.value)

    print("\tParams: ")
    for key, value in trial.params.items():
        print(f"\t\t{key}: {value}")
