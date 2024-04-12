import argparse
import copy

import numpy as np
import optuna
import pandas as pd
import torch
from optuna.samplers import TPESampler

from networks.classifier import CustomCls
from utils.data import get_chisq, get_class_weight, get_dataset, get_fold_masks
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
            if key == "n_heads" and "GAT" not in config["net_type"]:
                config["n_heads"] = 1
                continue
            config[key] = trial.suggest_categorical(key, value)

    # dataset
    dataset = get_dataset(config, gt_df, latent_df, adj)
    config["input_dim"] = dataset.x.shape[1]
    config["n_classes"] = len(dataset.class_labels)
    cls_y = dataset.y

    # metrics lists
    metrics = []
    custom = []

    try:

        for k in range(config["n_kfolds"]):

            # assign masks
            dataset.train_mask = fold_train_masks[k]
            dataset.val_mask = fold_val_masks[k]

            dataset.class_weights = get_class_weight(cls_y[dataset.train_mask])

            # Instanziate model
            model = CustomCls(config)

            model.to(DEVICE)
            data = dataset.to(DEVICE)

            # Train & val
            best_model, val_acc, val_f1, best_epoch = model.train_loop(data)

            custom.append(best_epoch)

            if args.metric == "f1":
                metrics.append(val_f1)
            else:
                if best_model is not None:
                    model = CustomCls(config)
                    model.load_state_dict(best_model)
                    model.to(DEVICE)

                chi, p_value = get_chisq(
                    data.surv_data,
                    model.predict_cls(data),
                    mask=data.val_mask,
                )
                metrics.append([val_f1, chi])

        trial.set_user_attr("avg_epoch", np.array(custom).mean())

        if args.metric == "f1":
            return np.array(metrics).mean()

        return np.array(metrics).mean(axis=0).tolist()

    except Exception as e:

        print(e)

        if args.metric == "f1":
            return 0
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
        help="Type of the network type (MLP, GCN, GAT, GATv2)",
        type=str,
        required=True,
    )

    parser.add_argument(
        "-cls_type",
        help="Type of the network type for classification (MLP, GCN, GAT, GATv2)",
        type=str,
        default="MLP",
    )

    parser.add_argument(
        "-metric",
        help="Metric to use for fine-tuning (f1 or f1+chi-square)",
        type=str,
        required=True,
    )

    parser.add_argument(
        "-n_edges",
        help="Number of edges to select for graph generation",
        type=int,
        default=None,
    )

    parser.add_argument(
        "-conf_tuning",
        help="Path to JSON config file",
        required=True,
    )

    args = parser.parse_args()

    # params setup
    TUNING = load_config(args.conf_tuning)

    config = copy.deepcopy(TUNING)
    setup_seed(config["seed"])
    config["net_type"] = args.net_type
    config["cls_type"] = args.cls_type

    if args.n_edges is not None:
        config["n_edges"] = args.n_edges

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

    sampler = TPESampler(seed=config["seed"])

    if args.metric == "f1":
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=storage_path,
            sampler=sampler,
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(
            directions=["maximize", "maximize"],
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

    for trial in study.best_trials[:3]:
        print("\tBest mean values: ", trial.values)
        print("\tCustom metrics: ", trial.user_attrs)

        print("\tParams: ")
        for key, value in trial.params.items():
            print(f"\t\t{key}: {value}")
