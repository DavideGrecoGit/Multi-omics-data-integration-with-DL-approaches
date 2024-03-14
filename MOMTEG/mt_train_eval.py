import argparse
import copy
import os
import time

import numpy as np
import pandas as pd
import torch
from networks.baselines import MoGCN_GCN
from networks.tuning import CustomTuning
from sklearn.model_selection import StratifiedKFold
from utils.data import get_bool_mask, get_dataset, get_fold_masks
from utils.plots import (
    get_chisq,
    plot_confusion_matrix,
    plot_latent_space,
    save_mean_metrics,
    save_metrics,
)
from utils.utils import load_config, save_config, setup_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def k_fold_experiment(dataset, config, args):

    metrics = []
    fold_train_masks, fold_val_masks = get_fold_masks(
        config["k_fold_path"], config["n_kfolds"]
    )

    for k in range(config["n_kfolds"]):

        print(f"\n=== Fold {k+1} ===\n")

        # assign masks
        dataset.train_mask = fold_train_masks[k]
        dataset.val_mask = fold_val_masks[k]

        model = CustomTuning(config)
        model.to(DEVICE)
        data = dataset.to(DEVICE)

        # Train & val
        # val_acc, val_f1, val_c, best_epoch = model.train_loop(data, args.verbose)
        val_acc, val_f1, val_c = model.train_loop(data, args.verbose)

        if config["cls_loss_weight"] == 1:
            results = [val_acc, val_f1]
            # results = [val_acc, val_f1, best_epoch]
            columns = [
                "mean_acc",
                "sd_acc",
                "mean_f1",
                "sd_f1",
                # "mean_best_epoch",
                # "sd_best_epoch",
            ]
        if config["cls_loss_weight"] == 0:
            results = [val_c]
            columns = [
                "mean_c_index",
                "sd_c_index",
                # "mean_best_epoch",
                # "sd_best_epoch",
            ]
        if config["cls_loss_weight"] != 0 and config["cls_loss_weight"] != 1:
            results = [val_acc, val_f1, val_c]
            # results = [val_acc, val_f1, val_c, best_epoch]
            columns = [
                "mean_acc",
                "sd_acc",
                "mean_f1",
                "sd_f1",
                "mean_c_index",
                "sd_c_index",
                # "mean_best_epoch",
                # "sd_best_epoch",
            ]
        print(results)
        metrics.append(results)

    save_mean_metrics(
        metrics,
        save_dir,
        columns=columns,
    )

    return metrics


def test_experiment(dataset, config, args, save_plots=False):
    gt_df = pd.read_csv(config["gt_path"])
    train_mask = get_bool_mask(gt_df, pd.read_csv(config["train_path"])["Sample_ID"])

    # assign masks
    dataset.train_mask = train_mask
    dataset.val_mask = []
    dataset.test_mask = ~train_mask

    for m in range(len(dataset.train_mask)):
        if dataset.train_mask[m] == dataset.test_mask[m]:
            print(m)

    if config["net_type"] == "MoGCN":
        model = MoGCN_GCN(config)
    else:
        model = CustomTuning(config)

    model.to(DEVICE)
    data = dataset.to(DEVICE)

    # Train
    model.train_loop(data, args.verbose)

    acc, f1, chi, p_value = None, None, None, None
    if config["cls_loss_weight"] != 0:
        acc, f1 = model.evaluate_cls(data, data.test_mask)

        # Log-rank test
        chi, p_value = get_chisq(gt_df, model.predict_cls(data), args.verbose)

    c_index = None
    if config["cls_loss_weight"] != 1:
        c_index = model.evaluate_surv(data, data.test_mask)

    metrics = [acc, f1, chi, p_value, c_index]

    save_metrics(
        metrics,
        save_dir,
        columns=["acc", "f1", "chisqr", "p_value", "c_index"],
    )

    if save_plots:

        plot_latent_space(
            model.get_latent_space(data),
            # model.predict_cls(data),
            data.gt_labels,
            config,
            os.path.join(save_dir, "latent_pred.jpg"),
        )

        if config["cls_loss_weight"] != 0:
            plot_confusion_matrix(
                dataset.cls_y.cpu(),
                model.predict_cls(data),
                os.path.join(save_dir, "cm.jpg"),
                labels=data.class_labels,
                normalize="true",
            )

        model.predict_cls(data)

    return metrics


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_id",
        help="Experiment id",
        default=time.strftime("%m%d%H%M%S", time.gmtime()),
    )

    parser.add_argument(
        "--cls_w",
        help="Weight of classifier loss",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--surv_w",
        help="Weight of survivor prediction loss",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--k_fold",
        help="Number of k fold to use. If not specified, the test set is used",
        type=bool,
        default=None,
    )

    # parser.add_argument(
    #     "--n_trials",
    #     help="Number of trials to perform",
    #     type=int,
    #     default=1,
    # )

    parser.add_argument(
        "--verbose",
        help="Whether or not to print metrics and losses",
        type=bool,
        default=False,
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
    # PARAMS = copy.deepcopy(config)
    # PARAMS["net_type"] = args.net_type
    # PARAMS["surv_loss_weight"] = 1 - PARAMS["cls_loss_weight"]

    if args.cls_w is not None:
        config["cls_loss_weight"] = args.cls_w
    config["surv_loss_weight"] = 1 - config["cls_loss_weight"]

    save_dir = os.path.join(
        "results",
        args.exp_id,
        f"{config['n_kfolds']}_fold" if args.k_fold else "test",
    )
    os.makedirs(save_dir)

    # dataset
    dataset = get_dataset(config)
    config["input_dim"] = dataset.x.shape[1]
    config["n_classes"] = len(dataset.class_labels)

    if args.k_fold:
        # for cls_w in config["cls_loss_weight"]:
        #     PARAMS["cls_loss_weight"] == cls_w
        #     for n_buckets in config["n_buckets"]:
        #         PARAMS["n_buckets"] == n_buckets
        #         for n_edges in config["n_edges"]:
        #             PARAMS["n_edges"] == n_edges
        #             for batch_size in config["batch_size"]:
        #                 PARAMS["batch_size"] == batch_size
        #                 for lr in config["lr"]:
        #                     PARAMS["lr"] == lr
        #                     for act_fn in config["act_fn"]:
        #                         PARAMS["act_fn"] == act_fn
        #                         for trunk_ls in config["trunk_ls"]:
        #                             PARAMS["trunk_ls"] == trunk_ls
        #                             for cls_ds in config["cls_ds"]:
        #                                 PARAMS["cls_ds"] == cls_ds
        #                                 for surv_ds in config["surv_ds"]:
        #                                     PARAMS["surv_ds"] == surv_ds
        #                                     k_fold_experiment(dataset, PARAMS, args)
        k_fold_experiment(dataset, config, args)
    else:
        test_experiment(dataset, config, args, save_plots=True)

    save_config(config, os.path.join(save_dir, "config.json"))
