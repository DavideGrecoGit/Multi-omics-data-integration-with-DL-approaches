import argparse
import os
import time

import pandas as pd
import torch
from networks.basemodels import MoGCN_GCN
from networks.tuning import CustomTuning
from utils.data import get_bool_mask, get_dataset, get_fold_masks
from utils.plots import (
    get_chisq,
    plot_confusion_matrix,
    plot_km,
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

        if config["net_type"] == "MoGCN":
            model = MoGCN_GCN(config)
        else:
            model = CustomTuning(config)
        model.to(DEVICE)
        data = dataset.to(DEVICE)

        # Train & val
        val_acc, val_f1, val_c, best_epoch = model.train_loop(data, args.verbose)
        # val_acc, val_f1, val_c = model.train_loop(data, args.verbose)

        if config["cls_loss_weight"] != 0:
            # Log-rank test
            chi, p_value = get_chisq(gt_df, model.predict_cls(data), args.verbose)

        if config["cls_loss_weight"] == 1:
            # results = [val_acc, val_f1]
            results = [val_acc, val_f1, chi, p_value, best_epoch]
            columns = [
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
            ]
        if config["cls_loss_weight"] == 0:
            results = [val_c, best_epoch]
            columns = [
                "mean_c_index",
                "sd_c_index",
                "mean_best_epoch",
                "sd_best_epoch",
            ]
        if config["cls_loss_weight"] != 0 and config["cls_loss_weight"] != 1:
            # results = [val_acc, val_f1, val_c]
            results = [val_acc, val_f1, chi, p_value, val_c, best_epoch]
            columns = [
                "mean_acc",
                "sd_acc",
                "mean_f1",
                "sd_f1",
                "mean_chisqr",
                "sd_chsqr",
                "mean_p_value",
                "sd_p_value",
                "mean_c_index",
                "sd_c_index",
                "mean_best_epoch",
                "sd_best_epoch",
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

        if config["cls_loss_weight"] != 1 and config["cls_loss_weight"] != 0:
            plot_km(
                dataset.T,
                dataset.E,
                model.predict_cls(data),
                os.path.join(save_dir, "km_plot.jpg"),
                conf=False,
            )

            plot_km(
                dataset.T,
                dataset.E,
                model.predict_cls(data),
                os.path.join(save_dir, "km_plot_conf.jpg"),
                conf=True,
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
        "-cls_w",
        help="Weight of classifier loss",
        type=float,
        default=None,
    )
    parser.add_argument(
        "-surv_w",
        help="Weight of survivor prediction loss",
        type=float,
        default=None,
    )

    parser.add_argument(
        "-k_fold",
        help="Number of k fold to use. If not specified, the test set is used",
        type=bool,
        default=None,
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
    parser.add_argument(
        "-config_tuning",
        help="Path to JSON config file for tuning",
        default=None,
    )
    args = parser.parse_args()

    # setup
    config = load_config(args.config)
    # TUNING = load_config(args.config_tuning)
    setup_seed(config["seed"])

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
    gt_df = pd.read_csv(config["gt_path"])
    dataset = get_dataset(config)
    config["input_dim"] = dataset.x.shape[1]
    config["n_classes"] = len(dataset.class_labels)

    iter = 0
    if args.k_fold:
        k_fold_experiment(dataset, config, args)
        # print(iter)
    else:
        test_experiment(dataset, config, args, save_plots=True)

    save_config(config, os.path.join(save_dir, "config.json"))
