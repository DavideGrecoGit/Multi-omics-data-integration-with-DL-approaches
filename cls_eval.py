import argparse
import os
import time

import numpy as np
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
# MODELS = {"MoGCN": MoGCN, "CustomClsTuning": CustomClsTuning}


def test_experiment(config, args, save_plots=False):
    setup_seed(config["seed"])

    # dataset
    gt_df = pd.read_csv(config["gt_path"])
    dataset = get_dataset(config)
    config["input_dim"] = dataset.x.shape[1]
    config["n_classes"] = len(dataset.class_labels)

    train_mask = get_bool_mask(gt_df, pd.read_csv(config["train_path"])["Sample_ID"])

    # assign masks
    dataset.train_mask = train_mask
    dataset.val_mask = []
    dataset.test_mask = ~train_mask

    dataset.class_weights = get_class_weight(dataset.y[dataset.train_mask])

    model = CustomClsTuning(config)

    model.to(DEVICE)
    data = dataset.to(DEVICE)

    # Train
    model.train_loop(data, args.verbose)

    acc, f1 = model.evaluate_cls(data, data.test_mask)
    chi, p_value = get_chisq(gt_df, model.predict_cls(data), args.verbose)

    metrics = [acc, f1, chi, p_value]

    save_metrics(
        metrics,
        save_dir,
        columns=["acc", "f1", "chisqr", "p_value"],
    )

    latent = model.get_latent_space(data)
    np.savetxt(os.path.join(save_dir, "latent_data.csv"), latent, delimiter=",")

    predictions = model.predict_cls(data)
    np.savetxt(os.path.join(save_dir, "predictions.csv"), predictions, delimiter=",")

    if save_plots:

        plot_pam50_latent_space(
            latent,
            data.gt_labels,
            config,
            os.path.join(save_dir, "latent_pam50_GT.jpg"),
        )

        plot_pam50_latent_space(
            latent,
            [dataset.class_labels[i] for i in model.predict_cls(data)],
            # data.gt_labels,
            config,
            os.path.join(save_dir, "latent_pam50.jpg"),
        )

        plot_confusion_matrix(
            dataset.y.cpu(),
            model.predict_cls(data),
            os.path.join(save_dir, "cm.jpg"),
            labels=data.class_labels,
            normalize="true",
        )

        plot_km(
            dataset.T,
            dataset.E,
            model.predict_cls(data),
            os.path.join(save_dir, "km_plot.jpg"),
            conf=False,
            labels=dataset.class_labels,
        )

    return metrics


if __name__ == "__main__":
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-exp_id",
        help="Experiment id (valid only for single test experiments)",
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

    args = parser.parse_args()

    subfolder = None

    # setup
    if os.path.isdir(args.config):
        files = [
            os.path.join(args.config, file)
            for file in os.listdir(args.config)
            if file.endswith(".json")
        ]
        exp_ids = [os.path.splitext(file)[0] for file in files]
        subfolder = os.path.basename(args.config)

    else:
        files = [args.config]
        exp_ids = (
            [args.exp_id]
            if args.exp_id
            else [time.strftime("%m%d%H%M%S", time.gmtime())]
        )

    for i in range(len(files)):

        if subfolder:
            save_dir = os.path.join(
                "results",
                subfolder,
                exp_ids[i],
                "test",
            )
        else:
            save_dir = os.path.join(
                "results",
                exp_ids[i],
                "test",
            )
        os.makedirs(save_dir)

        config = load_config(files[i])

        test_experiment(config, args, save_plots=True)

        save_config(config, os.path.join(save_dir, "config.json"))
