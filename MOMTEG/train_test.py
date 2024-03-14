import argparse
import os
import time

import pandas as pd
import torch
from networks.GNNs import MoGCN_GCN
from sklearn.model_selection import StratifiedKFold
from utils.data import get_bool_mask, get_dataset
from utils.plots import (
    get_c_index,
    get_chisq,
    plot_confusion_matrix,
    plot_latent_space,
    save_metrics,
)
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
        "test",
        args.exp_id,
    )
    os.makedirs(save_dir)

    # get pam50 data
    pam50_df = pd.read_csv(config["pam50_path"])
    train_mask = get_bool_mask(pam50_df, pd.read_csv(config["train_path"])["Sample_ID"])
    test_mask = get_bool_mask(pam50_df, pd.read_csv(config["test_path"])["Sample_ID"])

    # get surv_data
    surv_df = pd.read_csv(config["surv_path"])
    # mask is necessary since not all training samples are included into survival data
    surv_mask = get_bool_mask(pam50_df, surv_df["Sample_ID"])

    # load dataset
    dataset = get_dataset(config, train_mask, test_mask=test_mask)

    # Instanziate model
    config["gnn_input_dim"] = dataset.x.shape[1]
    config["n_classes"] = len(dataset.class_labels)

    model = MoGCN_GCN(config)

    model.to(DEVICE)
    data = dataset.to(DEVICE)

    # Train & val
    model.train_loop(data)
    test_acc, test_f1 = model.evaluate(data, dataset.test_mask)

    pred = model.get_predictions(data, save_dir=save_dir)
    latent = model.get_latent_space(data, os.path.join(save_dir, "latent.csv"))

    # Log-rank test
    chi, p_value = get_chisq(surv_df, model.get_predictions(data, mask=surv_mask))

    train_mask = get_bool_mask(surv_df, pd.read_csv(config["train_path"])["Sample_ID"])
    test_mask = get_bool_mask(surv_df, pd.read_csv(config["test_path"])["Sample_ID"])
    c_index = get_c_index(latent[surv_mask], surv_df, train_mask, test_mask)

    # Save data

    plot_latent_space(
        latent, dataset.gt_labels, config, os.path.join(save_dir, "latent.jpg")
    )
    plot_confusion_matrix(
        dataset.y.cpu().numpy(),
        pred,
        os.path.join(save_dir, "cm.jpg"),
        labels=dataset.class_labels,
    )

    save_metrics([test_acc, test_f1, chi, p_value], save_dir)
    save_config(config, os.path.join(save_dir, "config.json"))
