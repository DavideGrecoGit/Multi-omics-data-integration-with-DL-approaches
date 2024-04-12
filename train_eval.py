import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch  # For building the networks
import torchtuples as tt  # Some useful functions
from pycox.evaluation import EvalSurv
from pycox.models import MTLR, CoxPH
from pycox.preprocessing.feature_transforms import OrderedCategoricalLong
from pycox.utils import kaplan_meier
from sklearn.model_selection import StratifiedKFold, train_test_split

# For preprocessing
from sklearn_pandas import DataFrameMapper
from tqdm import trange

from networks.classifier import CustomCls
from utils.data import get_bool_mask, get_chisq, get_class_weight, get_dataset
from utils.plots import (
    plot_confusion_matrix,
    plot_km,
    plot_pam50_latent_space,
    save_mean_metrics,
    save_metrics,
)
from utils.utils import load_config, save_config, setup_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def coxph(x_train, x_val, x_test, df_train, df_val, df_test, name=""):

    get_target = lambda df: (df["Survival_in_days"].values, df["Status"].values)
    durations_test, events_test = get_target(df_test)

    y_train = get_target(df_train)
    y_val = get_target(df_val)
    val = x_val, y_val

    in_features = x_train.shape[1]
    num_nodes = [32, 32]
    out_features = 1
    batch_norm = True
    dropout = 0.1
    output_bias = False

    net = tt.practical.MLPVanilla(
        in_features,
        num_nodes,
        out_features,
        batch_norm,
        dropout,
        output_bias=output_bias,
    )

    model = CoxPH(net, tt.optim.Adam)
    batch_size = 256
    epochs = 100
    callbacks = [tt.callbacks.EarlyStopping()]

    log = model.fit(
        x_train,
        y_train,
        batch_size,
        epochs,
        callbacks,
        False,
        val_data=val,
        val_batch_size=batch_size,
    )

    _ = model.compute_baseline_hazards()
    surv = model.predict_surv_df(x_test)

    return eval_model(surv, durations_test, events_test, f"{name}_coxph")


def mtlr(
    x_train,
    x_val,
    x_test,
    df_train,
    df_val,
    df_test,
    num_durations=10,
    scheme="quantiles",
    name="",
):

    labtrans = MTLR.label_transform(num_durations, scheme)

    get_target = lambda df: (df["Survival_in_days"].values, df["Status"].values)
    durations_test, events_test = get_target(df_test)

    y_train = labtrans.fit_transform(*get_target(df_train))
    y_val = labtrans.transform(*get_target(df_val))

    train = (x_train, y_train)
    val = (x_val, y_val)

    # plt discretation grid
    if n_trials == 1:
        plt.vlines(
            labtrans.cuts,
            0,
            1,
            colors="gray",
            linestyles="--",
            label="Discretization Grid",
        )
        kaplan_meier(*get_target(df_train)).plot(label="Kaplan-Meier")
        plt.ylabel("S(t)")
        plt.legend()
        _ = plt.xlabel("Time")
        plt.savefig(os.path.join(save_dir, "./discretization_grid.png"))
        plt.close()

    # model
    in_features = x_train.shape[1]
    num_nodes = [32, 32]
    out_features = labtrans.out_features
    batch_norm = True
    dropout = 0.1

    net = tt.practical.MLPVanilla(
        in_features, num_nodes, out_features, batch_norm, dropout
    )

    batch_size = 256
    epochs = 100
    callbacks = [tt.cb.EarlyStopping()]

    model = MTLR(net, tt.optim.Adam(0.01), duration_index=labtrans.cuts)
    log = model.fit(
        x_train, y_train, batch_size, epochs, callbacks, verbose=False, val_data=val
    )
    surv = model.interpolate(10).predict_surv_df(x_test)

    return eval_model(surv, durations_test, events_test, f"{name}_mtlr")


def transform_features(df_train, df_val, df_test):
    categorical = [(col, OrderedCategoricalLong()) for col in ["class"]]
    x_mapper = DataFrameMapper(categorical)

    x_train = x_mapper.fit_transform(df_train).astype("float32")
    x_val = x_mapper.transform(df_val).astype("float32")
    x_test = x_mapper.transform(df_test).astype("float32")

    return x_train, x_val, x_test


def eval_model(surv, durations_test, events_test, name):
    if n_trials == 1:
        surv.plot(drawstyle="steps-post")
        plt.ylabel("S(t | x)")
        _ = plt.xlabel("Time")
        plt.legend("", frameon=False)
        plt.savefig(os.path.join(save_dir, f"surv_pred_{name}.png"))
        plt.close()

    ev = EvalSurv(surv, durations_test, events_test, censor_surv="km")

    time_grid = np.linspace(durations_test.min(), durations_test.max(), 100)
    if n_trials == 1:
        ev.brier_score(time_grid).plot()
        plt.ylabel("IPCW Brier Score")
        _ = plt.xlabel("Time")
        plt.savefig(os.path.join(save_dir, f"surv_pred_{name}.png"))
        plt.close()

    return ev.concordance_td(), ev.integrated_brier_score(time_grid)


def surv_prediction(gt_df, predictions, n_trials=1, name=""):
    gt_df["class"] = predictions

    metrics = []
    for i in trange(n_trials):
        if n_trials == 1:

            train, val = train_test_split(
                gt_df["Sample_ID"][train_mask],
                test_size=0.2,
                stratify=gt_df["Strat_ID"][train_mask],
                random_state=42,
            )

            df_train = gt_df[get_bool_mask(gt_df, train)]
            df_test = gt_df[test_mask]
            df_val = gt_df[get_bool_mask(gt_df, val)]
        else:

            temp, test = train_test_split(
                gt_df["Sample_ID"],
                test_size=0.2,
                stratify=gt_df["Strat_ID"],
                shuffle=True,
            )

            train, val = train_test_split(
                gt_df["Sample_ID"],
                test_size=0.2,
                stratify=gt_df["Strat_ID"],
                shuffle=True,
            )

            df_train = gt_df[get_bool_mask(gt_df, train)]
            df_test = gt_df[get_bool_mask(gt_df, test)]
            df_val = gt_df[get_bool_mask(gt_df, val)]

        x_train, x_val, x_test = transform_features(df_train, df_val, df_test)

        mtlr_c, mtlr_ibs = mtlr(
            x_train, x_val, x_test, df_train, df_val, df_test, name=name
        )
        coxph_c, coxph_ibs = coxph(
            x_train, x_val, x_test, df_train, df_val, df_test, name=name
        )
        metrics.append([mtlr_c, mtlr_ibs, coxph_c, coxph_ibs])
        # metrics.append([mtlr_c, mtlr_ibs])

    np.savetxt(
        os.path.join(save_dir, f"surv_results_{n_trials}.csv"),
        np.array(metrics),
        delimiter=",",
        fmt="%s",
    )

    return


def cls_eval(config, train_mask, id=""):
    dataset = get_dataset(config)
    config["input_dim"] = dataset.x.shape[1]
    config["n_classes"] = len(dataset.class_labels)

    # assign masks
    dataset.train_mask = train_mask
    dataset.val_mask = []
    dataset.test_mask = ~train_mask

    dataset.class_weights = get_class_weight(dataset.y[dataset.train_mask])

    model = CustomCls(config)

    model.to(DEVICE)
    data = dataset.to(DEVICE)

    # Train
    model.train_loop(data, args.verbose)
    predictions = model.predict_cls(data)
    np.savetxt(os.path.join(save_dir, "predictions.csv"), predictions, delimiter=",")

    # Evaluation
    acc, f1 = model.evaluate_cls(data, data.test_mask)
    # combined data log-rank test
    all_chi, all_p_value = get_chisq(data.surv_data, predictions)
    # test data log-rank test
    test_chi, test_p_value = get_chisq(data.surv_data, predictions, mask=data.test_mask)

    save_metrics(
        [acc, f1, all_chi, all_p_value, test_chi, test_p_value],
        os.path.join(save_dir, "cls_metrics.csv"),
        columns=["acc", "f1", "all_chi", "all_p", "test_chi", "test_p"],
    )

    # Save plots
    latent = model.get_latent_space(data)

    plot_pam50_latent_space(
        latent,
        data.gt_labels,
        config,
        os.path.join(save_dir, "latent_pam50_GT.jpg"),
    )

    plot_pam50_latent_space(
        latent,
        np.array([dataset.class_labels[i] for i in model.predict_cls(data)]),
        # data.gt_labels,
        config,
        os.path.join(save_dir, "latent_pam50.jpg"),
    )

    plot_pam50_latent_space(
        latent[test_mask],
        np.array([dataset.class_labels[i] for i in model.predict_cls(data)])[test_mask],
        # data.gt_labels,
        config,
        os.path.join(save_dir, f"latent_test_{id}.jpg"),
    )

    plot_confusion_matrix(
        dataset.y.cpu(),
        model.predict_cls(data),
        os.path.join(save_dir, f"cm_nm_{id}.jpg"),
        labels=data.class_labels,
        normalize="true",
    )

    plot_confusion_matrix(
        dataset.y.cpu(),
        model.predict_cls(data),
        os.path.join(save_dir, f"cm_{id}.jpg"),
        labels=data.class_labels,
        normalize=None,
    )

    plot_confusion_matrix(
        dataset.y.cpu()[test_mask],
        model.predict_cls(data)[test_mask],
        os.path.join(save_dir, f"cm_test_nm_{id}.jpg"),
        labels=data.class_labels,
        normalize="true",
    )

    plot_confusion_matrix(
        dataset.y.cpu()[data.test_mask],
        model.predict_cls(data)[data.test_mask],
        os.path.join(save_dir, f"cm_test_{id}.jpg"),
        labels=data.class_labels,
        normalize=None,
    )

    plot_km(
        dataset.T[test_mask],
        dataset.E[test_mask],
        np.array([dataset.class_labels[i] for i in model.predict_cls(data)])[test_mask],
        os.path.join(save_dir, "km_plot_test.jpg"),
        conf=False,
    )

    plot_km(
        dataset.T[test_mask],
        dataset.E[test_mask],
        np.array([dataset.class_labels[i] for i in model.predict_cls(data)])[test_mask],
        os.path.join(save_dir, "km_plot_test_conf.jpg"),
        conf=True,
    )

    plot_km(
        dataset.T,
        dataset.E,
        np.array([dataset.class_labels[i] for i in model.predict_cls(data)]),
        os.path.join(save_dir, "km_plot.jpg"),
        conf=False,
    )

    return predictions


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
        "-surv_pred",
        help="Whether or not to perform survival prediction",
        type=bool,
        default=False,
    )

    parser.add_argument(
        "-config",
        help="Path to JSON config file",
        required=True,
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
        exp_ids = [os.path.basename(file).split(".")[0] for file in files]
        subfolder = os.path.basename(args.config)

    else:
        files = [args.config]
        exp_ids = (
            [args.exp_id]
            if args.exp_id
            else [time.strftime("%m%d%H%M%S", time.gmtime())]
        )

    n_trials = 1

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
        setup_seed(config["seed"])

        # dataset
        gt_df = pd.read_csv(config["gt_path"])
        # gt_df = pd.concat(
        #     [gt_df, pd.read_csv("./data/latent_data.csv").iloc[:, 1:]], axis=1
        # )

        train_mask = get_bool_mask(
            gt_df, pd.read_csv(config["train_path"])["Sample_ID"]
        )
        test_mask = get_bool_mask(gt_df, pd.read_csv(config["test_path"])["Sample_ID"])

        # Cls
        predictions = cls_eval(config, train_mask, exp_ids[i])

        # Surv
        if args.surv_pred:
            n_trials = 1
            surv_prediction(gt_df, predictions, n_trials, exp_ids[i])
            n_trials = 30
            surv_prediction(gt_df, predictions, n_trials, exp_ids[i])

        save_config(config, os.path.join(save_dir, "config.json"))

    if args.surv_pred and subfolder is not None:
        save_dir = os.path.join(
            "results",
            subfolder,
            "Baseline",
            "test",
        )
        os.makedirs(save_dir)
        gt_df = pd.read_csv(config["gt_path"])
        n_trials = 1
        surv_prediction(gt_df, gt_df["class"], n_trials, "Baseline")
        n_trials = 30
        surv_prediction(gt_df, gt_df["class"], n_trials, "Baseline")
