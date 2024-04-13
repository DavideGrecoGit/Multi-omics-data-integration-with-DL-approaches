import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sksurv.nonparametric import kaplan_meier_estimator

from utils.data import get_chisq


def plot_pam50_latent_space(latent, pam_labels, config, save_path):
    sns.set_theme(context="paper", style="white", font_scale=1.25)

    idx = np.argsort(pam_labels)
    latent = latent[idx]
    pam_labels = pam_labels[idx]
    z = TSNE(n_components=2, random_state=config["seed"]).fit_transform(latent)

    sns_plot = sns.scatterplot(
        x=z[:, 0],
        y=z[:, 1],
        hue=pam_labels,
        palette=sns.color_palette("bright", n_colors=config["n_classes"]),
    )

    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(y_true, pred, save_path, labels=None, normalize="true"):
    cm = confusion_matrix(y_true, pred, normalize=normalize)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    cm_display.plot().figure_.savefig(save_path)
    plt.close()


def save_metrics(
    metrics,
    save_path,
    columns=[
        "acc",
        "f1",
        "chisqr",
        "p_value",
    ],
):
    print(f"Test metrics: {metrics}\n")

    df = pd.DataFrame(
        [metrics],
        columns=columns,
    )

    df.to_csv(save_path, index=False)


def save_mean_metrics(
    metrics,
    save_path,
    metric_names=[
        "mean_acc",
        "sd_acc",
        "mean_f1",
        "sd_f1",
        "mean_chisqr",
        "sd_chisqr",
        "mean_p_value",
        "sd_p_value",
    ],
):
    means = np.array(metrics).mean(axis=0)
    stds = np.array(metrics).std(axis=0)

    print(f"Mean: {means}, SD: {stds}\n")

    if len(metric_names) > 1:
        df = pd.DataFrame(
            [
                np.array(
                    [[means[i], stds[i]] for i in range(len(metric_names))]
                ).flatten()
            ],
            columns=np.array([[f"mean_{name}", f"std_{name}"] for name in metric_names])
            .flatten()
            .tolist(),
        )
    else:
        df = pd.DataFrame(
            [[means, stds]],
            columns=[metrics],
        )

    df.to_csv(save_path, index=False)


def plot_km(T, E, pam50=None, save_path=None, conf=False):

    sns.set_theme(context="paper", style="dark", font_scale=1.25)
    p = None

    if pam50 is None:
        time, survival_prob, conf_int = kaplan_meier_estimator(
            E, T // 365, conf_type="log-log"
        )

        plt.step(time, survival_prob, where="post")
        if conf:
            plt.fill_between(time, conf_int[0], conf_int[1], alpha=0.25, step="post")
    else:
        for group_type in np.unique(pam50):
            mask = pam50 == group_type

            time_treatment, survival_prob_treatment, conf_int = kaplan_meier_estimator(
                E[mask],
                T[mask] // 365,
                conf_type="log-log",
                conf_level=0.90,
            )
            plt.step(
                time_treatment,
                survival_prob_treatment,
                where="post",
                label=f"{group_type}",
            )
            plt.legend(loc="upper right")

            if conf:
                plt.fill_between(
                    time_treatment, conf_int[0], conf_int[1], alpha=0.25, step="post"
                )
        chi, p = get_chisq(pd.concat([E, T], axis=1).to_records(index=False), pam50)

    plt.ylim(0, 1.05)
    plt.ylabel(r"Survival probability ${S}(t)$")
    plt.xlabel(r"Year $(t)$")

    if p is not None:
        plt.text(0, 0.1, f"p-value = {round(p,5)}", fontsize=11)

    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
