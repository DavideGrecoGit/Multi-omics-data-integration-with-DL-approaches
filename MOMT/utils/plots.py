import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sksurv.compare import compare_survival
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.nonparametric import kaplan_meier_estimator


def plot_latent_space(latent, pam_labels, config, save_path):
    z = TSNE(n_components=2, random_state=config["seed"]).fit_transform(latent)

    sns_plot = sns.scatterplot(
        x=z[:, 0],
        y=z[:, 1],
        hue=pam_labels,
        palette=sns.color_palette("bright")[: config["n_classes"]],
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
    save_dir,
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

    df.to_csv(os.path.join(save_dir, "metrics.csv"), index=False)


def save_mean_metrics(
    metrics,
    save_dir,
    columns=[
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

    if (len(columns) / 2) > 1:
        df = pd.DataFrame(
            [
                np.array(
                    [[means[i], stds[i]] for i in range((len(columns) // 2))]
                ).flatten()
            ],
            columns=columns,
        )
    else:
        df = pd.DataFrame(
            [[means, stds]],
            columns=columns,
        )

    df.to_csv(os.path.join(save_dir, "metrics.csv"), index=False)


def plot_km(T, E, pam50, save_path, conf=False):

    # if group_clm is None:
    #     time, survival_prob, conf_int = kaplan_meier_estimator(
    #         E, T, conf_type="log-log"
    #     )

    #     plt.step(time, survival_prob, where="post")
    #     if conf:
    #         plt.fill_between(time, conf_int[0], conf_int[1], alpha=0.25, step="post")
    # else:
    for group_type in np.unique(pam50):
        mask = pam50 == group_type
        time_treatment, survival_prob_treatment, conf_int = kaplan_meier_estimator(
            E[mask],
            T[mask],
            conf_type="log-log",
        )

        plt.step(
            time_treatment,
            survival_prob_treatment,
            where="post",
            label=f"{group_type}",
        )
        plt.legend(loc="best")

        if conf:
            plt.fill_between(
                time_treatment, conf_int[0], conf_int[1], alpha=0.25, step="post"
            )

    plt.ylim(0, 1)
    plt.ylabel(r"est. probability of survival $\hat{S}(t)$")
    plt.xlabel("time $t$")
    plt.savefig(save_path)
    plt.close()


def get_chisq(surv_data, groups, verbose=False, mask=None):

    if len(np.unique(groups)) == 1:
        return 0, 1

    if mask is not None:
        surv_data = surv_data[mask]
        groups = groups[mask]

    data_y = surv_data[["Status", "Survival_in_days"]].to_records(index=False)
    chisq, pvalue, stats, covar = compare_survival(data_y, groups, return_stats=True)

    if verbose:
        print(f"Chi-square: {chisq}, P-value: {pvalue}")
    return chisq, pvalue


def get_c_index(latent, surv_data, train_mask, test_mask):
    data_y = surv_data[["Status", "Survival_in_days"]].to_records(index=False)

    estimator = CoxPHSurvivalAnalysis()
    estimator.fit(latent[train_mask], data_y[train_mask])

    c_index = estimator.score(latent[test_mask], data_y[test_mask])
    print("\nC-index: ", c_index)
    return c_index
