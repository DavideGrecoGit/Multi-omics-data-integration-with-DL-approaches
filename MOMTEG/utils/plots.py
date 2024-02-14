import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sksurv.compare import compare_survival


def plot_latent_space(h, y, config, save_path):
    z = TSNE(n_components=2, random_state=config["seed"]).fit_transform(h)

    sns_plot = sns.scatterplot(
        x=z[:, 0],
        y=z[:, 1],
        hue=y,
        palette=sns.color_palette("bright")[: config["n_classes"]],
    )

    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(y_true, pred, save_path, labels=None, normalize="true"):
    cm = confusion_matrix(y_true, pred, normalize=normalize)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    cm_display.plot().figure_.savefig(save_path)
    plt.close()


def save_mean_metrics(metrics, save_dir):
    means = np.array(metrics).mean(axis=0)
    stds = np.array(metrics).std(axis=0)

    print(f"Mean Test metrics: {means}, SD: {stds}\n")

    columns = [
        "mean_acc",
        "sd_acc",
        "mean_f1",
        "sd_f1",
        "mean_chisqr",
        "sd_chisqr",
        "mean_p_value",
        "sd_p_value",
    ]

    df = pd.DataFrame(
        [[means[0], stds[0], means[1], stds[1], means[2], stds[2], means[3], stds[3]]],
        columns=columns,
    )

    df.to_csv(os.path.join(save_dir, "metrics.csv"), index=False)


def get_chisq(surv_data, groups):
    data_y = surv_data[["Status", "Survival_in_days"]].to_records(index=False)
    chisq, pvalue, stats, covar = compare_survival(data_y, groups, return_stats=True)

    print(f"Chi-square: {chisq}, P-value: {pvalue}")
    return chisq, pvalue
