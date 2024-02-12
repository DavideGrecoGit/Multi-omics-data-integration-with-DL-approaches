import os
import sys

DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, DIR)


from utils.data import get_data, get_pam50_labels
import os
import numpy as np
import snf
from snf import compute
from sklearn import cluster
from sklearn.metrics import silhouette_score
import argparse
import time

from utils.settings import METABRIC_PATH

METABRIC_PATH = os.path.join("../", METABRIC_PATH)


def compute_silhoutte(
    metabric,
    omics_names,
    metric_real="correlation",
    metric_discrete="hamming",
    K=20,
    mu=0.5,
    n_clusters=5,
):
    affinities = []
    for name in omics_names:
        metric = metric_real
        if name == "CLI" or name == "CNA":
            metric = metric_discrete
        affinities.append(snf.make_affinity(metabric[name], metric=metric, K=K, mu=mu))

    if len(omics_names) == 1:
        fused = affinities[0]
    else:
        fused = compute.snf(affinities, K=K)

    fused_labels = cluster.spectral_clustering(fused, n_clusters=n_clusters)
    score = silhouette_score(fused, fused_labels)
    return score, fused


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-omics",
        help="Type of integration CLI+RNA, CNA+RNA, CLI+CNA or CLI+CNA+RNA",
        type=str,
        default="CLI+RNA",
        # default="CLI+RNA,CNA+RNA,CLI+CNA,CLI+CNA+RNA",
    )
    parser.add_argument(
        "-k",
        help="Number of neighbors (0,N) to consider when creating affinity matrix. See Notes of :py:func`snf.compute.affinity_matrix` for more details.",
        type=str,
        default="5,20,200",
    )
    parser.add_argument(
        "-mu",
        help="Normalization factor (0,1) to scale similarity kernel when constructing affinity matrix. See Notes of :py:func`snf.compute.affinity_matrix` for more details.",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    K = [int(k) for k in args.k.split(",")]
    if args.mu is not None:
        MU = [float(mu) for mu in args.mu.split(",")]
    else:
        MU = [i / 10 for i in range(0, 11, 2)]

    metabric = get_data(METABRIC_PATH)
    omics_combinations = args.omics.split(",")
    id = time.strftime("%m%d%H%M%S", time.gmtime())

    for omics_types in omics_combinations:
        omics_names = omics_types.split("+")
        print(f"\n>>> {omics_names} >>>\n")
        save_dir = os.path.join(
            "results",
            f"SNF_{omics_types}",
            id,
        )
        os.makedirs(save_dir)

        results = [["K", "mu", "silhoutte_score"]]
        best_score = float("-inf")
        best_X = None

        for k in K:
            print(f"--- K = {k} ---")
            for mu in MU:
                print(f"--- mu = {mu} ---")

                score, X = compute_silhoutte(metabric, omics_names)
                results.append([k, mu, score])

                if score > best_score:
                    best_score = score
                    best_X = X

                print(f"Best Score {best_score}")

        np.savetxt(
            os.path.join(save_dir, "results.csv"), results, delimiter=",", fmt="% s"
        )

        np.savetxt(
            os.path.join(save_dir, f"SNF_{'_'.join(omics_names)}.csv"),
            best_X,
            delimiter=",",
        )
