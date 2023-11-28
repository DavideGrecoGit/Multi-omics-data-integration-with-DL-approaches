import snf
import seaborn as sns
import numpy as np
from sklearn.cluster import spectral_clustering
from sklearn.metrics import v_measure_score
import pandas as pd
import os
from utils.data import read_data


def run_snf(omics_data, output_path=None, k=20, mu=0.5, metric="sqeuclidean"):
    """
    k - (0, N) int, number of neighbors to consider when creating affinity matrix.
    See Notes of :py:func snf.compute.affinity_matrix for more details.
    Default: 20.

    m - (0, 1) float, Normalization factor to scale similarity kernel when constructing affinity matrix.
    See Notes of :py:func snf.compute.affinity_matrix for more details.
    Default: 0.5.

    metric - Distance metric to compute. Must be one of available metrics
    in :py:func scipy.spatial.distance.pdist.
    """

    affinity_nets = snf.make_affinity(
        [
            omics_data[0].iloc[:, 1:].values.astype(np.float64),
            omics_data[1].iloc[:, 1:].values.astype(np.float64),
            omics_data[2].iloc[:, 1:].values.astype(np.float64),
        ],
        metric=metric,
        K=k,
        mu=mu,
    )

    fused_net = snf.snf(affinity_nets, K=k)

    if output_path:
        fused_df = pd.DataFrame(fused_net)
        fused_df.columns = omics_data[0]["Sample"].tolist()
        fused_df.index = omics_data[0]["Sample"].tolist()

        fused_df.to_csv(
            os.path.join(output_path, "SNF_fused_matrix.csv"), header=True, index=True
        )

        np.fill_diagonal(fused_df.values, 0)
        fig = sns.clustermap(
            fused_df.iloc[:, :],
            cmap="vlag",
            figsize=(8, 8),
        )
        fig.savefig(os.path.join(output_path, "SNF_fused_matrix.png"), dpi=300)

    return fused_net


def evaluate_snf(fused_net, gt_classes, n_clusters=4):
    labels = spectral_clustering(fused_net, n_clusters=n_clusters)
    return v_measure_score(labels, gt_classes)


def get_laplace(adj_m, threshold=0.005):
    if isinstance(adj_m, str):
        adj_df = read_data(adj_m)
        adj_m = adj_df.iloc[:, 1:].values

    # The SNF matrix is a completed connected graph, it is better to filter edges with a threshold
    adj_m[adj_m < threshold] = 0

    # adjacency matrix after filtering
    exist = (adj_m != 0) * 1.0
    # np.savetxt('result/adjacency_matrix.csv', exist, delimiter=',', fmt='%d')

    # calculate the degree matrix
    factor = np.ones(adj_m.shape[1])
    res = np.dot(exist, factor)  # degree of each node
    diag_matrix = np.diag(res)  # degree matrix
    # np.savetxt('result/diag.csv', diag_matrix, delimiter=',', fmt='%d')

    # calculate the laplace matrix
    d_inv = np.linalg.inv(diag_matrix)
    adj_hat = d_inv.dot(exist)

    return adj_hat
