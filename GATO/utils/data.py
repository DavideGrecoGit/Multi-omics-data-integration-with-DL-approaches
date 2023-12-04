import pandas as pd
import os
import argparse
import snf
import numpy as np
from torch_geometric.utils import to_edge_index
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split


def plot_latent_space(h, y):
    z = TSNE(n_components=2).fit_transform(h)

    sns.scatterplot(x=z[:, 0], y=z[:, 1], hue=y, palette=sns.color_palette("tab5"))
    plt.show()


def plot_confusion_matrix(y_true, pred, labels, normalize="true"):
    cm = confusion_matrix(y_true, pred, normalize=normalize)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    cm_display.plot()
    plt.show()


def get_pam50_labels(data):
    val_to_cat = {}
    cat = []
    index = 0
    for val in data:
        if val not in val_to_cat:
            val_to_cat[val] = index
            cat.append(val)
            index += 1
    return cat


def get_edge_index(omics, threshold=0.002, metric="cosine", sample_list=None):
    aff = snf.make_affinity(omics, metric=metric)

    fused_net = snf.snf(aff)

    fused_net[fused_net >= threshold] = 1
    fused_net[fused_net < threshold] = 0

    return to_edge_index((torch.tensor(fused_net, dtype=torch.long).to_sparse()))[0]


def array_index_to_bool(idx, n):
    mask = np.zeros(n, dtype=int)
    mask[idx] = 1
    return mask.astype(bool)


def save_csv_and_check(mask, output_path, filename):
    np.savetxt(
        os.path.join(output_path, f"{filename}.csv"),
        mask,
        delimiter=",",
        fmt="%s",
    )

    mask_check = np.genfromtxt(
        os.path.join(output_path, f"{filename}.csv"), delimiter=",", dtype=bool
    )

    assert (mask == mask_check).all(), "File incorrectly saved"


def to_categorical(data, dtype=None):
    """
    Source https://github.com/CancerAI-CL/IntegrativeVAEs.git
    """
    val_to_cat = {}
    cat = []
    index = 0
    for val in data:
        if dtype == "ic":
            if val not in [
                "1",
                "2",
                "3",
                "4ER+",
                "4ER-",
                "5",
                "6",
                "7",
                "8",
                "9",
                "10",
            ]:
                val = "1"
            if val in ["4ER+", "4ER-"]:
                val = "4"
        if val not in val_to_cat:
            val_to_cat[val] = index
            cat.append(index)
            index += 1
        else:
            cat.append(val_to_cat[val])
    return np.array(cat)


def normalizeRNA(*args):
    """
    Source https://github.com/CancerAI-CL/IntegrativeVAEs.git
    """
    if len(args) > 1:
        normalizeData = np.concatenate((args[0], args[1]), axis=0)
        normalizeData = (normalizeData - normalizeData.min(axis=0)) / (
            normalizeData.max(axis=0) - normalizeData.min(0)
        )
        return normalizeData[: args[0].shape[0]], normalizeData[args[0].shape[0] :]
    else:
        return (args[0] - args[0].min(axis=0)) / (args[0].max(axis=0) - args[0].min(0))


def get_data(data, complete_data):
    """
    Source https://github.com/CancerAI-CL/IntegrativeVAEs.git
    """
    d = {}
    clin_fold = data[["METABRIC_ID"]]

    rna = data[[col for col in data if col.startswith("GE")]]
    cna = data[[col for col in data if col.startswith("CNA")]]

    d["ic"] = list(data["iC10"].values)
    d["pam50"] = list(data["Pam50Subtype"].values)
    d["er"] = list(data["ER_Expr"].values)
    d["pr"] = list(data["PR_Expr"].values)
    d["her2"] = list(data["Her2_Expr"].values)
    d["drnp"] = list(data["DR"].values)

    d["rnanp"] = rna.astype(np.float32).values
    d["rnanp"] = normalizeRNA(d["rnanp"])
    d["cnanp"] = (cna.astype(np.float32).values + 2.0) / 4.0
    d["icnp"] = to_categorical(d["ic"], dtype="ic")
    d["pam50np"] = to_categorical(d["pam50"])
    d["ernp"] = to_categorical(d["er"])
    d["prnp"] = to_categorical(d["pr"])
    d["her2np"] = to_categorical(d["her2"])
    d["drnp"] = to_categorical(d["drnp"])

    """
    preprocessing for clinical data to match current pipeline
    """
    ## Clinical Data Quick Descriptions
    # clin["Age_At_Diagnosis"]           # Truly numeric
    # clin["Breast_Tumour_Laterality"]   # Categorical "L, R" (3 unique)
    # clin["NPI"]                        # Truly numeric
    # clin["Inferred_Menopausal_State"]  # Categorical "Pre, Post" (3 unique)
    # clin["Lymph_Nodes_Positive"]       # Ordinal ints 0-24
    # clin["Grade"]                      # Ordinal string (come on) 1-3 + "?"
    # clin["Size"]                       # Truly Numeric
    # clin["Histological_Type"]          # Categorical strings (9 unique)
    # clin["Cellularity"]                # Categorical strings (4 unique)
    # clin["Breast_Surgery"]             # Categorical strings (3 Unique)
    # clin["CT"]                         # Categorical strings (9 unique)
    # clin["HT"]                         # Categorical strings (9 Unique)
    # clin["RT"]                         # Categorical strings (9 Unique)

    ## Clinical Data Transformations
    # On the basis of the above we will keep some as numeric and others into one-hot encodings
    # (I am not comfortable binning the continuous numeric columns without some basis for their bins)
    # Or since we dont have that much anyway just one hot everything and use BCE Loss to train

    # Either we keep numerics as
    clin_numeric = complete_data[["METABRIC_ID", "Age_At_Diagnosis", "NPI", "Size"]]

    # Numerical binned to arbitrary ranges then one-hot dummies
    metabric_id = complete_data[["METABRIC_ID"]]
    aad = pd.get_dummies(
        pd.cut(complete_data["NPI"], 10, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        prefix="aad",
        dummy_na=True,
    )
    npi = pd.get_dummies(
        pd.cut(complete_data["NPI"], 6, labels=[1, 2, 3, 4, 5, 6]),
        prefix="npi",
        dummy_na=True,
    )
    size = pd.get_dummies(complete_data["Size"], prefix="size", dummy_na=True)

    # Categorical and ordinals to one-hot dummies
    btl = pd.get_dummies(
        complete_data["Breast_Tumour_Laterality"], prefix="btl", dummy_na=True
    )
    ims = pd.get_dummies(
        complete_data["Inferred_Menopausal_State"], prefix="ims", dummy_na=True
    )
    lnp = pd.get_dummies(
        complete_data["Lymph_Nodes_Positive"], prefix="lnp", dummy_na=True
    )
    grade = pd.get_dummies(complete_data["Grade"], prefix="grade", dummy_na=True)
    hist = pd.get_dummies(
        complete_data["Histological_Type"], prefix="hist", dummy_na=True
    )
    cellularity = pd.get_dummies(
        complete_data["Cellularity"], prefix="cellularity", dummy_na=True
    )
    ct = pd.get_dummies(complete_data["CT"], prefix="ct", dummy_na=True)
    ht = pd.get_dummies(complete_data["HT"], prefix="ht", dummy_na=True)
    rt = pd.get_dummies(complete_data["RT"], prefix="rt", dummy_na=True)

    clin_transformed = pd.concat(
        [clin_numeric, btl, ims, lnp, grade, size, hist, cellularity, ct, ht, rt],
        axis=1,
    )  # 222 columns
    clin_transformed = pd.concat(
        [
            metabric_id,
            aad,
            npi,
            btl,
            ims,
            lnp,
            grade,
            size,
            hist,
            cellularity,
            ct,
            ht,
            rt,
        ],
        axis=1,
    )  # 2278 columns non binned, 350 columns if binned

    # Now create the fold data by selecting from the complete transformed clinical data
    # print(list(clin_fold.flatten()))
    fold_ids = [x.item() for x in list(clin_fold.values)]
    clin_transformed = clin_transformed.loc[
        clin_transformed["METABRIC_ID"].isin(fold_ids)
    ]
    del clin_transformed["METABRIC_ID"]

    d["clin"] = clin_transformed.astype(np.float32).values
    return d


def make_path(new_path):
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    return new_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_path",
        "-o",
        type=str,
        help="Output path to save directory",
        default="GATO/data/",
    )
    parser.add_argument(
        "--metabric_path",
        "-mt",
        type=str,
        help="Path to METABRIC csv file",
        default="GATO/data/METABRIC_CLIN_GE_CNA.csv",
    )
    parser.add_argument(
        "--test_split", "-ts", type=str, help="Split test size", default=0.2
    )
    parser.add_argument(
        "--val_split", "-vs", type=str, help="Split val size", default=0.2
    )
    args = parser.parse_args()

    # Load METABRIC
    complete_data = pd.read_csv(
        args.metabric_path, index_col=None, header=0, low_memory=False
    )
    # Remove unknown classes
    complete_data = complete_data.drop(
        complete_data[complete_data["Pam50Subtype"] == "?"].index
    )
    # Get pre-processed data
    omics = get_data(complete_data, complete_data)

    y = omics["pam50np"]

    idx = np.arange(0, len(y), 1, dtype=int)
    idx_train, idx_test = train_test_split(
        idx, test_size=args.test_split, random_state=42, stratify=y
    )

    mask_test = array_index_to_bool(idx_test, len(y))

    idx_train, idx_val = train_test_split(
        idx_train,
        test_size=args.val_split,
        random_state=42,
        stratify=y[idx_train],
    )

    mask_train = array_index_to_bool(idx_train, len(y))
    mask_val = array_index_to_bool(idx_val, len(y))

    save_csv_and_check(mask_train, args.output_path, "mask_train")
    save_csv_and_check(mask_val, args.output_path, "mask_val")
    save_csv_and_check(mask_test, args.output_path, "mask_test")

    temp = np.logical_xor(mask_val, mask_train)
    assert (
        np.logical_xor(temp, mask_test).all() == True
    ), "Train, val and test masks are not disjoint"
