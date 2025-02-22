import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import Dataset
import torch

SEED = 42
transform = {"RNA": "rnanp", "CNA": "cnanp", "CLI": "clin"}


class Omics(Dataset):
    def __init__(self, fold_path, metabric_path, omics_names=["CNA", "RNA", "CLI"]):
        # Get pre-processed data
        omics = get_data(fold_path, metabric_path)

        self.omics_values = {}
        for name in omics_names:
            if name == "RNA":
                self.omics_values[name] = torch.tensor(
                    normalizeRNA(omics[transform[name]]), dtype=torch.float32
                )
            else:
                self.omics_values[name] = torch.tensor(
                    omics[transform[name]], dtype=torch.float32
                )

        self.omics_names = omics_names
        self.pam50 = torch.tensor(omics["pam50np"], dtype=torch.int)
        self.pam50_labels = omics["pam50"]

    def get_omics_data(self):
        return [self.omics_values[name] for name in self.omics_names]

    def get_input_dims(self, name=None):
        if name is None:
            dims = 0
            for name in self.omics_names:
                dims += self.omics_values[name].size()[1]
            return dims

        return self.omics_values[name].size()[1]

    def __len__(self):
        return len(self.pam50)

    def __getitem__(self, idx):
        return [self.omics_values[name][idx] for name in self.omics_names]


def plot_latent_space(h, y, save_path=None):
    z = TSNE(n_components=2, random_state=SEED).fit_transform(h)

    sns_plot = sns.scatterplot(
        x=z[:, 0], y=z[:, 1], hue=y, palette=sns.color_palette("bright")
    )
    if save_path is not None:
        plt.savefig(save_path)
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


def get_data(metabric_path, complete_metabric_path):
    """
    Source https://github.com/CancerAI-CL/IntegrativeVAEs.git
    """

    data = pd.read_csv(metabric_path, index_col=None, header=0, low_memory=False)
    # Remove unknown classes
    # data = data.drop(data[data["Pam50Subtype"] == "?"].index)

    d = {}
    clin_fold = data[["METABRIC_ID"]]

    rna = data[[col for col in data if col.startswith("GE")]]
    cna = data[[col for col in data if col.startswith("CNA")]]

    d["METABRIC_ID"] = data["METABRIC_ID"]
    d["ic"] = list(data["iC10"].values)
    d["pam50"] = list(data["Pam50Subtype"].values)
    d["er"] = list(data["ER_Expr"].values)
    d["pr"] = list(data["PR_Expr"].values)
    d["her2"] = list(data["Her2_Expr"].values)
    d["drnp"] = list(data["DR"].values)

    d["rnanp"] = rna.astype(np.float32).values
    # d["rnanp"] = normalizeRNA(d["rnanp"])
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

    # We have to get the entire dataset, transform them into one-hots, bins
    # complete_data = pd.read_csv(complete_data).set_index("METABRIC_ID")
    complete_data = pd.read_csv(
        complete_metabric_path, index_col=None, header=0, low_memory=False
    )

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
            size,
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
