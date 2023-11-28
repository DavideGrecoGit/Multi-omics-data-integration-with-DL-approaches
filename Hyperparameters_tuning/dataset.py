import torch
from torch.utils.data import Dataset
from utils.snf import get_laplace
from utils.data import read_data
import numpy as np


class MoGCN_Dataset(Dataset):
    """
    Dataset class for multi-omics dataset.
    Each omics file has samples as rows and features as columns.

    Parameters:
        path1 -- path to RNA-seq csv
        path2 -- path to CNV csv
        path3 -- path to RPPA csv
        path_gt_samples -- path to GT labels csv
        path_list_samples -- path to list of train/test samples csv
    """

    def __init__(self, df_omics_data, df_gt_data, latent_space=None, adj_matrix=None):
        self.samples_list = df_gt_data["Sample"].tolist()
        self.gt_labels = df_gt_data["PAM50Call_RNAseq"]
        # Note: -1 is required to remove the 'Sample' column from the count
        self.input_dims = [omics.shape[1] - 1 for omics in df_omics_data]

        # Convert omics_data to Tensor
        self.omics_data = [
            torch.tensor(
                omics.loc[:, omics.columns != "Sample"].values, dtype=torch.float
            )
            for omics in df_omics_data
        ]

        # Convert gt classes to Tensor
        self.gt_classes = torch.tensor(
            df_gt_data.loc[:, df_gt_data.columns == "class"].values,
            dtype=torch.long,
        )[:, 0]

        # Convert adj to Tensor
        self.adj_matrix = adj_matrix

        if self.adj_matrix:
            self.adj_matrix = torch.tensor(self.adj_matrix, dtype=torch.float)

        # Convert Laplace to Tensor
        self.latent_space = latent_space
        if self.latent_space:
            self.latent_space = torch.tensor(
                self.latent_space.loc[:, self.latent_space.columns != "Sample"].values,
                dtype=torch.float,
            )

    def set_latent_space(self, path=None):
        if isinstance(path, str):
            df = read_data(path)
            self.latent_space = torch.tensor(
                df.loc[:, df.columns != "Sample"].values,
                dtype=torch.float,
            )
        if isinstance(path, np.ndarray):
            self.latent_space = torch.tensor(
                path,
                dtype=torch.float,
            )

        if torch.is_tensor(path):
            self.latent_space = path

        if self.latent_space is None:
            print("Latent space is null")

        return self.latent_space

    def set_adj_matrix(self, path=None):
        if isinstance(path, str):
            df = get_laplace(path)
            self.adj_matrix = torch.tensor(df, dtype=torch.float)

        if isinstance(path, np.ndarray):
            self.adj_matrix = torch.tensor(
                path,
                dtype=torch.float,
            )

        if torch.is_tensor(path):
            self.adj_matrix = path

        if self.adj_matrix is None:
            print("Adj Matrix is null")

        return self.adj_matrix

    def __len__(self):
        return self.gt_classes.size()[0]

    def __getitem__(self, idx):
        items = [omics[idx, :] for omics in self.omics_data]

        label = self.gt_classes[idx]

        return items, label


class Omics_Dataset(Dataset):
    """
    Dataset class for multi-omics dataset.
    Each omics file has samples as rows and features as columns.

    Parameters:
        path1 -- path to RNA-seq csv
        path2 -- path to CNV csv
        path3 -- path to RPPA csv
        path_gt_samples -- path to GT labels csv
        path_list_samples -- path to list of train/test samples csv
    """

    def __init__(self, df_omics_data, df_gt_data, latent_space=None):
        self.samples_list = df_gt_data["Sample"].tolist()
        self.gt_labels = df_gt_data["PAM50Call_RNAseq"]
        # Note: -1 is required to remove the 'Sample' column from the count
        self.input_dims = df_omics_data.shape[1] - 1

        # Convert omics_data to Tensor
        self.omics_data = torch.tensor(
            df_omics_data.loc[:, df_omics_data.columns != "Sample"].values,
            dtype=torch.float,
        )

        # Convert gt classes to Tensor
        self.gt_classes = torch.tensor(
            df_gt_data.loc[:, df_gt_data.columns == "class"].values,
            dtype=torch.long,
        )[:, 0]

        # Convert Laplace to Tensor
        self.latent_space = latent_space

        if self.latent_space:
            self.latent_space = torch.tensor(
                self.latent_space.loc[:, self.latent_space.columns != "Sample"].values,
                dtype=torch.float,
            )

    def set_latent_space(self, latent_tensor):
        if torch.is_tensor(latent_tensor):
            self.latent_space = latent_tensor

        if self.latent_space is None:
            print("Latent space is null")

        return self.latent_space

    def __len__(self):
        return self.gt_classes.size()[0]

    def __getitem__(self, idx):
        return self.omics_data[idx, :], self.gt_classes[idx]
