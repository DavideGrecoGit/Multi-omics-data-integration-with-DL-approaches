from utils import read_MoGCN_data, filter_MoGCN_data
import torch
from torch.utils.data import Dataset


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

    def __init__(
        self,
        omics_data,
        gt_labels,
        path_list_samples,
    ):
        # Read and filter omics data
        self.omics_data = [
            filter_MoGCN_data(omics, path_list_samples) for omics in omics_data
        ]

        # Store features dimensions of each omics.
        # Note: -1 is required to remove the 'Sample' column from the count
        self.input_dims = [omics.shape[1] - 1 for omics in self.omics_data]

        # Convert omics_data to Tensor
        self.omics_data = [
            torch.tensor(
                omics.loc[:, omics.columns != "Sample"].values, dtype=torch.float
            )
            for omics in omics_data
        ]

        # Read and filter GT data
        self.gt_labels = filter_MoGCN_data(gt_labels, path_list_samples)
        # Convert gt_labels to Tensor
        self.labels = torch.tensor(
            self.gt_labels.loc[:, self.gt_labels.columns == "class"].values,
            dtype=torch.float,
        )

    def __len__(self):
        return self.labels.size()[0]

    def __getitem__(self, idx):
        items = [omics[idx, :] for omics in self.omics_data]

        label = self.labels[idx, :]

        return items, label
