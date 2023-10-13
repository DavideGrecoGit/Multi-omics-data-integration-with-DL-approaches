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
        path_1="data/fpkm_data.csv",  # RNA-seq
        path_2="data/gistic_data.csv",  # CNV
        path_3="data/rppa_data.csv",  # RPPA
        path_gt_samples="data/sample_classes.csv",  # GT labels
        path_list_samples="data/train_samples.csv",
    ):
        # Read and filter omics data
        self.omics_data = read_MoGCN_data([path_1, path_2, path_3])
        self.omics_data = [
            filter_MoGCN_data(omics, path_list_samples).reset_index(drop=True)
            for omics in self.omics_data
        ]
        # Store features dimensions of each omics
        self.input_dims = [omics.shape[1] for omics in self.omics_data]

        # Read and filter GT data
        self.gt_labels = read_MoGCN_data([path_gt_samples])[0]
        self.gt_labels = filter_MoGCN_data(
            self.gt_labels, path_list_samples
        ).reset_index(drop=True)

    def __len__(self):
        return self.gt_labels.shape[0]

    def __getitem__(self, idx):
        items = [
            torch.Tensor(omics.loc[idx, omics.columns != "Sample"])
            for omics in self.omics_data
        ]

        label = torch.Tensor(self.gt_labels.loc[idx, self.gt_labels.columns == "class"])

        return items, label
