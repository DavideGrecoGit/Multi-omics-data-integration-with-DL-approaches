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
        all_omics_data,
        all_gt_labels,
        path_list_samples,
    ):
        # Read and filter omics data
        self.omics_data = [
            filter_MoGCN_data(omics, path_list_samples) for omics in all_omics_data
        ]

        self.samples_list = self.omics_data[0]["Sample"].tolist()

        # Store features dimensions of each omics.
        # Note: -1 is required to remove the 'Sample' column from the count
        self.input_dims = [omics.shape[1] - 1 for omics in self.omics_data]

        # Convert omics_data to Tensor
        self.omics_data = [
            torch.tensor(
                omics.loc[:, omics.columns != "Sample"].values, dtype=torch.float
            )
            for omics in self.omics_data
        ]

        # Read and filter GT data
        filtered_gt = filter_MoGCN_data(all_gt_labels, path_list_samples)

        # Convert gt_labels to Tensor
        self.gt_classes = torch.tensor(
            filtered_gt.loc[:, filtered_gt.columns == "class"].values,
            dtype=torch.float,
        )

        self.gt_labels = filtered_gt["PAM50Call_RNAseq"]

    def __len__(self):
        return self.gt_classes.size()[0]

    def __getitem__(self, idx):
        items = [omics[idx, :] for omics in self.omics_data]

        label = self.gt_classes[idx, :]

        return items, label
