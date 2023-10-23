import random
import numpy as np
import torch
import pandas as pd
import os
import argparse


def make_path(new_path):
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    return new_path


def read_data(path):
    """
    @source https://github.com/lifoof/mogcn
    """

    df = pd.read_csv(path, header=0, index_col=None)
    df.rename(columns={df.columns.tolist()[0]: "Sample"}, inplace=True)
    df.sort_values(by="Sample", ascending=True, inplace=True)

    return df


def filter_MoGCN_data(data, allowed_samples_path):
    allowed_samples = read_data(allowed_samples_path)
    return data.loc[data["Sample"].isin(allowed_samples["Sample"])].reset_index(
        drop=True
    )


def read_MoGCN_data(
    omics_paths=["data/fpkm_data.csv", "data/gistic_data.csv", "data/rppa_data.csv"],
    gt_data_path="data/sample_classes.csv",
):
    """
    Read a list of csv path of omics data
    and assert they all have the same sample list.

    Parameters:
        paths -- list of paths to omics-data csv files
    """

    # read data
    omics_data = [read_data(path) for path in omics_paths]

    # Test that all omics dat ahave the SAME Samples IDs
    for omics in omics_data:
        assert omics["Sample"].equals(
            omics_data[0]["Sample"]
        ), "Sample IDs are not consistent between different omics data"

    gt_data = read_data(gt_data_path)

    samples_list = omics_data[0]["Sample"].tolist()
    classes_list = gt_data["class"].tolist()

    return omics_data, gt_data, samples_list, classes_list


# def train_test_split(gt_data, percentage=0.2):
#     samples = gt_data.index.groupby('class', group_keys=False).apply(lambda x: x.sample(frac=percentage))

#     omics_data.loc[omics_data['Sample'] == some_value]

#     test_omics_data = [omics.iloc[test_index] for omics in omics_data]
#     test_labels = gt_labels.iloc[test_index]

#     return samples

# [-1, 1]
# self.omics_data = [
#     -1 + (2 * ((omics - omics.min()) / (omics.max() - omics.min())))
#     for omics in self.omics_data
# ]

# [0, 1]
# self.omics_data[0] = (self.omics_data[0] - self.omics_data[0].min()) / (
#     self.omics_data[0].max() - self.omics_data[0].min()
# )
# self.omics_data[2] = (self.omics_data[2] - self.omics_data[2].min()) / (
#     self.omics_data[2].max() - self.omics_data[2].min()
# )

# Standardisation

# self.omics_data = [
#     (omics - torch.mean(omics)) / torch.std(omics) for omics in self.omics_data
# ]

# self.omics_data[0] = (
#     self.omics_data[0] - torch.mean(self.omics_data[0])
# ) / torch.std(self.omics_data[0])
# self.omics_data[2] = (
#     self.omics_data[2] - torch.mean(self.omics_data[2])
# ) / torch.std(self.omics_data[2])

# for omics in self.omics_data:
#     print(omics[1])

# self.omics_data[1][self.omics_data[1] == -1] = 2.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", "-p", type=str, required=True, help="Path to data files"
    )
    args = parser.parse_args()

    data_path = args.path
    omics_file_names = ["fpkm_data.csv", "gistic_data.csv", "rppa_data.csv"]
    gt_file_name = "sample_classes.csv"

    omics_data, gt_data, samples_list, classes_list = read_MoGCN_data(
        omics_paths=[os.path.join(data_path, file) for file in omics_file_names],
        gt_data_path=os.path.join(data_path, gt_file_name),
    )

    for omics in omics_data:
        print(omics.head(5))

    print(gt_data.head(5))
