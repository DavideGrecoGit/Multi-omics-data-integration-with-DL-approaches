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
