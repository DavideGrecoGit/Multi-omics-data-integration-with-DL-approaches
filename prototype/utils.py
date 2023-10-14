import os
import shutil
import torch
import random
import numpy as np
import pandas as pd


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


def read_MoGCN_data(paths):
    """
    Read a list of csv path of omics data
    and assert they all have the same sample list.

    Parameters:
        paths -- list of paths to omics-data csv files
    """

    # read data
    omics_data = [read_data(path) for path in paths]

    # Test that all omics dat ahave the SAME Samples IDs
    for omics in omics_data:
        assert omics["Sample"].equals(
            omics_data[0]["Sample"]
        ), "Sample IDs are not consistent between different omics data"

    return omics_data
