import random
import numpy as np
import torch
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


def setup_seed(seed):
    """
    setup seed to make the experiments deterministic

    Parameters:
        seed(int) -- the random seed

    @source https://github.com/zhangxiaoyu11/OmiEmbed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


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
