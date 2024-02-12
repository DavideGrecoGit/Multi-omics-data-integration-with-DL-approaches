import torch

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_FOLDS = 5
FOLD_DIR = "./data/5-fold_pam50stratified/"
FILE_NAME = "MBdata_33CLINwMiss_1KfGE_1KfCNA"

METABRIC_PATH = "./data/MBdata_33CLINwMiss_1KfGE_1KfCNA.csv"
REMOVE_UNKNOWN = True
