import os
import numpy as np
import optuna
from optuna.samplers import TPESampler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data import Omics, get_data
from GATO.networks.VAEs import VAE, Params_VAE
from classifiers import Benchmark_Classifier
import argparse

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
METABRIC_PATH = "./data/MBdata_33CLINwMiss_1KfGE_1KfCNA.csv"
FOLD_DIR = "./data/5-fold_pam50stratified/"
FILE_NAME = "MBdata_33CLINwMiss_1KfGE_1KfCNA"
N_FOLDS = 5


def get_activation_fn(trial, activation_functions=None):
    if not activation_functions:
        activation_functions = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "elu": nn.ELU(),
            "leakyrelu": nn.LeakyReLU(),
        }

    activation = trial.suggest_categorical(
        "activation", list(activation_functions.keys())
    )
    return activation_functions[activation]


def define_hyperparams(trial, omics_type):
    # bs = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    bs = 64
    # lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    lr = trial.suggest_categorical("lr", [0.01, 0.005, 0.001, 0.0005])
    wd = trial.suggest_categorical("weight_decay", [0.0001, 0.00005, 0.00001])

    d_p = trial.suggest_int("d_p", 0, 8)
    d_p = d_p / 10

    beta = trial.suggest_categorical("beta", [0, 0.5, 1, 15, 50])
    activation_fn = get_activation_fn(trial)

    if omics_type == "CLI":
        # ds = trial.suggest_categorical("ds", np.arange(32, 200, 32).tolist())
        ds = trial.suggest_categorical("ds", [64, 128, 192])
        ls = trial.suggest_categorical("ls", [16, 32, 64])
    else:
        # ds = trial.suggest_categorical("ds", np.arange(64, 900, 64).tolist())
        ds = trial.suggest_categorical("ds", [128, 256, 512, 768])
        ls = trial.suggest_categorical("ls", [32, 64, 128])

    if omics_type == "CLI" or omics_type == "CNA":
        loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
    else:
        loss_fn = nn.MSELoss(reduction="mean")

    return bs, lr, wd, d_p, beta, activation_fn, ds, ls, loss_fn


def objective(trial):
    try:
        bs, lr, wd, d_p, beta, activation_fn, ds, ls, loss_fn = define_hyperparams(
            trial, omics_type
        )

        params = Params_VAE(
            None, ds, ls, lr, wd, bs, epochs, loss_fn, d_p, activation_fn, beta
        )

        f1_scores = []

        for k in range(1, N_FOLDS + 1):
            # Get pre-processed data
            train_omics = get_data(
                os.path.join(FOLD_DIR, f"fold{k}", FILE_NAME + "_train.csv"),
                METABRIC_PATH,
            )
            train_omics = Omics(train_omics, [omics_type])
            train_dataloader = DataLoader(
                train_omics, batch_size=params.batch_size, shuffle=False
            )

            test_omics = get_data(
                os.path.join(FOLD_DIR, f"fold{k}", FILE_NAME + "_test.csv"),
                METABRIC_PATH,
            )
            test_omics = Omics(test_omics, [omics_type])
            test_dataloader = DataLoader(
                test_omics, batch_size=params.batch_size, shuffle=False
            )

            # Train VAE
            params.input_dim = train_omics.get_input_dims(omics_type)
            model = VAE(params, omics_index=0)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=params.lr,
                weight_decay=params.weight_decay,
            )
            model.to(DEVICE)
            model.train_loop(
                train_dataloader, test_dataloader, optimizer, params.epochs
            )

            # Embeddings
            train_embed = model.get_latent_space(train_dataloader)
            test_embed = model.get_latent_space(test_dataloader)

            train_gt = train_omics.pam50
            test_gt = test_omics.pam50

            classifier = Benchmark_Classifier(NB=False, RF=False)
            accTrain, f1Train = classifier.train(train_embed, train_gt)
            accTest, f1Test = classifier.evaluate(test_embed, test_gt)

            f1 = np.mean(f1Test)

            print(f"Fold: {k}, F1 score: {f1}")

            f1_scores.append(f1)

            trial.report(f1, k - 1)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return np.mean(np.array(f1_scores))

    except:
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-omics",
        help="Type of integration CLI, CNA or RNA",
        type=str,
        default="RNA",
    )
    parser.add_argument(
        "-study_name",
        help="Name of the optuna study to run",
        type=str,
        required=True,
    )
    parser.add_argument("-epochs", help="Number of epochs", type=int, default=20)
    parser.add_argument(
        "-n_trials", help="Number of optuna trials to run", type=int, default=100
    )
    parser.add_argument(
        "-timeout", help="Max seach running time (in minutes)", type=int, default=30
    )

    args = parser.parse_args()
    omics_type = args.omics
    epochs = args.epochs
    n_trials = args.n_trials
    timeout = args.timeout * 60  # in seconds

    study_name = args.study_name
    storage_path = f"sqlite:///{omics_type}_{study_name}.db"

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage_path,
        sampler=TPESampler(seed=SEED),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    pruned_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    ]
    complete_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    print("Study statistics: ")
    print("\tN finished trials: ", len(study.trials))
    print("\tN pruned trials: ", len(pruned_trials))
    print("\tN completed trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial
    print("\tMean F1 score: ", trial.value)

    print("\tParams: ")
    for key, value in trial.params.items():
        print(f"\t\t{key}: {value}")

    # fig = optuna.visualization.plot_optimization_history(study)
    # fig.show()
