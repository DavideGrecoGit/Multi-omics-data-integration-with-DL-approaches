import argparse
import time
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from networks import Params_VAE, VAE, CNC_VAE, H_VAE
from data import Omics, get_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_FOLDS = 5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-omics",
        help="Type of integration CLI+RNA, CNA+RNA, CLI+CNA or CLI+CNA+RNA",
        type=str,
        default="CLI+RNA",
    )

    parser.add_argument(
        "-ds", help="The intermediate dense layers size", type=int, default=256
    )

    parser.add_argument("-ls", help="The latent layer size", type=int, default=64)
    parser.add_argument("-e", help="Number of epochs", type=int, default=150)
    parser.add_argument("-b", help="Beta value", type=int, default=50)
    parser.add_argument("-r", help="Regularisation function", type=str, default="mmd")
    parser.add_argument("-w_d", help="Weight decay", type=float, default=0.00001)
    parser.add_argument("-d_p", help="Dropout probability", type=float, default=0.2)
    parser.add_argument(
        "-remove_unknown",
        help="Remove samples with unkown Ground Truth class",
        type=bool,
        default=True,
    )

    parser.add_argument("-m", help="Model name", type=str, required=True)

    args = parser.parse_args()

    metabric_path = "./data/MBdata_33CLINwMiss_1KfGE_1KfCNA.csv"
    fold_dir = "./data/5-fold_pam50stratified/"
    file_name = "MBdata_33CLINwMiss_1KfGE_1KfCNA"
    id = time.strftime("%m%d%H%M%S", time.gmtime())

    omics_combinations = args.omics.split(",")

    for omics_types in omics_combinations:
        omics_names = omics_types.split("+")
        print(f"\n>>> {omics_names} >>>\n")
        save_dir = os.path.join(
            "results",
            f"{args.m}_{omics_types}",
            id,
        )
        os.makedirs(save_dir)

        vae_params = Params_VAE(
            None,
            args.ds,
            args.ls,
            epochs=args.e,
            beta=args.b,
            regularisation=args.r,
            weight_decay=args.w_d,
            d_p=args.d_p,
            remove_unknown=args.remove_unknown,
        )

        acc_scores = []

        for k in range(1, N_FOLDS + 1):
            print(f"=== FOLD {k} ===")

            # Get pre-processed data
            train_omics = get_data(
                os.path.join(fold_dir, f"fold{k}", file_name + "_train.csv"),
                metabric_path,
                args.remove_unknown,
            )
            train_omics = Omics(train_omics, omics_names)
            train_dataloader = DataLoader(
                train_omics, batch_size=vae_params.batch_size, shuffle=False
            )

            test_omics = get_data(
                os.path.join(fold_dir, f"fold{k}", file_name + "_test.csv"),
                metabric_path,
                args.remove_unknown,
            )
            test_omics = Omics(test_omics, omics_names)
            test_dataloader = DataLoader(
                test_omics, batch_size=vae_params.batch_size, shuffle=False
            )

            match args.m:
                case "CNC-VAE":
                    vae_params.input_dim = train_omics.get_input_dims()
                    if omics_types == "CLI+CNA":
                        vae_params.loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
                    model = CNC_VAE(vae_params)

                case "H-VAE":
                    vae_params.input_dim = args.ds // 2 * len(omics_names)
                    omics_dims = [
                        train_omics.get_input_dims(name) for name in omics_names
                    ]
                    model = H_VAE(vae_params, omics_names, omics_dims)

                case "VAE":
                    vae_params.input_dim = train_omics.get_input_dims(omics_types)
                    if omics_types == "CLI" or omics_types == "CNA":
                        vae_params.loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
                    model = VAE(vae_params, omics_index=0)

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=vae_params.lr,
                weight_decay=vae_params.weight_decay,
            )
            model.to(DEVICE)
            model.train_loop(
                train_dataloader, test_dataloader, optimizer, vae_params.epochs
            )

            save_path = os.path.join(save_dir, f"fold_{k}")
            os.makedirs(save_path)

            torch.save(model.state_dict(), os.path.join(save_path, f"{args.m}.pth"))

            # Embeddings
            train_save_path = os.path.join(save_path, "train_latent.csv")
            test_save_path = os.path.join(save_path, "test_latent.csv")

            train_embed = model.get_latent_space(train_dataloader, train_save_path)
            test_embed = model.get_latent_space(test_dataloader, test_save_path)

        vae_params.save_parameters(save_dir)
