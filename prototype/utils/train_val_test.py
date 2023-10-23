import copy
from dataset import MoGCN_Dataset
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
import pandas as pd
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold
import numpy as np
import random
from utils.data import make_path
import os

SEED = 42


def empty_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def get_torch_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(model, path="./checkpoints/"):
    # Save checkpoint
    output_path = make_path(path)
    torch.save(model, os.path.join(output_path, f"{model.name}.pkl"))


def setup_seed(seed=SEED):
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


class Early_Stopping:
    def __init__(self, tolerance=5):
        self.best_loss = 9999999
        self.best_epoch = 0
        self.best_model = None
        self.tolerance = tolerance

    def check(self, model, new_loss, epoch):
        if new_loss < self.best_loss:
            self.best_loss = new_loss
            self.best_epoch = epoch
            self.best_model = copy.deepcopy(model)

        elif epoch - self.best_epoch > self.tolerance:
            print(f"Early stopped training at epoch {epoch}")
            return True
        return False


def train_loop(
    model,
    optimizer,
    train_data,
    output_path=None,
    num_epochs=100,
    val_data=None,
    early_stopping_mode="train_loss",
    tolerance=5,
):
    train_loss_ls = []
    val_loss_ls = []
    train_loss = 0.0
    val_loss = 0

    early_stopping = Early_Stopping(tolerance)

    for epoch in range(num_epochs):
        model.train()

        # If the data passed is a dataloader, the model being trained is an encoder-decoder
        # which only needs the multi-omics data
        if isinstance(train_data, DataLoader):
            train_loss_sum = 0.0
            for batch_idx, (x_omics, _) in enumerate(train_data):
                output, loss = model.forward_pass(x_omics)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item()

            train_loss = train_loss_sum / len(train_data)

        # If the data is the dataset, the model being trained is a GNN,
        # which needs a latent space, adj_matrix and the gt classes
        if isinstance(train_data, MoGCN_Dataset):
            output, loss = model.forward_pass(
                train_data.latent_space, train_data.adj_matrix, train_data.gt_classes
            )

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss = loss.item()

        train_loss_ls.append(train_loss)

        if val_data:
            val_loss, acc, f1, cm = val_loop(model, val_data)
            val_loss_ls.append(val_loss)

        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(
                f"epoch: {epoch + 1} | train loss: {train_loss:.4f}",
                f" | val loss: {val_loss:.4f}" if val_data else "",
                f" | accuracy: {acc}" if acc else "",
                f" | f1 score: {f1:.4f}" if f1 else "",
            )

        match early_stopping_mode:
            case "train_loss":
                new_loss = train_loss
            case "val_loss":
                new_loss = val_loss

        if early_stopping.check(model, new_loss, epoch + 1):
            break

    print(
        f"\nBest epoch: {early_stopping.best_epoch} | {early_stopping_mode} {early_stopping.best_loss:.4f}"
    )

    if output_path:
        save_checkpoint(early_stopping.best_model, output_path)

    if val_data:
        return {"train": train_loss_ls, "val": val_loss_ls}, cm
    return {"train": train_loss_ls}


from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt


def val_loop(model, data):
    model.eval()

    val_loss = 0.0
    acc = None
    f1 = None
    cm = None

    with torch.no_grad():
        if isinstance(data, DataLoader):
            batch_loss_sum = 0.0
            for batch_idx, (x_omics, _) in enumerate(data):
                # Forward & Loss
                latent_data, loss = model.forward_pass(x_omics)

                batch_loss_sum += loss.item()
                val_loss = batch_loss_sum / len(data)

        if isinstance(data, MoGCN_Dataset):
            output, loss = model.forward_pass(
                data.latent_space, data.adj_matrix, data.gt_classes
            )

            output = output.detach().cpu().numpy()
            print(output)
            output = np.argmax(output, axis=1)
            print(output)

            acc = accuracy_score(data.gt_classes, output)
            f1 = f1_score(data.gt_classes, output, average="weighted")

            val_loss = loss.item()

            cm = confusion_matrix(data.gt_classes, output)

    return val_loss, acc, f1, cm


def train_test(
    omics_data,
    gt_labels,
    model_class,
    loss_fn=nn.MSELoss(),
    activation_fn=nn.Sigmoid(),
    dropout_p=0,
    num_epochs=100,
    batch_size=32,
    lr_rate=0.001,
    n_splits=5,
):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    sum_train_loss = 0.0
    sum_test_loss = 0.0
    sum_kl_divergence = 0.0

    for i, (train_index, test_index) in enumerate(
        skf.split(omics_data[0], gt_labels["class"])
    ):
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_omics_data = [omics.iloc[train_index] for omics in omics_data]
        train_labels = gt_labels.iloc[train_index]

        test_omics_data = [omics.iloc[test_index] for omics in omics_data]
        test_labels = gt_labels.iloc[test_index]

        # Train dataset
        MoGCN_train = MoGCN_Dataset(train_omics_data, train_labels)
        train_loader = DataLoader(MoGCN_train, batch_size=batch_size, shuffle=True)

        # Test dataset
        MoGCN_test = MoGCN_Dataset(test_omics_data, test_labels)
        test_loader = DataLoader(MoGCN_test, batch_size=batch_size, shuffle=True)

        # Model
        model = model_class(
            MoGCN_train.input_dims, activation_fn=activation_fn, dropout_p=dropout_p
        )
        optimizer = optim.Adam(model.parameters(), lr=lr_rate)

        # Train
        model.to(DEVICE)
        model.train()
        train_loss = train_loop(model, train_loader, optimizer, loss_fn, num_epochs)

        # Test
        model.eval()  # before save and test, fix the variables
        test_loss = test_loop(model, test_loader, loss_fn)

        # # Save model
        # torch.save(model, f"model/AE/{model.name}_model.pkl")

        # Save and Plot latent space
        kl_divergence = save_latent_data(MoGCN_train, model)

        sum_train_loss += train_loss
        sum_test_loss += test_loss
        sum_kl_divergence += kl_divergence

    return (
        sum_train_loss / n_splits,
        sum_test_loss / n_splits,
        sum_kl_divergence / n_splits,
    )


def save_latent_data(dataset, model, output_path, perplexity=40):
    output_path = make_path(output_path)
    latent_space = model.get_latent_space(dataset)

    # Save latent space
    latent_df = pd.DataFrame(latent_space)
    latent_df.insert(0, "Sample", dataset.samples_list)
    output_path_csv = os.path.join(output_path, f"{model.name}_latent_space.csv")
    latent_df.to_csv(output_path_csv, header=True, index=False)

    # Plot latent space
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_train_tsne = tsne.fit_transform(latent_space)

    print("T-SNE KL Divergence: ", tsne.kl_divergence_)
    output_path_png = os.path.join(output_path, f"{model.name}_latent_space.png")
    sns.scatterplot(x=X_train_tsne[:, 0], y=X_train_tsne[:, 1], hue=dataset.gt_labels)
    plt.savefig(output_path_png)
    plt.show()

    return tsne.kl_divergence_
