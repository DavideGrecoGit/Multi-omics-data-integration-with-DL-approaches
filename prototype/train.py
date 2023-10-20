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


def train_loop(model, train_loader, optimizer, loss_fn, num_epochs=100):
    loss_ls = []

    # Train
    for epoch in range(num_epochs):
        train_loss_sum = 0.0  # Record the loss of each epoch

        for batch_idx, (x_omics, _) in enumerate(train_loader):
            # Forward & Loss
            output, loss = model.forward_pass(x_omics, loss_fn)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()

        loss_ls.append(train_loss_sum)

        if epoch == 0 or (epoch + 1) % 20 == 0:
            print("epoch: %d | loss: %.4f" % (epoch + 1, train_loss_sum))

    # draw the training loss curve
    plt.plot([i + 1 for i in range(num_epochs)], loss_ls)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.savefig(f"result/{model.name}_train_{num_epochs}.png")
    plt.show()

    return loss_ls[-1]


def test(model, test_loader, loss_fn):
    batch_loss_sum = 0.0  # Record the loss of each epoch

    # Test
    with torch.no_grad():
        for batch_idx, (x_omics, _) in enumerate(test_loader):
            # Forward & Loss
            output, loss = model.forward_pass(x_omics, loss_fn)

            batch_loss_sum += loss.item()

    test_loss = batch_loss_sum / len(test_loader)
    print("Test Loss: %.4f" % (test_loss))

    return test_loss


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
    SEED=42,
):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    sum_train_loss = 0.0
    sum_test_loss = 0.0
    sum_kl_divergence = 0.0

    for i, (train_index, test_index) in enumerate(
        skf.split(omics_data[0], gt_labels["class"])
    ):
        DEVICE = torch.device("cpu")

        # Config
        if torch.cuda.is_available():
            # Clear CUDA
            torch.cuda.empty_cache()
            gc.collect()
            DEVICE = torch.device("cuda")

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
        test_loss = test(model, test_loader, loss_fn)

        # Save model
        torch.save(model, f"model/AE/{model.name}_model.pkl")

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


def save_latent_data(dataset, model):
    with torch.no_grad():
        latent_space = model.get_latent_space(dataset.omics_data)

    # Save latent space
    latent_df = pd.DataFrame(latent_space)
    latent_df.insert(0, "Sample", dataset.samples_list)
    latent_df.to_csv(f"result/{model.name}_latent_data.csv", header=True, index=False)

    # Plot latent space
    tsne = TSNE(n_components=2, perplexity=40, random_state=42)
    X_train_tsne = tsne.fit_transform(latent_space)

    print("T-SNE KL Divergence: ", tsne.kl_divergence_)

    sns.scatterplot(x=X_train_tsne[:, 0], y=X_train_tsne[:, 1], hue=dataset.gt_labels)
    plt.savefig(f"result/{model.name}_latent_space.png")
    plt.show()

    return tsne.kl_divergence_
