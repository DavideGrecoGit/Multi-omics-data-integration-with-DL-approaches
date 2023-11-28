from torch import nn, optim
import torch
from networks.encoders_decoders import MoGCN_AE, MoGCN_VAE
from networks.gnn import GCN
from utils.train_val_test import train_loop
from torch.utils.data import DataLoader
from utils.snf import run_snf


class Classifier(nn.Module):
    """
    Defines the classifier made up of encoder-decoder and GNN networks
    """

    def __init__(
        self,
        dataset,
        encoder_input_dims,
        encoder,
        gnn,
        encoder_latent_dim=100,
        encoder_activation_fn=nn.Sigmoid(),
        encoder_d_p=0,
        beta=0.001,
        gnn_hidden_dim=64,
        gnn_d_p=0,
        n_classes=5,
        name="Classifier",
    ):
        super(Classifier, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = name

        match encoder:
            case "ae":
                self.encoder = MoGCN_AE(
                    encoder_input_dims,
                    encoder_activation_fn,
                    encoder_d_p,
                    encoder_latent_dim,
                )
            case "vae":
                self.encoder = MoGCN_VAE(
                    encoder_input_dims,
                    encoder_activation_fn,
                    encoder_d_p,
                    encoder_latent_dim,
                    beta,
                )
            case _:
                self.encoder = torch.load(encoder)

        gnn_input_dims = encoder_latent_dim

        match gnn:
            case "gcn":
                self.gnn = GCN(gnn_input_dims, gnn_hidden_dim, n_classes, gnn_d_p)
            case _:
                self.gnn = torch.load(gnn)

    # def test(self, x_omics,  gt_classes):
    #     encoder_latent_data = self.encoder.get_latent_space(x_omics)

    #     output, gnn_loss = self.gnn.forward_pass(
    #         encoder_latent_data, x_adj_matrix, gt_classes
    #     )

    #     # calculate the accuracy
    #     acc_test = accuracy(output, gt_classes)

    #     # output is the one-hot label
    #     ot = output.detach().cpu().numpy()
    #     # change one-hot label to digit label
    #     ot = np.argmax(ot, axis=1)
    #     # original label
    #     lb = gt_classes.detach().cpu().numpy()
    #     print("predict label: ", ot)
    #     print("original label: ", lb)

    #     # calculate the f1 score
    #     f = f1_score(ot, lb, average="weighted")

    #     print(
    #         "Test set results:",
    #         "loss= {:.4f}".format(gnn_loss.item()),
    #         "accuracy= {:.4f}".format(acc_test.item()),
    #     )

    #     # return accuracy and f1 score
    #     return acc_test.item(), f
