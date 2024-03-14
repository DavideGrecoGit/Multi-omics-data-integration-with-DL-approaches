import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy_loss(y_pred, y_true, weight=None, reduction="mean"):
    criterion = nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    return criterion(y_pred, y_true)


def norm_diff(W):
    """Special norm function for the last layer of the MTLR"""
    dims = len(W.shape)
    if dims == 1:
        diff = W[1:] - W[:-1]
    elif dims == 2:
        diff = W[1:, :] - W[:-1, :]
    return torch.sum(diff * diff)


def l2_loss(
    model,
    l2_reg=1e-2,
    l2_smooth=1e-2,
):
    loss = 0.0
    # Adding the regularized loss
    nb_set_parameters = len(list(model.parameters()))
    for i, w in enumerate(model.parameters()):
        loss += l2_reg * torch.sum(w * w) / 2.0

        if i >= nb_set_parameters - 2:
            loss += l2_smooth * norm_diff(w)

    return loss


def MTLR_loss(
    y_pred,
    y_true,
    uncensor_mask,
    tri_matrix_1,
    reduction="mean",
):
    """
    Compute the MTLR survival loss
    """

    y_true_censor = y_true[~uncensor_mask]
    y_true_uncensor = y_true[uncensor_mask]
    y_pred_censor = y_pred[~uncensor_mask]
    y_pred_uncensor = y_pred[uncensor_mask]

    # Calculate likelihood for censored datapoint
    phi_censor = torch.exp(torch.mm(y_pred_censor, tri_matrix_1))
    reduc_phi_censor = torch.sum(phi_censor * y_true_censor, dim=1)

    # Calculate likelihood for uncensored datapoint
    phi_uncensor = torch.exp(torch.mm(y_pred_uncensor, tri_matrix_1))
    reduc_phi_uncensor = torch.sum(phi_uncensor * y_true_uncensor, dim=1)

    # Likelihood normalisation
    z_censor = torch.exp(torch.mm(y_pred_censor, tri_matrix_1))
    reduc_z_censor = torch.sum(z_censor, dim=1)
    z_uncensor = torch.exp(torch.mm(y_pred_uncensor, tri_matrix_1))
    reduc_z_uncensor = torch.sum(z_uncensor, dim=1)

    # MTLR loss
    loss = -(
        torch.sum(torch.log(reduc_phi_censor))
        + torch.sum(torch.log(reduc_phi_uncensor))
        - torch.sum(torch.log(reduc_z_censor))
        - torch.sum(torch.log(reduc_z_uncensor))
    )

    if reduction == "mean":
        loss = loss / uncensor_mask.shape[0]

    return loss
