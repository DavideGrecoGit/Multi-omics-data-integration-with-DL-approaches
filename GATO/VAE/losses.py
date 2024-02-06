import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_kernel(x, y):
    x_size = x.shape[0]
    y_size = y.shape[0]
    dim = x.shape[1]

    tiled_x = x.view(x_size, 1, dim).repeat(1, y_size, 1)
    tiled_y = y.view(1, y_size, dim).repeat(x_size, 1, 1)

    return torch.exp(-torch.mean((tiled_x - tiled_y) ** 2, dim=2) / dim * 1.0)


def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)


def compute_kld(latent_mean, latent_log_var):
    return -0.5 * torch.mean(
        1 + latent_log_var - latent_mean.pow(2) - latent_log_var.exp()
    )


def compute_vae_loss(
    loss_fn,
    regularisation,
    beta,
    x,
    reconstructed,
    latent_mean,
    latent_log_var,
    z,
    latent_dim,
):
    reconstruction_loss = loss_fn(reconstructed, x)

    match regularisation:
        case "mmd":
            true_samples = torch.randn(z.size()[0], latent_dim).to(DEVICE)
            regularisation_loss = compute_mmd(true_samples, z)
        case "kld":
            regularisation_loss = compute_kld(latent_mean, latent_log_var)
        case _:
            regularisation_loss = 0

    return reconstruction_loss + (beta * regularisation_loss)
