import os
import sys

DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, DIR)

import torch
from networks.VAEs import Params_VAE, VAE
from utils.settings import DEVICE


def get_latents(metabric, omics_names, vae_path, fold_k, dense_dim=256, latent_dim=128):

    latents = []
    for name in omics_names:
        input_dim = metabric[name].shape[0]

        vae_path = vae_path % (name, fold_k)

        vae_params = Params_VAE(input_dim, dense_dim, latent_dim)
        vae = VAE(vae_params)
        vae.load_state_dict(torch.load(vae_path))
        vae.eval()
        vae.to(DEVICE)

        _, _, _, _, z = vae.forward(torch.tensor(metabric[name], dtype=torch.float32))
        z = z.detach().cpu().numpy()
        latents.append(z)

    return latents
