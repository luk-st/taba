from typing import Tuple

import torch
import torch.nn.functional as F


def get_cossim_mean_std(vec1, vec2):
    vec1 = vec1.view(vec1.shape[0], -1)
    vec2 = vec2.view(vec2.shape[0], -1)
    vec1 = torch.nn.functional.normalize(vec1, dim=1)
    vec2 = torch.nn.functional.normalize(vec2, dim=1)

    cossims = torch.sum(vec1 * vec2, dim=1)
    return cossims.mean(), cossims.std()


def calc_angles(noises, samples, latents):
    vec_image_to_noise = noises - samples
    vec_image_to_latent = latents - samples

    vec_noise_to_image = samples - noises
    vec_noise_to_latent = latents - noises

    vec_latent_to_noise = noises - latents
    vec_latent_to_image = samples - latents

    mean_cossim_img, std_cossim_img = get_cossim_mean_std(vec_image_to_noise, vec_image_to_latent)
    mean_cossim_noise, std_cossim_noise = get_cossim_mean_std(vec_noise_to_image, vec_noise_to_latent)
    mean_cossim_latent, std_cossim_latent = get_cossim_mean_std(vec_latent_to_noise, vec_latent_to_image)

    # convert to angles (degrees)
    mean_cossim_img = torch.rad2deg(torch.acos(mean_cossim_img))
    mean_cossim_noise = torch.rad2deg(torch.acos(mean_cossim_noise))
    mean_cossim_latent = torch.rad2deg(torch.acos(mean_cossim_latent))
    std_cossim_img = torch.rad2deg(torch.acos(std_cossim_img))
    std_cossim_noise = torch.rad2deg(torch.acos(std_cossim_noise))
    std_cossim_latent = torch.rad2deg(torch.acos(std_cossim_latent))

    return {
        "mean_cossim_img": mean_cossim_img,
        "mean_cossim_noise": mean_cossim_noise,
        "mean_cossim_latent": mean_cossim_latent,
        "std_cossim_img": std_cossim_img,
        "std_cossim_noise": std_cossim_noise,
        "std_cossim_latent": std_cossim_latent,
    }


def _mean_mse(tens1: torch.Tensor, tens2: torch.Tensor) -> float:
    mse = F.mse_loss(tens1, tens2, reduction="none")
    return mse.mean(dim=[1, 2, 3]).mean()


def l2_dist_mean_std(tens1: torch.Tensor, tens2: torch.Tensor) -> Tuple[float, float]:
    l2 = torch.norm(tens1 - tens2, p=2, dim=[1, 2, 3])
    return l2.mean(), l2.std()


def calc_distances(noise, sample_from_noise, latent, sample_from_latent=None) -> dict:
    mean_sample_latent, std_sample_latent = l2_dist_mean_std(sample_from_noise, latent)
    mean_sample_noise, std_sample_noise = l2_dist_mean_std(sample_from_noise, noise)
    mean_noise_latent, std_noise_latent = l2_dist_mean_std(noise, latent)

    out = {
        "mean_l2_img_lat": mean_sample_latent,
        "mean_l2_img_noise": mean_sample_noise,
        "mean_l2_noise_lat": mean_noise_latent,
        "std_l2_img_lat": std_sample_latent,
        "std_l2_img_noise": std_sample_noise,
        "std_l2_noise_lat": std_noise_latent,
    }
    if sample_from_latent is not None:
        out["mean_mse_img_img2"] = _mean_mse(tens1=sample_from_noise, tens2=sample_from_latent)
    return out


def reconstruction_error(tensor_original: torch.Tensor, tensor_reconstructed: torch.Tensor):
    return torch.abs(tensor_original - tensor_reconstructed).mean().item()
