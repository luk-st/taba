import torch

from taba.exp_pipelines.interpolate_with_steps import (
    get_interpolations_distances,
    plot_interpolation_distances,
)


def min_max_scale(arr):
    min_vals = arr.min(axis=1, keepdims=True)
    max_vals = arr.max(axis=1, keepdims=True)
    return (arr - min_vals) / (max_vals - min_vals)


def run_model(model_name: str, path: str):
    dir_name = model_name.split("-")[1]
    noises = torch.load(f"{path}/{dir_name}/noises_{model_name}.pt")
    # samples = torch.load(f"{path}/{dir_name}/samples_{model_name}.pt")
    latents = torch.load(f"{path}/{dir_name}/latents_{model_name}.pt")
    all_t_samples = torch.load(f"{path}/{dir_name}/all_t_samples_{model_name}.pt")
    # all_t_latents = torch.load(f"{path}/{dir_name}/all_t_latents_{model_name}.pt")
    distances_ddim = get_interpolations_distances(all_t_samples, noises, latents)
    scaled_distances_ddim = min_max_scale(distances_ddim)
    return scaled_distances_ddim


model_name = "ddpm-imagenet-64"

path = "<PROJECT_PATH>/experiments/interpolate_diffusion"
scaled_distances_ddim = run_model(model_name, path)
plot_interpolation_distances(scaled_distances_ddim.T, filename="imagenet_heatmap.pdf")

model_name = "ddpm-cifar10-32"

path = "<PROJECT_PATH>/experiments/interpolate_diffusion"
scaled_distances_ddim = run_model(model_name, path)
plot_interpolation_distances(scaled_distances_ddim.T, filename="cifar_heatmap.pdf")

model_name = "ldm-celeba-64"
path = "<PROJECT_PATH>/experiments/interpolate_diffusion"
scaled_distances_ddim = run_model(model_name, path)
plot_interpolation_distances(scaled_distances_ddim.T, filename="ldm_heatmap.pdf")
