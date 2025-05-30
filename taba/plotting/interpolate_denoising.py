import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from taba.exp_pipelines.interpolate_with_steps import get_interpolations_distances


def min_max_scale(arr):
    min_vals = arr.min(axis=1, keepdims=True)
    max_vals = arr.max(axis=1, keepdims=True)
    return (arr - min_vals) / (max_vals - min_vals)


def run_model(model_name: str, path: str):
    dir_name = model_name
    noises = torch.load(f"{path}/{dir_name}/noises_{model_name}.pt")
    latents = torch.load(f"{path}/{dir_name}/latents_{model_name}.pt")
    all_t_samples = torch.load(f"{path}/{dir_name}/all_t_samples_{model_name}.pt")
    distances_ddim = get_interpolations_distances(all_t_samples, noises, latents)
    scaled_distances_ddim = min_max_scale(distances_ddim)
    return scaled_distances_ddim


path = "experiments/interpolate_diffusion"
scaled_distances_ddim_dit = run_model(model_name="imagenet_dit_256", path=path)
scaled_distances_ddim_celeba = run_model(model_name="celeba_ldm_256", path=path)
scaled_distances_ddim_imagenet = run_model(model_name="imagenet_pixel_64", path=path)


def plot_interpolation_distances(distances_list: list, models_names: list, filename=None):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    plt.subplots_adjust(wspace=0.3)

    plt.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "font.size": 15,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "lines.linewidth": 2,
            "text.usetex": False,
            "pgf.rcfonts": False,
        }
    )
    plt.style.use("seaborn-v0_8-paper")

    for ax, distances, model_name in zip(axes, distances_list, models_names):
        sns.heatmap(distances, cmap="flare", annot_kws={"size": 100}, ax=ax, cbar=False)
        ax.set_title(model_name, fontsize=34)

    # Set common labels
    # fig.text(0.5, 0.04, "Denoising Step $t$", ha="center", fontsize=32)
    # fig.text(0.04, 0.5, "Interpolation Step", va="center", rotation="vertical", fontsize=32)
    # axes[0].set_ylabel("Interpolation Step", fontsize=32, labelpad=15)
    fig.supylabel("Interpolation Step", fontsize=34, x=0.03)
    fig.supxlabel("Denoising Step $t$", fontsize=34, y=-0.2, x=0.51)

    # Set common x and y limits
    for ax in axes:
        ax.set_xlim(0, distances_list[0].shape[1])
        ax.set_ylim(0, distances_list[0].shape[0])

    # Set common ticks
    for ax in axes:
        ax.set_yticks([5, 12.5, 20])
        ax.set_yticklabels(["$0.2$", "$0.5$", "$0.8$"], fontsize=32, rotation=0)
        ax.set_xticks([20, 40, 60, 80])
        ax.set_xticklabels(reversed(["$20$", "$40$", "$60$", "$80$"]), fontsize=32, rotation=0)
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(0.5)

    x1 = 0.13
    y1 = -0.01
    x2 = 0.36
    y2 = 0.82
    fig.text(x1, y1, r"$x^T$", fontsize=32, ha="right", va="bottom")
    fig.text(x1, y2, r"$\hat{x}^T$", fontsize=32, ha="right", va="bottom")
    fig.text(x2, y1, r"$x^0$", fontsize=32, ha="right", va="bottom")

    x1 = x1 + 0.28
    x2 = x2 + 0.28
    fig.text(x1, y1, r"$x^T$", fontsize=32, ha="right", va="bottom")
    fig.text(x1, y2, r"$\hat{x}^T$", fontsize=32, ha="right", va="bottom")
    fig.text(x2, y1, r"$x^0$", fontsize=32, ha="right", va="bottom")

    x1 = x1 + 0.28
    x2 = x2 + 0.28
    fig.text(x1, y1, r"$x^T$", fontsize=32, ha="right", va="bottom")
    fig.text(x1, y2, r"$\hat{x}^T$", fontsize=32, ha="right", va="bottom")
    fig.text(x2, y1, r"$x^0$", fontsize=32, ha="right", va="bottom")

    # Add a single colorbar
    cbar_ax = fig.add_axes(rect=(0.92, 0.15, 0.02, 0.7))
    norm = plt.Normalize(vmin=0.0, vmax=1.0)
    sm = plt.cm.ScalarMappable(cmap="flare", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax)
    cbar_ax.tick_params(labelsize=28)

    if filename is not None:
        plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.show()


plot_interpolation_distances(
    [scaled_distances_ddim_imagenet.T, scaled_distances_ddim_celeba.T, scaled_distances_ddim_dit.T],
    ["ADM-64", "LDM", "DiT"],
    filename="interpolation_distances.pdf",
)
