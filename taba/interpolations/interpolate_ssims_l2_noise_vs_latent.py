import argparse
import random
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patheffects import withStroke
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

from taba.interpolations.interpolations import linear_interpolation, slerp_interpolation
from taba.models.adm.models import get_ddpm_imagenet256
from taba.models.adm.ddim import generate_samples

DEFAULT_PATH = "experiments/outputs_per_T/imagenet_pixel_256/T_100"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 64
DEFAULT_N_ALPHAS = 12
DEFAULT_N_EXAMPLES = 128


def load_data(os_path: str) -> Path:
    path_dir = Path(os_path)
    samples = torch.load(path_dir / "samples.pt").cpu()
    noises = torch.load(path_dir / "noise.pt").cpu()
    latents = torch.load(path_dir / "latents.pt").cpu()
    return noises, samples, latents


def seed_everything(seed: int = DEFAULT_SEED) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_model(
    model_type: str, n_steps: int = 100, device: str = DEFAULT_DEVICE
) -> Tuple[torch.nn.Module, torch.nn.Module, dict]:
    if model_type == "imagenet_pixel_256":
        model, diffusion, args = get_ddpm_imagenet256(steps=n_steps, device=device)
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")
    return model, diffusion, args


def calculate_ssim(img1, img2):
    ssim_values = []
    for i in range(3):  # For each color channel
        ssim_value = ssim(img1[:, :, i], img2[:, :, i], data_range=1)
        ssim_values.append(ssim_value)
    return np.mean(ssim_values)


def calculate_l2_norm(img1, img2):
    return torch.norm(img1 - img2, p=2)


def plot_interpolation_distances(outs: dict, filename=None):
    plt.figure(figsize=(12, 8))
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
    plt.tight_layout(rect=[0, 0.0, 1, 1])
    plt.style.use("seaborn-v0_8-paper")

    colors = ["#FF6464", "#8E51F8", "#003049", "#DDA217"]

    for idx, (key, value) in enumerate(outs.items()):
        plt.plot(
            value,
            label=key,
            color=colors[idx],
            linewidth=3,
            path_effects=[withStroke(linewidth=4, foreground="black")],
        )

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=2, fontsize="x-large")
    plt.xlabel("Interpolation Step", fontsize=32)
    plt.ylabel("Metric value", fontsize=32)
    plt.xlim(0, outs[next(iter(outs.keys()))].shape[0] - 1)
    plt.ylim(0, 1.1)

    plt.xticks([0.0, 3, 7, 11], ["$0.0$", "$0.33$", "$0.66$", "$1.0$"], fontsize=20, rotation=0)
    plt.yticks([0.1, 0.4, 0.7, 1.0], ["$0.1$", "$0.4$", "$0.7$", "$1.0$"], fontsize=20, rotation=0)

    if filename is not None:
        plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.show()


def run(path, model_type, interpolation, batch_size, n_alphas, n_examples):
    seed_everything()
    noises, samples, latents = load_data(path)
    model, diffusion, model_args = load_model(model_type)

    idxs_32l = torch.randperm(samples.shape[0])[:DEFAULT_N_EXAMPLES]
    idxs_32r = torch.randperm(samples.shape[0])[:DEFAULT_N_EXAMPLES]

    if interpolation == "slerp":
        interpolation_func = slerp_interpolation
    elif interpolation == "linear":
        interpolation_func = linear_interpolation
    else:
        raise ValueError(f"Interpolation type {args.interpolation} not supported")

    objects1_32l = latents[idxs_32l]
    objects1_32r = latents[idxs_32r]
    objects2_32l = noises[idxs_32l]
    objects2_32r = noises[idxs_32r]

    alphas = np.linspace(0, 1, n_alphas)
    all_interpolated_objects1 = []
    for idx in range(n_examples):
        o1_1 = objects1_32l[idx]
        o1_2 = objects1_32r[idx]
        interpolated_objects1 = torch.stack([interpolation_func(o1_1, o1_2, alpha) for alpha in alphas])
        all_interpolated_objects1.append(interpolated_objects1)
    all_interpolated_objects1 = torch.concat(all_interpolated_objects1)

    gens1_interpolated = generate_samples(
        random_noises=all_interpolated_objects1.to("cuda"),
        number_of_samples=all_interpolated_objects1.shape[0],
        batch_size=batch_size,
        diffusion_pipeline=diffusion,
        ddim_model=model,
        diffusion_args=model_args,
        device="cuda",
    )
    torch.save(gens1_interpolated.cpu(), path + "/gens1_interpolated.pt")

    all_interpolated_objects2 = []
    for idx in range(n_examples):
        o2_1 = objects2_32l[idx]
        o2_2 = objects2_32r[idx]
        interpolated_objects2 = torch.stack([interpolation_func(o2_1, o2_2, alpha) for alpha in alphas])
        all_interpolated_objects2.append(interpolated_objects2)
    all_interpolated_objects2 = torch.concat(all_interpolated_objects2)

    gens2_interpolated = generate_samples(
        random_noises=all_interpolated_objects2.to("cuda"),
        number_of_samples=all_interpolated_objects2.shape[0],
        batch_size=batch_size,
        diffusion_pipeline=diffusion,
        ddim_model=model,
        diffusion_args=model_args,
        device="cuda",
    )
    torch.save(gens2_interpolated.cpu(), path + "/gens2_interpolated.pt")
    ssims = [
        calculate_ssim(
            gens1_interpolated[i].permute(1, 2, 0).numpy(),
            gens2_interpolated[i].permute(1, 2, 0).numpy(),
        )
        for i in tqdm(range(gens1_interpolated.shape[0]))
    ]
    l2_norms = [
        calculate_l2_norm(gens1_interpolated[i], gens2_interpolated[i])
        for i in tqdm(range(gens1_interpolated.shape[0]))
    ]

    ssims_out = np.mean(np.array(ssims).reshape(-1, n_alphas), axis=0)
    l2_outs = np.mean(np.array(l2_norms).reshape(-1, n_alphas), axis=0)

    print(f"SSIM: {ssims_out}")
    print(f"L2: {l2_outs}")

    plot_interpolation_distances(
        {"SSIM": ssims_out, r"$L_2$-Norm $\times 10^{-2}$": l2_outs / 100},
        filename=f"interpolation_{interpolation}_noise_latent.pdf",
    )


def main(args):
    run(args.path, args.model_type, args.interpolation, args.batch_size, args.n_alphas, args.n_examples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=DEFAULT_PATH)
    parser.add_argument("--model_type", type=str, default="imagenet_pixel_256")
    parser.add_argument("--interpolation", type=str, choices=["slerp", "linear"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--n_alphas", type=int, default=DEFAULT_N_ALPHAS)
    parser.add_argument("--n_examples", type=int, default=DEFAULT_N_EXAMPLES)
    args = parser.parse_args()
    main(args)
