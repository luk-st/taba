import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from taba.models.adm.ddim import generate_latents, generate_noises, generate_samples
from taba.models.adm.models import get_openai_cifar, get_openai_imagenet
from taba.models.ldms.models import get_inv_scheduler, get_ldm_celeba, get_scheduler
from taba.models.ldms.sample_ldm import generate_latents as ldm_generate_latents
from taba.models.ldms.sample_ldm import generate_samples as ldm_generate_samples
from taba.models.ldms.sample_ldm import get_noises as ldm_get_noises

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T_DEFAULT = 100
N_SAMPLES = 2048
N_ALPHAS = 25
BATCH_SIZE = 128


def plot_df_closer(df: pd.DataFrame, name: str, name_math: str = None, n_samples: int = N_SAMPLES):
    name_math = name_math or name
    plt.figure(figsize=(20, 6))
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
    plt.tight_layout(rect=[0, 0.03, 1, 0.85])
    plt.style.use("seaborn-v0_8-paper")
    plt.bar(df[f"{name} Diffusion Step"], df["Closer to Noise"] / n_samples, label="Closer to Noise", color="blue")
    plt.bar(
        df[f"{name} Diffusion Step"],
        df["Closer to Latent"] / n_samples,
        bottom=df["Closer to Noise"] / n_samples,
        label="Closer to Latent",
        color="orange",
    )
    plt.xlabel("Diffusion Step $t$", fontsize=25)
    plt.ylabel("")
    plt.xlim(0, 100)
    plt.ylim(0, 1)
    plt.yticks([0.0, 0.5, 1.0], fontsize=25)
    plt.xticks(list(range(0, 101, 20)), fontsize=25)
    plt.xticks(list(range(0, 101, 20)), fontsize=25)
    plt.title(
        f"{name_math}: % of Intermediate Diffusion Steps Closer to Noise and Latent\nper Diffusion Step ($N={n_samples}$)",
        fontsize=30,
        pad=55,
    )
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, fontsize=22, markerscale=2)
    plt.show()


def plot_interpolation_distances(distances: np.ndarray, filename=None):
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
    sns.heatmap(distances, cmap="flare", annot_kws={"size": 100})

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2, fontsize="x-large")
    plt.ylabel("Interpolation Step", fontsize=32)
    plt.xlabel("Denoising Step $t$", fontsize=32)
    plt.xlim(0, distances.shape[1])
    plt.ylim(0, distances.shape[0])

    # Set y-ticks with a diagonal label in the bottom-left corner
    plt.yticks([5, 10, 15, 20], ["$5$", "$10$", "$15$", "$20$"], fontsize=20, rotation=0)
    plt.xticks([10, 30, 50, 70, 90], reversed(["$10$", "$30$", "$50$", "$70$", "$90$"]), fontsize=20, rotation=0)
    plt.text(0, -2.0, r"$x^T$", fontsize=30, ha="right", va="bottom")
    plt.text(0, 24.0, r"$\hat{x}^T$", fontsize=30, ha="right", va="bottom")
    plt.text(106.0, -2.0, r"$x^0$", fontsize=30, ha="right", va="bottom")

    # Customize the diagonal label for the bottom-left corner
    # ax = plt.gca()
    # ax.set_xticks([0], minor=True)
    # ax.set_yticks([0], minor=True)
    # ax.tick_params(which='minor', length=0)  # Hide minor tick marks

    # Set diagonal label for the bottom-left corner
    # ax.set_xticklabels([r'$x^T$'], minor=True, fontsize=20, rotation=45, ha='right', rotation_mode='anchor')
    # ax.set_yticklabels([r'$x^T$'], minor=True, fontsize=25, rotation=0, ha='right', rotation_mode='anchor')

    if filename is not None:
        plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.show()


def build_norm_df_closer(
    noises: torch.Tensor,
    latents: torch.Tensor,
    ts_samples: torch.Tensor,
    ts_latents: torch.Tensor,
    T: int = T_DEFAULT,
    n_samples: int = N_SAMPLES,
):
    """Builds normalized df with number of samples closer to noise and latent per diffusion step."""
    sampling_num_closer_to_noise = torch.zeros(T + 1)
    sampling_num_closer_to_latent = torch.zeros(T + 1)
    rev_sampling_num_closer_to_noise = torch.zeros(T + 1)
    rev_sampling_num_closer_to_latent = torch.zeros(T + 1)

    for t in range(T + 1):
        for i in range(n_samples):
            sample_t = ts_samples[t, i]
            latent_t = ts_latents[t, i]

            dist_to_noise = torch.linalg.vector_norm(sample_t - noises[i])
            dist_to_latent = torch.linalg.vector_norm(sample_t - latents[i])
            if dist_to_noise < dist_to_latent:
                sampling_num_closer_to_noise[t] += 1
            else:
                sampling_num_closer_to_latent[t] += 1
            dist_to_noise = torch.linalg.vector_norm(latent_t - noises[i])
            dist_to_latent = torch.linalg.vector_norm(latent_t - latents[i])
            if dist_to_noise < dist_to_latent:
                rev_sampling_num_closer_to_noise[t] += 1
            else:
                rev_sampling_num_closer_to_latent[t] += 1

    df_sampling = pd.DataFrame(
        {
            "DDIM Diffusion Step": list(range(T + 1)),
            "Closer to Noise": sampling_num_closer_to_noise.tolist(),
            "Closer to Latent": sampling_num_closer_to_latent.tolist(),
        }
    )
    df_rev_sampling = pd.DataFrame(
        {
            "DDIM^-1 Diffusion Step": list(range(T + 1)),
            "Closer to Noise": rev_sampling_num_closer_to_noise.tolist(),
            "Closer to Latent": rev_sampling_num_closer_to_latent.tolist(),
        }
    )
    return df_sampling, df_rev_sampling


def get_samples_latents(
    model_name: str, T: int = T_DEFAULT, batch_size: int = BATCH_SIZE, n_samples: int = N_SAMPLES, device: str = DEVICE
):
    if model_name in {"ddpm-imagenet-64", "ddpm-cifar10-32"}:
        f_get_model = get_openai_imagenet if model_name == "ddpm-imagenet-64" else get_openai_cifar
        model, diffusion, args = f_get_model(steps=T, device=device)
        noises = generate_noises(n_samples, args)
        outs_sampling = generate_samples(
            random_noises=noises,
            number_of_samples=n_samples,
            batch_size=batch_size,
            diffusion_pipeline=diffusion,
            ddim_model=model,
            diffusion_args=args,
            device=device,
            from_each_t=True,
        )
        samples = outs_sampling["samples"]
        all_t_samples = outs_sampling["all_t_samples"]
        outs_noising = generate_latents(
            ddim_generations=samples,
            batch_size=batch_size,
            diffusion_pipeline=diffusion,
            ddim_model=model,
            device=device,
            from_each_t=True,
        )
        latents, all_t_latents = outs_noising["latents"], outs_noising["all_t_latents"]
    elif model_name == "ldm-celeba-64":
        ldm_unet, _ = get_ldm_celeba(device=device)
        scheduler = get_scheduler(T=T)
        inv_scheduler = get_inv_scheduler(T=T)

        noises = ldm_get_noises(n_samples=n_samples)
        samples, all_t_samples = ldm_generate_samples(
            noise=noises,
            diffusion_unet=ldm_unet,
            diffusion_scheduler=scheduler,
            batch_size=batch_size,
            device=device,
            from_each_t=True,
        )
        latents, all_t_latents = ldm_generate_latents(
            samples=samples,
            diffusion_unet=ldm_unet,
            diffusion_scheduler=inv_scheduler,
            batch_size=batch_size,
            device=device,
            from_each_t=True,
        )
    else:
        raise NotImplementedError(f"Unknown model name: {model_name}")

    return noises.cpu(), samples, all_t_samples, latents, all_t_latents


def linear_inter(l1: torch.Tensor, l2: torch.Tensor, a: float):
    return a * l1 + (1 - a) * l2


def get_interpolations_distances(
    all_t: torch.Tensor, noises: torch.Tensor, latents: torch.Tensor, alphas: int = N_ALPHAS
):
    n_samples = noises.shape[0]
    n_steps = all_t.shape[0]
    interpolations = torch.stack([linear_inter(latents, noises, alpha) for alpha in np.linspace(0, 1, alphas)])

    distances = torch.zeros(n_samples, n_steps, alphas)
    for i in tqdm(range(n_steps), desc="Calculating per step distances"):
        for j in range(alphas):
            for k in range(n_samples):
                distances[k, i, j] = torch.linalg.vector_norm(all_t[i][k] - interpolations[j][k])
    distances = torch.mean(distances, dim=(0))
    return distances.numpy()


def run_model_exp(model_name: str):
    # noises, samples, all_t_samples, latents, all_t_latents = get_samples_latents(model_name)

    # torch.save(noises, f"noises_{model_name}.pt")
    # torch.save(samples, f"samples_{model_name}.pt")
    # torch.save(latents, f"latents_{model_name}.pt")
    # torch.save(all_t_samples, f"all_t_samples_{model_name}.pt")
    # torch.save(all_t_latents, f"all_t_latents_{model_name}.pt")

    path = "experiments/outputs_all_ts/imagenet256/T_100"
    # dir_name = model_name.split("-")[1]
    noises = torch.load(f"{path}/all_noise.pt")
    latents = torch.load(f"{path}/all_latents.pt")
    all_t_samples = torch.load(f"{path}/all_ts_samples.pt")

    # df_sampling, df_rev_sampling = build_norm_df_closer(noises, latents, all_t_samples, all_t_latents)
    # plot_df_closer(df_sampling, "DDIM", "$DDIM$")
    # plot_df_closer(df_rev_sampling, "DDIM^-1", "$DDIM^{-1}$")
    distances_ddim = get_interpolations_distances(all_t_samples, noises, latents)
    plot_interpolation_distances(distances_ddim.T)


def run_models():
    for model_name in ["ddpm-imagenet-64", "ddpm-cifar10-32", "ldm-celeba-64"]:
        print(f"Running {model_name}")
        run_model_exp(model_name=model_name)
