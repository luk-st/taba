# %% [markdown]
# # Imports, etc.

import pickle
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

# %%
from taba.utils import image_grid


def load_file(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def common_keys(dict1, dict2, dict3):
    return set(dict1.keys()) & set(dict2.keys()) & set(dict3.keys())


# %%
from torchvision import transforms

from taba.utils import image_grid


def tensors_to_pils_single(tensor):
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()
    return transforms.ToPILImage()(tensor)


def grid_tensors(tensors, rows: int = 1, border_size: int = 0):
    assert len(tensors.shape) == 4, f"tensors must have 4 dimensions, got {tensors.shape}"
    return image_grid([tensors_to_pils_single(i_tens) for i_tens in tensors], rows=rows, border_size=border_size)


# %%
IMAGENET_CKPTS = sorted(
    list(
        set(
            [0, 25, 100]
            + list(range(0, 10_308, 2500))
            + list(range(0, 522_500, 2 * 10_307))
            + list(range(522_500, 1_130_613, 2 * 10_307))
            + list(range(1_130_613, 1_500_613, 2 * 10_307))
        )
    )
)

CIFAR_CKPT = sorted(
    list(
        set(
            [0, 25]  # 2
            + list(range(50, 401, 100))  # 4
            + list(range(0, 100 * 390 + 1, 5 * 390))  # 20
            + list(range(101 * 390, 429390, 25 * 390))  # 40
            + list(range(425000, 700_000, 5_000))  # 55
        )
    )
)

# %% [markdown]
# # STATS


# %%
def get_full_stats_dict(path, seeds=[0, 10, 42], steps_to_remove=[]):
    assert len(seeds) > 0, "Seeds list cannot be empty"

    seeds_stats = []
    for curr_seed in seeds:
        stats_files_s = list(path.glob(f"stats_s{curr_seed}_stidx*"))
        stats_s = [load_file(path) for path in stats_files_s]
        merged_s = {}
        for stat in stats_s:
            merged_s.update(stat)
        seeds_stats.append(merged_s)
    common_keys_set = common_keys(*seeds_stats)
    print("Common keys in all three dictionaries:", common_keys_set)

    average_dict = {}
    for step_key_idx in common_keys_set:
        vals = [seeds_stats[idx][step_key_idx] for idx in range(len(seeds_stats))]
        val_key = {}
        for k in vals[0].keys():
            new_vals = np.stack([val[k].numpy() if isinstance(val[k], torch.Tensor) else val[k] for val in vals])
            if k.endswith("mean"):
                val_key[k] = np.mean(new_vals)
            elif k.endswith("std"):
                val_key[k] = np.sqrt(np.mean(new_vals**2))
            elif k in ["svcca", "cka"]:
                val_key[f"{k}_mean"] = np.mean(new_vals)
                val_key[f"{k}_std"] = np.std(new_vals)
        average_dict[step_key_idx] = val_key

    for step_key_idx in steps_to_remove:
        if step_key_idx in average_dict:
            del average_dict[step_key_idx]
        else:
            print(f"Step key index to remove: {step_key_idx} not found in average dictionary")

    return average_dict


# %%
from matplotlib import pyplot as plt
from matplotlib.patheffects import withStroke


def plot_trainstep_alignment(
    data,
    first_batch_idx=None,
    limit_to=None,
    title=None,
    filename=None,
    used_metrics=["DINO", "CLIP-I", "VIT", "Inception", "KNN", "SVCCA", "SSIM", "CKA", "PSNR"],
    hor_lines=[],
    x_ticks=None,
    y_lim=(0, 1.05),
    legend_args={"show": True, "size": 25, "fontsize": 15},
    label_modifiers={},
    ylabel="Image Alignment",
):
    for k in ["show", "size", "fontsize"]:
        assert k in legend_args, f"Legend argument {k} not found in legend_args"

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
            "axes.facecolor": "#ffffff",
            "figure.facecolor": "#ffffff",
            "grid.color": "white",
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
        }
    )
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.style.use("seaborn-v0_8-paper")

    METRIC_TO_DICT_NAME = {
        "DINO": "dino_cos_sim",
        "CLIP-I": "clip_cos_sim",
        "VIT": "annotator_cos_sim",
        "Inception": "inception_cos_sim",
        "KNN": "knn",
        "SVCCA": "svcca",
        "SSIM": "ssim",
        "CKA": "cka",
        "PSNR": "psnr",
        "CORR": "corr_unnorm",
        "L2_UNNORM": "l2_unnorm",
        "L2_NORM": "l2_norm",
    }
    METRIC_TO_COLOR = {
        "DINO": "#8E51F8",
        "CLIP-I": "lightseagreen",
        "VIT": "mediumblue",
        "Inception": "darkorange",
        "KNN": "darkred",
        "SVCCA": "#FF6464",
        "SSIM": "#DDA217",
        "CKA": "#003049",
        "PSNR": "#FF6464",
        "CORR": "#8E51F8",
        "L2_UNNORM": "darkgreen",
        "L2_NORM": "lightgreen",
    }

    iterations = list(data.keys())
    iterations_int = sorted([int(x) for x in iterations])
    if limit_to is not None:
        iterations_int = [x for x in iterations_int if x <= limit_to]

    for metric_name in used_metrics:
        assert (
            f"{METRIC_TO_DICT_NAME[metric_name]}_mean" in data[str(iterations_int[0])].keys()
        ), f"Metric {metric_name} not found in data"

    means = {
        metric_name: [data[str(it)][f"{METRIC_TO_DICT_NAME[metric_name]}_mean"] for it in iterations_int]
        for metric_name in used_metrics
    }
    stds = {
        metric_name: [data[str(it)][f"{METRIC_TO_DICT_NAME[metric_name]}_std"] for it in iterations_int]
        for metric_name in used_metrics
    }

    fig, axs = plt.subplots(1, 1, figsize=(10, 5))

    if len(hor_lines) > 0:
        # names: ['(A)', '(B)', '(C)', ...]
        names = [f"({chr(65 + i)})" for i in range(len(hor_lines))]
        for hor_line, name in zip(hor_lines, names):
            axs.axvline(x=hor_line, color="black", linestyle="--", linewidth=2, alpha=0.4)
            axs.text(
                hor_line,
                0.06,
                name,
                fontsize=22,
                alpha=1.0,
                ha="center",
                fontfamily="serif",
                bbox=dict(facecolor="#ffffff", edgecolor="none", pad=3.0),
            )

    for metric_name in used_metrics:
        axs.plot(
            iterations_int,
            means[metric_name],
            METRIC_TO_COLOR[metric_name],
            label=metric_name if not label_modifiers.get(metric_name, False) else label_modifiers[metric_name],
            linewidth=3,
            path_effects=[withStroke(linewidth=4, foreground="black")],
        )

        # mean+std cant be above 1, mean-std cant be below 0
        upper_std = np.stack(
            [np.array(means[metric_name]) + np.array(stds[metric_name]), np.ones_like(np.array(means[metric_name]))],
            axis=1,
        ).min(axis=1)
        lower_std = np.stack(
            [np.array(means[metric_name]) - np.array(stds[metric_name]), np.zeros_like(np.array(means[metric_name]))],
            axis=1,
        ).max(axis=1)
        axs.fill_between(iterations_int, lower_std, upper_std, color=METRIC_TO_COLOR[metric_name], alpha=0.2)

    axs.set_xlabel("Train step", fontsize=25)
    axs.set_xlim((0, iterations_int[-1]))
    if x_ticks is not None:
        x_tickslabels = [f"${x_ticks[i] / 10**5:.1f} \cdot 10^5$" for i in range(len(x_ticks))]
        axs.set_xticks(x_ticks)
        axs.set_xticklabels(x_tickslabels, fontsize=15)

    axs.set_ylabel(ylabel, fontsize=25)
    axs.set_ylim(y_lim)
    axs.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    axs.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=15)

    axs.grid(axis="y", linestyle="--", color="grey", alpha=0.3)
    if legend_args["show"]:
        axs.legend(
            fontsize=legend_args["fontsize"],
            loc="center right",
            bbox_to_anchor=(1.0, 0.3),
            ncol=2,
            prop={"size": legend_args["size"]},
        )

    if first_batch_idx is not None:
        axs.axvline(x=first_batch_idx, color="black", linestyle="-", linewidth=1)

    if title is not None:
        plt.suptitle(title, fontsize=35, y=0.99, fontweight="bold")
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, format="pdf")
        plt.close()
    else:
        plt.show()


# %%
path = Path("experiments/trainsteps__lastsample_noising_currentdenoising/imgnet_64_last")
stats_dict = get_full_stats_dict(path=path)

# %% [markdown]
# ### Usage

# %%
# CIFAR-10 | DDPM | Current noises and current denoises
cifar_xticks = [50_000, 250_000, 450_000, 650_000]

path = Path("experiments/trainsteps__lastsample_noising_currentdenoising/cifar10_32_last")
stats_dict = get_full_stats_dict(path=path)

plot_trainstep_alignment(stats_dict, x_ticks=cifar_xticks, used_metrics=["CLIP-I", "DINO", "SVCCA", "SSIM", "CKA"])
# plot_trainstep_alignment(stats_dict, x_ticks=cifar_xticks, used_metrics = ['DINO', 'SVCCA', 'SSIM', 'CKA'], filename="trainstep_alignment_lastsample_currentnoising_currentdenoising_pixelcifar32.pdf")


# %%
# CIFAR-10 | DDPM | Current noises and last denoises
cifar_xticks = [50_000, 250_000, 450_000, 650_000]

path = Path("experiments/trainsteps__lastsample_noising_lastdenoising/cifar10_32_last")
stats_dict = get_full_stats_dict(path=path)

# plot_trainstep_alignment(stats_dict, x_ticks=cifar_xticks, used_metrics = ['CLIP-I','DINO', 'SVCCA', 'SSIM', 'CKA'])
plot_trainstep_alignment(
    stats_dict,
    x_ticks=cifar_xticks,
    used_metrics=["DINO", "SVCCA", "SSIM", "CKA"],
    filename="trainstep_alignment_lastsample_currentnoising_lastdenoising_pixelcifar32.pdf",
)


# %%
# ImageNet64 | DDPM | Current noises and currentdenoises

imagenet_xticks = [100_000, 500_000, 900_000, 1_300_000]
path = Path("experiments/trainsteps__lastsample_noising_currentdenoising/imgnet_64_last")

stats_dict = get_full_stats_dict(path=path)

plot_trainstep_alignment(stats_dict, x_ticks=imagenet_xticks, used_metrics=["CLIP-I", "DINO", "SVCCA", "SSIM", "CKA"])
# plot_trainstep_alignment(stats_dict, x_ticks=imagenet_xticks, used_metrics = ['DINO', 'SVCCA', 'SSIM', 'CKA'], filename="trainstep_alignment_lastsample_currentnoising_currentdenoising_pixelimagenet64.pdf")

# %%
# ImageNet64 | DDPM | Current noises and last denoises

imagenet_xticks = [100_000, 500_000, 900_000, 1_300_000]
path = Path("experiments/trainsteps__lastsample_noising_lastdenoising/imgnet_64_last")

stats_dict = get_full_stats_dict(path=path)

plot_trainstep_alignment(stats_dict, x_ticks=imagenet_xticks, used_metrics=["CLIP-I", "DINO", "SVCCA", "SSIM", "CKA"])
# plot_trainstep_alignment(stats_dict, x_ticks=imagenet_xticks, used_metrics = ['DINO', 'SVCCA', 'SSIM', 'CKA'], filename="trainstep_alignment_lastsample_currentnoising_lastdenoising_pixelimagenet64.pdf")

# %%
imagenet_xticks = [100_000, 500_000, 900_000, 1_300_000]
path = Path("experiments/trainsteps__latents_sim/imgnet_64_last")

stats_dict = get_full_stats_dict(path=path)

for k in stats_dict.keys():
    for kk in stats_dict[k].keys():
        if kk.startswith("psnr"):
            stats_dict[k][kk] = stats_dict[k][kk] / 100

plot_trainstep_alignment(
    stats_dict,
    x_ticks=imagenet_xticks,
    used_metrics=["CKA", "CORR", "SSIM", "PSNR"],
    y_lim=(0, 1.05),
    legend_args={"show": True, "size": 20, "fontsize": 15},
    label_modifiers={"PSNR": r"$PSNR \times 10^{-2}$"},
)
# plot_trainstep_alignment(stats_dict, x_ticks=imagenet_xticks, used_metrics = ['CKA', 'CORR', 'SSIM', 'PSNR'], y_lim=(0,1.05), legend_args={"show": True, "size": 20, "fontsize": 15}, label_modifiers={"PSNR": r"$PSNR \times 10^{-2}$"}, ylabel=r"Latent Alignment", filename="trainstep_alignment_latents_sim_imagenet64.pdf")

# %%
cifar_xticks = [50_000, 250_000, 450_000, 650_000]
path = Path("experiments/trainsteps__latents_sim/cifar10_32_last")

stats_dict = get_full_stats_dict(path=path)

for k in stats_dict.keys():
    for kk in stats_dict[k].keys():
        if kk.startswith("psnr"):
            stats_dict[k][kk] = stats_dict[k][kk] / 100

plot_trainstep_alignment(
    stats_dict,
    x_ticks=cifar_xticks,
    used_metrics=["CKA", "CORR", "SSIM", "PSNR"],
    y_lim=(0, 1.05),
    legend_args={"show": True, "size": 20, "fontsize": 15},
    label_modifiers={"PSNR": r"$PSNR \times 10^{-2}$"},
)
# plot_trainstep_alignment(stats_dict, x_ticks=imagenet_xticks, used_metrics = ['CKA', 'CORR', 'SSIM', 'PSNR'], y_lim=(0,1.05), legend_args={"show": True, "size": 20, "fontsize": 15}, label_modifiers={"PSNR": r"$PSNR \times 10^{-2}$"}, ylabel=r"Latent Alignment", filename="trainstep_alignment_latents_sim_imagenet64.pdf")

# %% [markdown]
# # Plots Noise -> Sample (KNN)

# %%
path = Path("experiments/samples_similarity/imgnet_64")

stats_files_s0 = list(path.glob("stats_s0_stidx*"))
stats_s0 = [load_file(path) for path in stats_files_s0]
merged_s0 = {}
for stat in stats_s0:
    merged_s0.update(stat)

stats_files_s10 = list(path.glob("stats_s10_stidx*"))
stats_s10 = [load_file(path) for path in stats_files_s10]
merged_s10 = {}
for stat in stats_s10:
    merged_s10.update(stat)
stats_files_s42 = list(path.glob("stats_s42_stidx*"))
stats_s42 = [load_file(path) for path in stats_files_s42]
merged_s42 = {}
for stat in stats_s42:
    merged_s42.update(stat)
common_keys_set = common_keys(merged_s0, merged_s10, merged_s42)

average_dict_imgnet = {}
for key in common_keys_set:
    val_s0 = merged_s0[key]
    val_s10 = merged_s10[key]
    val_s42 = merged_s42[key]
    val_key = {k: sum([val_s0[k], val_s10[k], val_s42[k]]) / 3 for k in val_s10.keys()}

    average_dict_imgnet[key] = val_key

# %%
path = Path("experiments/samples_similarity/cifar10_32/")

stats_files_s0 = list(path.glob("stats_s0_stidx*"))
stats_s0 = [load_file(path) for path in stats_files_s0]
merged_s0 = {}
for stat in stats_s0:
    merged_s0.update(stat)

stats_files_s10 = list(path.glob("stats_s10_stidx*"))
stats_s10 = [load_file(path) for path in stats_files_s10]
merged_s10 = {}
for stat in stats_s10:
    merged_s10.update(stat)
stats_files_s42 = list(path.glob("stats_s42_stidx*"))
stats_s42 = [load_file(path) for path in stats_files_s42]
merged_s42 = {}
for stat in stats_s42:
    merged_s42.update(stat)
common_keys_set = common_keys(merged_s0, merged_s10, merged_s42)

average_dict_cifar = {}
for key in common_keys_set:
    val_s0 = merged_s0[key]
    val_s10 = merged_s10[key]
    val_s42 = merged_s42[key]
    val_key = {k: sum([val_s0[k], val_s10[k], val_s42[k]]) / 3 for k in val_s10.keys()}

    average_dict_cifar[key] = val_key

# %%
from matplotlib import pyplot as plt
from matplotlib.patheffects import withStroke


def plot_noise_sample(data_cifar, data_imgnet, limit_to=None, filename=None):
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
            "axes.facecolor": "#ffffff",
            "figure.facecolor": "#ffffff",
            "grid.color": "white",
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
        }
    )
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.style.use("seaborn-v0_8-paper")

    iterations_imgnet = list(data_imgnet.keys())
    iterations_cifar = list(data_cifar.keys())
    iterations_imgnet_int = sorted([int(x) for x in iterations_imgnet])
    iterations_cifar_int = sorted([int(x) for x in iterations_cifar])

    if limit_to is not None:
        iterations_imgnet_int = [x for x in iterations_imgnet_int if x <= limit_to]
        iterations_cifar_int = [x for x in iterations_cifar_int if x <= limit_to]

    knn_means_imgnet = [data_imgnet[str(it)]["knn_mean"] for it in iterations_imgnet_int]
    knn_stds_imgnet = [data_imgnet[str(it)]["knn_std"] for it in iterations_imgnet_int]

    knn_means_cifar = [data_cifar[str(it)]["knn_mean"] for it in iterations_cifar_int]
    knn_stds_cifar = [data_cifar[str(it)]["knn_std"] for it in iterations_cifar_int]

    fig, axs = plt.subplots(1, 1, figsize=(5, 5))

    axs.plot(
        iterations_imgnet_int,
        np.array(knn_means_imgnet),
        "mediumturquoise",
        label="ImageNet",
        linewidth=3,
        path_effects=[withStroke(linewidth=4, foreground="black")],
    )
    axs.fill_between(
        iterations_imgnet_int,
        (np.array(knn_means_imgnet) - np.array(knn_stds_imgnet)),
        (np.array(knn_means_imgnet) + np.array(knn_stds_imgnet)),
        color="mediumturquoise",
        alpha=0.2,
    )

    axs.plot(
        iterations_cifar_int,
        np.array(knn_means_cifar),
        "#003049",
        label="CIFAR-10",
        linewidth=3,
        path_effects=[withStroke(linewidth=4, foreground="black")],
    )
    axs.fill_between(
        iterations_cifar_int,
        (np.array(knn_means_cifar) - np.array(knn_stds_cifar)),
        (np.array(knn_means_cifar) + np.array(knn_stds_cifar)),
        color="#003049",
        alpha=0.2,
    )

    axs.set_xlabel("Training step", fontsize=25)
    axs.set_ylabel("KNN Coverage [%]", fontsize=25)
    axs.set_ylim((0, 110))
    axs.set_xlim((0, iterations_imgnet_int[-1]))
    axs.grid(axis="y", linestyle="--", color="grey", alpha=0.3)
    axs.legend(fontsize=13, loc="center right", bbox_to_anchor=(1.0, 0.82), ncol=1, prop={"size": 20})

    axs.set_xticks([100_000, 500_000, 900_000, 1_300_000])
    axs.set_xticklabels(["$10^5$", "$5 \cdot 10^5$", "$9 \cdot 10^5$", "$13 \cdot 10^5$"], fontsize=15)
    # axs.set_xticks([50_000, 250_000, 450_000, 650_000])
    # axs.set_xticklabels(["$5 \cdot 10^4$", "$2.5 \cdot 10^5$", "$4.5 \cdot 10^5$", "$6.5 \cdot 10^5$"], fontsize=15)

    axs.set_yticks([20, 40, 60, 80, 100])
    axs.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=15)

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, format="pdf")
        plt.close()
    else:
        plt.show()


# %%
plot_noise_sample(average_dict_cifar, average_dict_imgnet, filename="knn_cifar_imgnet.pdf")
# plot_noise_sample(average_dict_cifar, average_dict_imgnet)


# %%
plot_noise_sample(average_dict2, average_dict, indices, filename="accuracies_imagenet64.pdf")


# %%
plot_noise_sample(average_dict2, average_dict, indices, filename="noise_image_imagenet64.pdf")


# %%
path = Path("experiments/samples_similarity/cifar10_32")

stats_files_s0 = list(path.glob("stats_s0_stidx*"))
stats_s0 = [load_file(path) for path in stats_files_s0]
merged_s0 = {}
for stat in stats_s0:
    merged_s0.update(stat)
stats_files_s10 = list(path.glob("stats_s10_stidx*"))
stats_s10 = [load_file(path) for path in stats_files_s10]
merged_s10 = {}
for stat in stats_s10:
    merged_s10.update(stat)
stats_files_s42 = list(path.glob("stats_s42_stidx*"))
stats_s42 = [load_file(path) for path in stats_files_s42]
merged_s42 = {}
for stat in stats_s42:
    merged_s42.update(stat)
common_keys_set = common_keys(merged_s0, merged_s10, merged_s42)

average_dict = {}
for step_key_idx in common_keys_set:
    val_s0 = merged_s0[step_key_idx]
    val_s10 = merged_s10[step_key_idx]
    val_s42 = merged_s42[step_key_idx]

    val_key = {}
    for k in val_s10.keys():
        vals = np.stack(
            [
                val_s0[k].numpy() if isinstance(val_s0[k], torch.Tensor) else val_s0[k],
                val_s10[k].numpy() if isinstance(val_s10[k], torch.Tensor) else val_s10[k],
                val_s42[k].numpy() if isinstance(val_s42[k], torch.Tensor) else val_s42[k],
            ]
        )
        if k.endswith("mean"):
            val_key[k] = np.mean(vals)
        elif k.endswith("std"):
            val_key[k] = np.sqrt(np.mean(vals**2))
        elif k in ["svcca", "cka"]:
            val_key[f"{k}_mean"] = np.mean(vals)
            val_key[f"{k}_std"] = np.std(vals)

    average_dict[step_key_idx] = val_key


# %%
path = Path("experiments/samples_similarity/cifar10_32")

stats_files_s0 = list(path.glob("stats_s0_stidx*"))
stats_s0 = [load_file(path) for path in stats_files_s0]
merged_s0 = {}
for stat in stats_s0:
    merged_s0.update(stat)

stats_files_s10 = list(path.glob("stats_s10_stidx*"))
stats_s10 = [load_file(path) for path in stats_files_s10]
merged_s10 = {}
for stat in stats_s10:
    merged_s10.update(stat)
stats_files_s42 = list(path.glob("stats_s42_stidx*"))
stats_s42 = [load_file(path) for path in stats_files_s42]
merged_s42 = {}
for stat in stats_s42:
    merged_s42.update(stat)
common_keys_set = common_keys(merged_s0, merged_s10, merged_s42)

average_dict2 = {}
for key in common_keys_set:
    val_s0 = merged_s0[key]
    val_s10 = merged_s10[key]
    val_s42 = merged_s42[key]
    val_key = {k: sum([val_s0[k], val_s10[k], val_s42[k]]) / 3 for k in val_s10.keys()}

    average_dict2[key] = val_key

# %% [markdown]
# # Examples

# %%
checkpoints = IMAGENET_CKPTS[:: len(IMAGENET_CKPTS) // 12]


# %%
from pathlib import Path

recons_path = Path(
    "experiments/trainsteps__lastsample_noising_currentdenoising/imgnet_64_last/s10/reconstructed_samples"
)
samples_path = Path("experiments/trainsteps/imgnet_64/s10/samples")
final_samples_path = Path(
    "experiments/trainsteps__lastsample_noising_currentdenoising/imgnet_64_last/s10/samples/1481051.pt"
)

# %%
recons_samples = []
all_samples = []
for ckpt in checkpoints:
    recons_ckpt = torch.load((recons_path / f"{ckpt}.pt").resolve())
    samples_ckpt = torch.load((samples_path / f"{ckpt}.pt").resolve())

    how_many = recons_ckpt.shape[0] // 19

    recons_samples.append(recons_ckpt[::how_many])
    all_samples.append(samples_ckpt[::how_many])

final_samples = torch.load(final_samples_path)[::how_many]

# %%
rows = [0, 1, 5, 8, 9, 11, 14, 16]

# %%
grid = torch.stack(recons_samples)
grid = grid.permute(1, 0, 2, 3, 4)
grid = grid[rows]
n_rows, n_cols = grid.shape[0], grid.shape[1]
grid = grid.reshape(n_rows * n_cols, 3, 64, 64)
grid_tensors(grid, rows=n_rows)

# %%
grid = torch.stack(all_samples)
grid = grid.permute(1, 0, 2, 3, 4)
grid = grid[rows]
n_rows, n_cols = grid.shape[0], grid.shape[1]
grid = grid.reshape(n_rows * n_cols, 3, 64, 64)
grid_tensors(grid, rows=n_rows)

# %%
grid_tensors(final_samples[rows], rows=len(rows))

# %%


# %%
from pathlib import Path

import torch

path = Path("experiments/outputs_per_T2/imagenet_dit_256/T_100")
noises = []
latents = []
samples = []
samples2 = []
for class_idx in range(16):
    noise = torch.load(path / f"class_idx_{class_idx}/noise.pt", map_location="cpu")
    latent = torch.load(path / f"class_idx_{class_idx}/latents.pt", map_location="cpu")
    latent = torch.stack(latent)
    sample = torch.load(path / f"class_idx_{class_idx}/samples.pt", map_location="cpu")
    sample = torch.stack(sample)
    sample2 = torch.load(path / f"class_idx_{class_idx}/samples2.pt", map_location="cpu")
    sample2 = torch.stack(sample2)

    noises.append(noise)
    latents.append(latent)
    samples.append(sample)
    samples2.append(sample2)

# %%
noises = torch.cat(noises, dim=0)
latents = torch.cat(latents, dim=0)
samples = torch.cat(samples, dim=0)
samples2 = torch.cat(samples2, dim=0)

print(noises.shape, latents.shape, samples.shape, samples2.shape)

torch.save(noises, path / "noise.pt")
torch.save(latents, path / "latents.pt")
torch.save(samples, path / "samples.pt")
torch.save(samples2, path / "samples2.pt")


# %%
