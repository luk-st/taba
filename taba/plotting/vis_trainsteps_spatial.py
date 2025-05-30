# %% [markdown]
# # Imports, etc.

# %%
import os

os.chdir("<PROJECT_PATH>")

# %%
import torch

# %%
from taba.models.adm.ddim import generate_latents, generate_noises, generate_samples
from taba.utils import plot_diffusion

# %%
IMAGENET_CKPT = sorted(
    list(
        set(
            [0, 5, 10, 25, 50, 100]
            + list(range(0, 10_308, 250))
            + list(range(0, 522_500, 10_307))
            + list(range(522_500, 1_130_613, 10_307))
            + list(range(1_130_613, 1_500_001, 10_307))
            + list(range(0, 1_130_613, 2_500))
            + list(range(1_130_613, 1_500_001, 2_500))
        )
    )
)

CIFAR_CKPT = sorted(
    list(
        set(
            [0, 5, 10, 25]
            + list(range(50, 401, 50))
            + list(range(0, 100 * 390 + 1, 390))
            + list(range(101 * 390, 427_501, 5 * 390))
            + list(range(0, 427_501, 2_500))
            + list(range(500_000, 702_500, 2_500))
        )
    )
)

# %%
# IMAGENET_CKPT = sorted(
#     list(
#         set(
#             [0, 25, 100]
#             + list(range(0, 10_308, 2500))
#             + list(range(0, 522_500, 2 * 10_307))
#             + list(range(522_500, 1_130_613, 2*10_307))
#             + list(range(1_130_613, 1_500_613, 2* 10_307))
#             # + list(range(0, 1_130_613, 2_500))
#             # + list(range(1_130_613, 1_500_613, 2_500))
#         )
#     )
# )


# CIFAR_CKPT = sorted(
#     list(
#         set(
#             [0, 25] # 2
#             + list(range(50, 401, 100)) # 4
#             + list(range(0, 100 * 390 + 1, 5 * 390)) # 20
#             + list(range(101 * 390, 429390, 25 * 390)) # 40
#             + list(range(425000, 700_000, 5_000)) # 55
#         )
#     )
# )

# %%
from pathlib import Path

path = Path("experiments/trainsteps/imgnet_64/s0/samples")

# %%
# from pathlib import Path

# path = Path("experiments/trainsteps/imgnet_64_last/s0/samples")

# %%
import torch


# %%
checkpoints = CIFAR_CKPT[::18]
samples = []
for ckpt in checkpoints:
    sample_ckpt = torch.load((path / f"{ckpt}.pt").resolve())
    how_many = sample_ckpt.shape[0] // 14
    samples.append(sample_ckpt[::how_many])

# %%
grid = torch.stack(samples)
grid = grid.permute(1,0,2,3,4)
n_rows, n_cols = grid.shape[0], grid.shape[1]
grid = grid.reshape(n_rows * n_cols, 3, 64, 64)
plot_diffusion(grid, nrow=n_cols)

# %%
path = Path("experiments/trainsteps/cifar10_32/s0/samples")
samples = []
checkpoints = CIFAR_CKPT[::18]
for ckpt in checkpoints:
    sample_ckpt = torch.load((path / f"{ckpt}.pt").resolve())
    how_many = sample_ckpt.shape[0] // 14
    samples.append(sample_ckpt[::how_many])

# %%
grid = torch.stack(samples)
grid = grid.permute(1,0,2,3,4)
n_rows, n_cols = grid.shape[0], grid.shape[1]
grid = grid.reshape(n_rows * n_cols, 3, 32, 32)
plot_diffusion(grid, nrow=n_cols)

# %%
path = Path("experiments/trainsteps/cifar10_32/s0/samples")
samples = []
checkpoints = CIFAR_CKPT[::18]
for ckpt in checkpoints:
    sample_ckpt = torch.load((path / f"{ckpt}.pt").resolve())
    samples.append(sample_ckpt[:5])

# %%
grid = torch.stack(samples)
grid = grid.permute(1,0,2,3,4)
n_rows, n_cols = grid.shape[0], grid.shape[1]
grid = grid.reshape(n_rows * n_cols, 3, 32, 32)
plot_diffusion(grid, nrow=n_cols)

# %%
path = Path("experiments/trainsteps/cifar10_32/s10/samples")
samples = []
checkpoints = CIFAR_CKPT[::18]
for ckpt in checkpoints:
    sample_ckpt = torch.load((path / f"{ckpt}.pt").resolve())
    samples.append(sample_ckpt[:5])

grid = torch.stack(samples)
grid = grid.permute(1,0,2,3,4)
n_rows, n_cols = grid.shape[0], grid.shape[1]
grid = grid.reshape(n_rows * n_cols, 3, 32, 32)
plot_diffusion(grid, nrow=n_cols)

# %%
path = Path("experiments/trainsteps/imgnet_64/s0/samples")
samples = []
checkpoints = IMAGENET_CKPT[::30]
for ckpt in checkpoints:
    sample_ckpt = torch.load((path / f"{ckpt}.pt").resolve())
    samples.append(sample_ckpt[:5])

grid = torch.stack(samples)
grid = grid.permute(1,0,2,3,4)
n_rows, n_cols = grid.shape[0], grid.shape[1]
grid = grid.reshape(n_rows * n_cols, 3, 64, 64)
plot_diffusion(grid, nrow=n_cols)

# %%
path = Path("experiments/trainsteps/imgnet_64/s10/samples")
samples = []
checkpoints = IMAGENET_CKPT[::30]
for ckpt in checkpoints:
    sample_ckpt = torch.load((path / f"{ckpt}.pt").resolve())
    samples.append(sample_ckpt[:5])

grid = torch.stack(samples)
grid = grid.permute(1,0,2,3,4)
n_rows, n_cols = grid.shape[0], grid.shape[1]
grid = grid.reshape(n_rows * n_cols, 3, 64, 64)
plot_diffusion(grid, nrow=n_cols)

# %%
path = Path("experiments/trainsteps/imgnet_64/s42/samples")
samples = []
checkpoints = IMAGENET_CKPT[::30]
for ckpt in checkpoints:
    sample_ckpt = torch.load((path / f"{ckpt}.pt").resolve())
    samples.append(sample_ckpt[:5])

grid = torch.stack(samples)
grid = grid.permute(1,0,2,3,4)
n_rows, n_cols = grid.shape[0], grid.shape[1]
grid = grid.reshape(n_rows * n_cols, 3, 64, 64)
plot_diffusion(grid, nrow=n_cols)

# %% [markdown]
# # STATS

# %%
import pickle
from pathlib import Path
import numpy as np

def load_file(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


# %%
path = Path("experiments/trainsteps/cifar10_32_last")

stats_files_s0 = list(path.glob("stats_s0_stidx*"))
stats_s0 = [
    load_file(path) for path in stats_files_s0
]
merged_s0 = {}
for stat in stats_s0:
    merged_s0.update(stat)

# %%

stats_files_s10 = list(path.glob("stats_s10_stidx*"))
stats_s10 = [
    load_file(path) for path in stats_files_s10
]
merged_s10 = {}
for stat in stats_s10:
    merged_s10.update(stat)

# %%

stats_files_s42 = list(path.glob("stats_s42_stidx*"))
stats_s42 = [
    load_file(path) for path in stats_files_s42
]
merged_s42 = {}
for stat in stats_s42:
    merged_s42.update(stat)

# %%
def common_keys(dict1, dict2, dict3):
    return set(dict1.keys()) & set(dict2.keys()) & set(dict3.keys())

common_keys_set = common_keys(merged_s0, merged_s10, merged_s42)
print("Common keys in all three dictionaries:", common_keys_set)


# %%


# %%
average_dict = {}
for key in common_keys_set:
    val_s0 = merged_s0[key]
    val_s10 = merged_s10[key]
    val_s42 = merged_s42[key]
    val_key = {
        k: sum([val_s0[k], val_s10[k], val_s42[k]]) / 3
        for k in val_s10.keys() if k != "mean_mse_img_img2"
    }

    average_dict[key] = val_key



# %%
from matplotlib import pyplot as plt 
from matplotlib.patheffects import withStroke
import math


def plot(data, first_batch_idx = None, limit_to = None, filename = None):
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'font.size' : 15,
        'axes.labelsize': 15,
        'xtick.labelsize':12,
        'ytick.labelsize':12,
        'legend.fontsize': 12,
        'lines.linewidth':2,
        'text.usetex': False,
        'pgf.rcfonts': False,
        "axes.facecolor": "#ffffff",
        "figure.facecolor": "#ffffff",
        "grid.color": "white",
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
    })
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.style.use("seaborn-v0_8-paper")

    iterations = list(data.keys())
    iterations_int = sorted([int(x) for x in iterations])
    if limit_to is not None:
        iterations_int = [x for x in iterations_int if x<=limit_to]
    mean_cossim_img2noise_img2lat = [data[str(it)]['mean_cossim_img2noise_img2lat'] for it in iterations_int]

    mean_cossim_img2noise_lat2noise = [data[str(it)]['mean_cossim_img2noise_lat2noise'] for it in iterations_int]
    angle_sample = np.array([
        math.degrees(math.acos(x)) for x in mean_cossim_img2noise_lat2noise
    ])
    angle_noise = np.array([
        math.degrees(math.acos(x)) for x in mean_cossim_img2noise_img2lat
    ])
    angle_latent = 180 - angle_sample - angle_noise


    mean_mse_img_lat = [data[str(it)]['mean_mse_img_lat'] for it in iterations_int]
    print(mean_mse_img_lat)
    mean_mse_img_noise = [data[str(it)]['mean_mse_img_noise'] for it in iterations_int]
    mean_mse_noise_lat = [data[str(it)]['mean_mse_noise_lat'] for it in iterations_int]

    closest_noise_to_sample_acc = [data[str(it)]['closest_noise_to_sample_acc'] for it in iterations_int]
    closest_sample_to_noise_acc = [data[str(it)]['closest_sample_to_noise_acc'] for it in iterations_int]

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    axs[0].plot(iterations_int, angle_sample, '#FF6464', label=r'$\angle x^0$', linewidth=3, path_effects=[withStroke(linewidth=4, foreground='black')])
    axs[0].plot(iterations_int, angle_noise, '#DDA217', label=r'$\angle x^T$', linewidth=3, path_effects=[withStroke(linewidth=4, foreground='black')])
    axs[0].plot(iterations_int, angle_latent, '#8E51F8', label=r'$\angle \hat{x}^T$', linewidth=3, path_effects=[withStroke(linewidth=4, foreground='black')])
    # axs[0].set_title('Angles', fontsize=25, pad=35)
    axs[0].set_xlabel('Training step', fontsize=25)
    axs[0].set_ylabel('Angle [$^{\circ}$]', fontsize=25)

    axs[0].set_yticks([20, 40, 60,80,100])
    axs[0].set_yticklabels(["$20$", "$40$", "$60$", "$80$", "$100$"], fontsize=15)
    axs[0].set_ylim((0,120))
    axs[0].set_xlim((0,iterations_int[-1]))
    axs[0].set_xticks([50_000, 250_000, 450_000, 650_000])
    axs[0].set_xticklabels(["$5 \cdot 10^4$", "$2.5 \cdot 10^5$", "$4.5 \cdot 10^5$", "$6.5 \cdot 10^5$"], fontsize=15)

    axs[0].grid(axis='y', linestyle='--', color='grey', alpha=0.5)
    axs[0].legend(fontsize=13, loc='center right', bbox_to_anchor=(0.8, 1.13), ncol=3, prop={ 'size': 20})

    axs[1].plot(iterations_int, mean_mse_noise_lat, 'mediumturquoise', label=r'$||x^T-\hat{x}^T||_2$', linewidth=3, path_effects=[withStroke(linewidth=4, foreground='black')])
    axs[1].plot(iterations_int, mean_mse_img_lat, '#003049',label=r'$||x^0-\hat{x}^T||_2$', linewidth=3, path_effects=[withStroke(linewidth=4, foreground='black')])
    axs[1].set_xlabel('Training step', fontsize=25)
    axs[1].set_ylabel('$L2$ Distance', fontsize=25)
    axs[1].set_xlim((0,iterations_int[-1]))

    axs[1].set_yticks([0.1, 0.4, 0.8,1.1])
    axs[1].set_yticklabels(["$0.1$", "$0.4$", "$0.8$", "$1.1$"], fontsize=15)
    axs[1].set_xticks([50_000, 250_000, 450_000, 650_000])
    axs[1].set_xticklabels(["$5 \cdot 10^4$", "$2.5 \cdot 10^5$", "$4.5 \cdot 10^5$", "$6.5 \cdot 10^5$"], fontsize=15)

    axs[1].grid(axis='y', linestyle='--', color='grey', alpha=0.5)
    axs[1].set_ylim((0,1.2))
    axs[1].legend(fontsize=13, loc='center right', bbox_to_anchor=(0.8, 1.15), ncol=2, prop={ 'size': 20})



    if first_batch_idx is not None:
        axs[0].axvline(x=first_batch_idx, color='black', linestyle='-', linewidth=1)
        axs[1].axvline(x=first_batch_idx, color='black', linestyle='-', linewidth=1)
        # axs[2].axvline(x=first_batch_idx, color='black', linestyle='-', linewidth=1)


    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


# %%
plot(average_dict,filename="angles_cifar.pdf")

# %%
path = Path("experiments/trainsteps/cifar10_32")

stats_files_s0 = list(path.glob("stats_s0_stidx*"))
stats_s0 = [
    load_file(path) for path in stats_files_s0
]
merged_s0 = {}
for stat in stats_s0:
    merged_s0.update(stat)


stats_files_s10 = list(path.glob("stats_s10_stidx*"))
stats_s10 = [
    load_file(path) for path in stats_files_s10
]
merged_s10 = {}
for stat in stats_s10:
    merged_s10.update(stat)

stats_files_s42 = list(path.glob("stats_s42_stidx*"))
stats_s42 = [
    load_file(path) for path in stats_files_s42
]
merged_s42 = {}
for stat in stats_s42:
    merged_s42.update(stat)


common_keys_set = common_keys(merged_s0, merged_s10, merged_s42)

average_dict = {}
for key in common_keys_set:
    val_s0 = merged_s0[key]
    val_s10 = merged_s10[key]
    val_s42 = merged_s42[key]
    val_key = {
        k: sum([val_s0[k], val_s10[k], val_s42[k]]) / 3
        for k in val_s10.keys()
    }

    average_dict[key] = val_key


# %%
plot(average_dict)


