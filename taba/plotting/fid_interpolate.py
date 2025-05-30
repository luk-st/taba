import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patheffects import withStroke


def plot_fid_interpolate(
    models_outs: dict, ylabel: str, yticklabels: list = None, filename=None, use_log=False, ncols=None
):
    # create subplots
    fig, axs = plt.subplots(ncols=len(models_outs), nrows=1, figsize=(12, 3))

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

    colors = ["#DDA217", "#8E51F8", "#003049", "#DDA217", "#000000"]

    temp_outs = models_outs[list(models_outs.keys())[0]]
    alphas_str = [alpha for alpha in temp_outs[list(temp_outs.keys())[0]].keys()]
    alphas_str = sorted(alphas_str, key=lambda x: float(x))
    alphas = [float(alpha) for alpha in alphas_str]
    alphas_xticks = [alphas[0], alphas[len(alphas) // 2], alphas[-1]]

    for idx_models, (model_name, outs) in enumerate(models_outs.items()):
        all_model_vals = []
        for idx_type, key in enumerate(outs.keys()):
            v = outs[key].values()
            v = [np.log(val) for val in v] if use_log else v
            all_model_vals.extend(v)
            axs[idx_models].plot(
                alphas,
                v,
                label=f"{key}",
                color=colors[idx_type],
                linewidth=4,
                path_effects=[withStroke(linewidth=5, foreground="black")],
            )

        min_val = min(all_model_vals)
        max_val = max(all_model_vals)
        diff = max_val - min_val
        half_diff = diff / 2
        axs[idx_models].set_ylim(min_val - diff * 0.1, max_val + diff * 0.1)
        yticks = [min_val, min_val + half_diff, max_val]
        yticks_str = [f"{val:.1f}" for val in yticks]
        axs[idx_models].set_yticks(yticks, yticks_str, fontsize=12)

        if idx_models == 0:
            axs[idx_models].set_ylabel(ylabel, fontsize=20)
        if len(models_outs) % 2 == 1 and idx_models == len(models_outs) // 2:
            axs[idx_models].set_xlabel(r"SLERP $\alpha$", fontsize=20)
        axs[idx_models].set_xlim(0, 1)
        axs[idx_models].set_xticks(alphas_xticks, alphas_xticks, fontsize=12, rotation=0)
        # axs[idx_models].legend(loc="upper center", bbox_to_anchor=(0.5, 1.45), ncol=1, fontsize="large", title=model_name, title_fontsize="large")
        axs[idx_models].set_title(fontsize="large", label=model_name, pad=5)
        # if yticklabels is not None:
        #     axs[idx_models, idx_type].set_yticks(yticklabels, fontsize=20, rotation=45)
        # else:
        #     axs[idx_models, idx_type].set_yticks(fontsize=20, rotation=45)

    # axs[0].legend(loc="upper center", bbox_to_anchor=(3, 1.75), ncol=2, fontsize="large", title="Interpolation", title_fontsize="large")
    axs[0].legend(loc="upper center", bbox_to_anchor=(3, 1.38), ncol=2, fontsize="medium")

    if filename is not None:
        plt.savefig(filename, format="pdf", bbox_inches="tight")
        plt.close()
    else:
        plt.show()


T = 50
outs_per_model = {}
model_name_to_abbr = {
    "imagenet_pixel_64": "ADM-64",
    "imagenet_pixel_256": "ADM-256",
    "cifar_pixel_32": "ADM-32",
    "imagenet_dit_256": "DIT",
    "celeba_ldm_256": "LDM",
}
for model_name in ["cifar_pixel_32", "imagenet_pixel_64", "imagenet_pixel_256", "celeba_ldm_256", "imagenet_dit_256"]:
    path = Path(f"experiments/fid_results/s10_42_s420_100/{model_name}/T{T}")
    with open((path / "noise" / "fid_scores.json").resolve(), "r") as f:
        noise_fid = json.load(f)
    with open((path / "latent" / "fid_scores.json").resolve(), "r") as f:
        latent_fid = json.load(f)
    outs_per_model[model_name_to_abbr[model_name]] = {
        r"Noise ($x^T$) interpolation": noise_fid["outputs"],
        r"Latent ($\hat{x}^T$) interpolation": latent_fid["outputs"],
    }

plot_fid_interpolate(outs_per_model, ylabel="FID (to trainset)", use_log=False, filename="fid_results.pdf")
