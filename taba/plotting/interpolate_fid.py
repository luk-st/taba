# %%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patheffects import withStroke


def plot_outs(outs: dict, x_keys: list, ylabel: str, yticklabels: list = None, filename=None):
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

    for idx, key in enumerate(outs.keys()):
        plt.plot(
            x_keys,
            outs[key],
            label=key,
            color=colors[idx],
            linewidth=3,
            path_effects=[withStroke(linewidth=4, foreground="black")],
        )

    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2),
        ncol=2,
        fontsize="x-large",
        title=r"Generations from:",
        title_fontsize="x-large",
    )
    plt.xlabel(r"Spherical Interpolation Step $\alpha$", fontsize=32)
    plt.ylabel(ylabel, fontsize=32)
    plt.xlim(0, x_keys[-1])

    plt.xticks(list(range(len(alphas))), alphas, fontsize=20, rotation=45)
    if yticklabels is not None:
        plt.yticks(yticklabels, fontsize=20, rotation=45)

    if filename is not None:
        plt.savefig(filename, format="pdf", bbox_inches="tight")
    plt.show()


# %%
noise_sample_imagenet = [
    29.210176956283192,
    29.203875791942266,
    28.945937579207055,
    28.758558735086865,
    28.890673197251772,
    28.887953822145676,
    28.88136726744665,
]

latent_recon_imagenet = [
    32.469856507431416,
    33.73865617452992,
    36.29457061028222,
    37.39150131438669,
    36.16094708411157,
    33.61086974828788,
    32.08472819004055,
]

noise_sample_cifar = [
    9.513232969589353,
    9.434378799749368,
    9.377070523692225,
    9.562095804815442,
    9.278137503884295,
    9.305495632064776,
    9.173232712054585,
]

latent_recon_cifar = [
    15.18492830802927,
    19.375250795667853,
    25.1382410215283,
    27.87927846552202,
    25.43600389851025,
    19.231847333309133,
    14.864657125415249,
]

alphas = [f"{x:.2f}" for x in [0 / 6, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1]]

# %%
plot_outs(
    outs={r"Noise $x^T$": noise_sample_imagenet, r"Latent $\hat{x}^T$": latent_recon_imagenet},
    x_keys=alphas,
    ylabel=r"FID (to ImageNet $64\times64$)",
    yticklabels=[30, 32, 34, 36],
)


# %%
plot_outs(
    outs={r"Noise $x^T$": noise_sample_cifar, r"Latent $\hat{x}^T$": latent_recon_cifar},
    x_keys=alphas,
    ylabel=r"FID (to CIFAR $32\times32$)",
    yticklabels=[10, 15, 20, 25],
)
