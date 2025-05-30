import matplotlib.pyplot as plt
import torch
import torch.distributions as dist
from scipy.stats import shapiro
from tqdm import tqdm


def normal_dist_test(dist_tens):
    flattened_data = dist_tens.flatten().numpy()

    _, shapiro_p_value = shapiro(flattened_data)

    return shapiro_p_value


def kl_div(dist1: torch.Tensor, dist2: torch.Tensor):
    dist1_mu = torch.tensor([dist1.mean()])
    dist1_sigma = torch.tensor([torch.std(dist1)])

    dist2_mu = torch.tensor([dist2.mean()])
    dist2_sigma = torch.tensor([torch.std(dist2)])

    P = dist.Normal(dist1_mu, dist1_sigma)
    Q = dist.Normal(dist2_mu, dist2_sigma)

    kl_divergence = dist.kl_divergence(P, Q)
    return kl_divergence.item()

def kl_div2(noise: torch.Tensor, latents: torch.Tensor):
    noise_mu = noise.mean(dim=0)
    noise_sigma = noise.std(dim=0) + 1e-6  # Avoid zero variance

    latents_mu = latents.mean(dim=0)
    latents_sigma = latents.std(dim=0) + 1e-6

    P = dist.Normal(noise_mu, noise_sigma)
    Q = dist.Normal(latents_mu, latents_sigma)

    kl_divergence = dist.kl_divergence(P, Q).sum()  # Sum over (C, W, H)

    # Normalize by total number of dimensions per sample
    kl_divergence = kl_divergence / (noise.shape[1] * noise.shape[2] * noise.shape[3])

    return kl_divergence.item()



def stat_test_normality(latents, limit=5_000, p_val=0.05):
    n_tests = latents.shape[0]
    n_rej = 0
    n_acc = 0
    for lat_idx in tqdm(range(n_tests)):
        lat = latents[lat_idx].flatten()
        n_latent_samples = min(lat.shape[0], limit)
        indices = torch.randint(0, lat.shape[0], (n_latent_samples,))
        test_p = normal_dist_test(lat[indices])
        if test_p > p_val:
            n_acc += 1
        else:
            n_rej += 1
    return {"n_rej": n_rej, "%_rej": n_rej / n_tests, "n_acc": n_acc, "%_acc": n_acc / n_tests}


def plt_qq(noise_tens, lat_tens, ds_name, t, diff_type, path=None):
    import statsmodels.api as sm

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    sm.qqplot_2samples(noise_tens.cpu().flatten().numpy(), lat_tens.cpu().flatten().numpy(), ax=ax, line="s")
    plt.title(f"{diff_type} | {ds_name} | T={t}", fontsize=25)
    plt.xlabel("Noise", fontsize=20)
    plt.ylabel(f"Latent ($T={t}$)", fontsize=20)
    plt.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
    if path is None:
        plt.show()
    else:
        plt.savefig(path)


def stats_from_tensor(tensor: torch.Tensor):
    return {"mean": tensor.mean().item(), "std": tensor.std().item()}
