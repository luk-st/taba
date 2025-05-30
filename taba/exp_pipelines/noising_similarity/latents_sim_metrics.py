import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import cdist
from torchvision import transforms
from tqdm import tqdm

from taba.metrics.correlation import get_top_k_corr
from taba.metrics.angles_distances import l2_dist_mean_std
from taba.metrics.alignment import calculate_psnr, calculate_ssim, cka, svcca

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


N_SAMPLES = 2048
BATCH_SIZE = 128
T = 100
N_NEIGHBOURS = 50


def tensors_to_tensors(tensors):
    tensors = tensors - tensors.min()
    tensors = tensors / tensors.max()
    return tensors


def tensors_to_pils(tensors):
    tensors = tensors - tensors.min()
    tensors = tensors / tensors.max()
    return [transforms.ToPILImage()(i_tens) for i_tens in tensors]


def calculate_features(samples: torch.Tensor):
    features = {}
    img_samples = tensors_to_tensors(samples)
    dist = torch.from_numpy(cdist(img_samples.flatten(1, 3), img_samples.flatten(1, 3)))
    img_knn = dist.topk(N_NEIGHBOURS + 1, largest=False)[1][:, 1:]
    features["knn"] = img_knn
    features["cka"] = img_samples.flatten(1, 3)
    features["svcca"] = img_samples.flatten(1, 3)
    features["ssim"] = img_samples
    features["psnr"] = img_samples
    features["l2_unnorm"] = samples
    features["l2_norm"] = img_samples
    features["corr_unnorm"] = samples
    features["corr_norm"] = img_samples

    return features


def process_features(features_to_compare, features_ckpt):
    # knn
    to_compare_knn, ckpt_knn = features_to_compare["knn"], features_ckpt["knn"]
    knn_in_num = []
    for i in range(len(ckpt_knn)):
        knn_in_num.append(torch.isin(ckpt_knn[i], to_compare_knn[i]).sum().item())
    knn_mean, knn_std = np.array(knn_in_num).mean(), np.array(knn_in_num).std()

    # cka
    cka_value = cka(features_to_compare["cka"], features_ckpt["cka"])

    # svcca
    svcca_value = svcca(features_to_compare["svcca"], features_ckpt["svcca"])

    # ssim, psnr
    ssims, psnrs = [], []
    for img_1, img_2 in zip(features_to_compare["ssim"], features_ckpt["ssim"]):
        ssims.append(calculate_ssim(img_1.permute(1, 2, 0).numpy(), img_2.permute(1, 2, 0).numpy()))
        psnrs.append(calculate_psnr(img_1.permute(1, 2, 0).numpy(), img_2.permute(1, 2, 0).numpy()))
    ssim_mean = np.array(ssims).mean()
    ssim_std = np.array(ssims).std()
    psnr_mean = np.array(psnrs).mean()
    psnr_std = np.array(psnrs).std()

    # correlation
    corr_unnorm = get_top_k_corr(features_ckpt["corr_unnorm"], top_k=10)
    corr_norm = get_top_k_corr(features_ckpt["corr_norm"], top_k=10)

    # l2 distance to last
    l2_unnorm_mean, l2_unnorm_std = l2_dist_mean_std(features_to_compare["l2_unnorm"], features_ckpt["l2_unnorm"])
    l2_norm_mean, l2_norm_std = l2_dist_mean_std(features_to_compare["l2_norm"], features_ckpt["l2_norm"])

    return {
        "knn_mean": knn_mean,
        "knn_std": knn_std,
        "svcca": svcca_value,
        "ssim_mean": ssim_mean,
        "ssim_std": ssim_std,
        "psnr_mean": psnr_mean,
        "psnr_std": psnr_std,
        "cka": cka_value,
        "corr_unnorm_mean": corr_unnorm["mean"],
        "corr_unnorm_std": corr_unnorm["std"],
        "corr_norm_mean": corr_norm["mean"],
        "corr_norm_std": corr_norm["std"],
        "l2_unnorm_mean": l2_unnorm_mean,
        "l2_unnorm_std": l2_unnorm_std,
        "l2_norm_mean": l2_norm_mean,
        "l2_norm_std": l2_norm_std,
    }


def load_stuff(model: str, seed: int, st_idx: int = 0):

    if model == "imagenet64":
        work_dir = Path(f"experiments/trainsteps__latents_sim/imgnet_64_last").resolve()
        latents_dir = Path(f"experiments/trainsteps__lastsample_noising_lastdenoising/imgnet_64_last").resolve()
        trainsteps_to_sample = sorted(
            list(
                set(
                    [0, 25, 100]
                    + list(range(0, 10_308, 2500))
                    + list(range(0, 522_500, 2 * 10_307))
                    + list(range(522_500, 1_130_613, 2 * 10_307))
                    + list(range(1_130_613, 1481051, 2 * 10_307))
                )
            )
        )
        checkpoint_idx_to_compare = 1_481_051
    elif model == "cifar32":
        work_dir = Path(f"experiments/trainsteps__latents_sim/cifar10_32_last").resolve()
        latents_dir = Path(f"experiments/trainsteps__lastsample_noising_lastdenoising/cifar10_32_last").resolve()
        trainsteps_to_sample = sorted(
            list(
                set(
                    [0, 25]  # 2
                    + list(range(50, 401, 100))  # 4
                    + list(range(0, 100 * 390 + 1, 5 * 390))  # 20
                    + list(range(101 * 390, 429390, 25 * 390))  # 40
                    + list(range(425000, 695_000, 5_000))  # 55
                )
            )
        )
        checkpoint_idx_to_compare = 695_000
    else:
        raise NotImplementedError(f"Unknown setup: {model}")

    latents_dir = (latents_dir / f"s{seed}/latents").resolve()
    os.makedirs(work_dir, exist_ok=True)
    stats_output_file = (work_dir / f"stats_s{seed}_stidx{st_idx}.pkl").resolve()

    return stats_output_file, trainsteps_to_sample, checkpoint_idx_to_compare, latents_dir


def save_stats_to_file(dct: dict, stats_output_file: str):
    with open(stats_output_file, "wb") as file:
        pickle.dump(dct, file)


def process(model: str, seed: int, start_idx: int = None, stop_idx: int = None):
    print(f"INFO | Processing model {model} with seed {seed} from {start_idx} to {stop_idx}")
    stats_output_file, trainsteps_to_sample, checkpoint_idx_to_compare, latents_dir = load_stuff(
        model=model, seed=seed, st_idx=start_idx or 0
    )

    ranges = trainsteps_to_sample
    if stop_idx is not None:
        ranges = [r for r in ranges if r <= stop_idx]
    if start_idx is not None:
        ranges = [r for r in ranges if r >= start_idx]
    loop = tqdm(ranges, total=len(ranges))

    all_steps_metrics = {}
    latents_to_compare = torch.load(latents_dir / f"{checkpoint_idx_to_compare}.pt", weights_only=False)
    feats_to_compare = calculate_features(latents_to_compare)

    for checkpoint_step in loop:
        ckpt_latents = torch.load(latents_dir / f"{checkpoint_step}.pt", weights_only=False)
        ckpt_feats = calculate_features(ckpt_latents)

        ckpt_stats = process_features(feats_to_compare, ckpt_feats)

        all_steps_metrics[str(checkpoint_step)] = ckpt_stats
        save_stats_to_file(all_steps_metrics, stats_output_file=stats_output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process train steps.")
    parser.add_argument("-m", "--model", type=str, required=True, help="Model seed", choices=["cifar32", "imagenet64"])
    parser.add_argument("-s", "--seed", type=int, required=True, help="Model seed", choices=[0, 10, 42])
    parser.add_argument(
        "-start", "--start_sampling_idx", type=int, required=False, help="Start index of train steps to sample"
    )
    parser.add_argument(
        "-stop", "--stop_sampling_idx", type=int, required=False, help="Stop index of train steps to sample"
    )
    args = parser.parse_args()

    process(
        model=args.model,
        seed=args.seed,
        start_idx=args.start_sampling_idx,
        stop_idx=args.stop_sampling_idx,
    )
