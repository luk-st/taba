import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from skimage.metrics import structural_similarity as ssim
from sklearn.cross_decomposition import CCA
from torchvision import transforms
from tqdm import tqdm

from taba.metrics.alignment import calculate_psnr, calculate_ssim, cka, svcca
from taba.metrics.annotator import get_vit_cifar10_annotator, get_vit_imagenet_annotator
from taba.metrics.clip_dino import get_clip, get_clip_features, get_dino, get_dino_features
from taba.metrics.inception import InceptionV3, inception_feats

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


N_SAMPLES = 2048
BATCH_SIZE = 128
T = 100
N_NEIGHBOURS = 50


def get_annotator_hidden_states(samples, annotator, batch_size=32):
    all_feats = []
    with torch.no_grad():
        for idx_start in range(0, len(samples), batch_size):
            idx_end = idx_start + batch_size
            dat_in = samples[idx_start:idx_end]
            inputs = annotator.processor(images=dat_in, return_tensors="pt").to(DEVICE)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            outputs = annotator.model(**inputs, output_hidden_states=True)
            feats = outputs.hidden_states[-1][:, 0, :]
            all_feats.append(feats)
    return torch.cat(all_feats).cpu()


def tensors_to_tensors(tensors):
    tensors = tensors - tensors.min()
    tensors = tensors / tensors.max()
    return tensors


def tensors_to_pils(tensors):
    tensors = tensors - tensors.min()
    tensors = tensors / tensors.max()
    return [transforms.ToPILImage()(i_tens) for i_tens in tensors]


def calculate_features(samples: torch.Tensor, models):
    features = {}
    img_samples = tensors_to_tensors(samples)
    dist = torch.from_numpy(cdist(img_samples.flatten(1, 3), img_samples.flatten(1, 3)))
    img_knn = dist.topk(N_NEIGHBOURS + 1, largest=False)[1][:, 1:]
    features["knn"] = img_knn
    features["cka"] = img_samples.flatten(1, 3)
    features["svcca"] = img_samples.flatten(1, 3)
    features["ssim"] = img_samples
    features["psnr"] = img_samples

    img_samples_pils = tensors_to_pils(samples)
    features["dino"] = get_dino_features(img_samples_pils, models["dino"])
    features["clip"] = get_clip_features(img_samples_pils, models["clip_processor"], models["clip_model"])
    features["annotator"] = get_annotator_hidden_states(img_samples_pils, models["annotator"])
    features["inception"] = inception_feats(samples_tensor=samples, inception_model=models["inception"])
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

    # dino
    dino_cos_sim = F.cosine_similarity(features_to_compare["dino"], features_ckpt["dino"]).cpu()
    dino_cos_sim_mean = dino_cos_sim.mean()
    dino_cos_sim_std = dino_cos_sim.std()

    # clip
    clip_cos_sim = F.cosine_similarity(features_to_compare["clip"], features_ckpt["clip"]).cpu()
    clip_cos_sim_mean = clip_cos_sim.mean()
    clip_cos_sim_std = clip_cos_sim.std()

    # annotator
    annotator_cos_sim = F.cosine_similarity(features_to_compare["annotator"], features_ckpt["annotator"]).cpu()
    annotator_cos_sim_mean = annotator_cos_sim.mean()
    annotator_cos_sim_std = annotator_cos_sim.std()

    # inception
    inception_cos_sim = F.cosine_similarity(features_to_compare["inception"], features_ckpt["inception"]).cpu()
    inception_cos_sim_mean = inception_cos_sim.mean()
    inception_cos_sim_std = inception_cos_sim.std()

    return {
        "knn_mean": knn_mean,
        "knn_std": knn_std,
        "svcca": svcca_value,
        "ssim_mean": ssim_mean,
        "ssim_std": ssim_std,
        "psnr_mean": psnr_mean,
        "psnr_std": psnr_std,
        "cka": cka_value,
        "dino_cos_sim_mean": dino_cos_sim_mean,
        "dino_cos_sim_std": dino_cos_sim_std,
        "clip_cos_sim_mean": clip_cos_sim_mean,
        "clip_cos_sim_std": clip_cos_sim_std,
        "annotator_cos_sim_mean": annotator_cos_sim_mean,
        "annotator_cos_sim_std": annotator_cos_sim_std,
        "inception_cos_sim_mean": inception_cos_sim_mean,
        "inception_cos_sim_std": inception_cos_sim_std,
    }


def load_all_models(model: str):
    inception_module = InceptionV3(use_fid_inception=True).to(DEVICE)
    dino_model = get_dino(device=DEVICE)
    clip_processor, clip_model = get_clip(device=DEVICE)
    annotator = get_vit_imagenet_annotator() if model == "imagenet64" else get_vit_cifar10_annotator()

    return {
        "inception": inception_module,
        "dino": dino_model,
        "clip_model": clip_model,
        "clip_processor": clip_processor,
        "annotator": annotator,
    }


def load_stuff(model: str, seed: int, st_idx: int = 0, is_last_denoiser: bool = False):
    name_workdir = "lastdenoising" if is_last_denoiser else "currentdenoising"

    if model == "imagenet64":
        work_dir = Path(f"experiments/trainsteps__lastsample_noising_{name_workdir}/imgnet_64_last").resolve()
        samples_dir = Path(f"experiments/trainsteps__lastsample_noising_{name_workdir}/imgnet_64_last").resolve()
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
        work_dir = Path(f"experiments/trainsteps__lastsample_noising_{name_workdir}/cifar10_32_last").resolve()
        samples_dir = Path(f"experiments/trainsteps__lastsample_noising_{name_workdir}/cifar10_32_last").resolve()
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

    samples_dir = (samples_dir / f"s{seed}/samples").resolve()
    os.makedirs(work_dir, exist_ok=True)
    stats_output_file = (work_dir / f"stats_s{seed}_stidx{st_idx}.pkl").resolve()

    return stats_output_file, trainsteps_to_sample, checkpoint_idx_to_compare, samples_dir


def save_stats_to_file(dct: dict, stats_output_file: str):
    with open(stats_output_file, "wb") as file:
        pickle.dump(dct, file)


def process(model: str, seed: int, start_idx: int = None, stop_idx: int = None, is_last_denoiser: bool = False):
    print(
        f"INFO | Processing model {model} with seed {seed} from {start_idx} to {stop_idx}. Latents: {is_last_denoiser}"
    )
    stats_output_file, trainsteps_to_sample, checkpoint_idx_to_compare, samples_dir = load_stuff(
        model=model, seed=seed, st_idx=start_idx or 0, is_last_denoiser=is_last_denoiser
    )
    metrics_models = load_all_models(model)

    ranges = trainsteps_to_sample
    if stop_idx is not None:
        ranges = [r for r in ranges if r <= stop_idx]
    if start_idx is not None:
        ranges = [r for r in ranges if r >= start_idx]
    loop = tqdm(ranges, total=len(ranges))

    all_steps_metrics = {}
    samples_to_compare = torch.load(samples_dir / f"{checkpoint_idx_to_compare}.pt", weights_only=False)
    feats_to_compare = calculate_features(samples_to_compare, metrics_models)

    for checkpoint_step in loop:
        ckpt_samples = torch.load(samples_dir / f"{checkpoint_step}.pt", weights_only=False)
        ckpt_feats = calculate_features(ckpt_samples, metrics_models)

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
    parser.add_argument(
        "-denoiser",
        "--is_last_denoiser",
        default=False,
        action="store_true",
        help="Whether to use last checkpoint denoiser or current denoiser",
    )
    args = parser.parse_args()

    process(
        model=args.model,
        seed=args.seed,
        start_idx=args.start_sampling_idx,
        stop_idx=args.stop_sampling_idx,
        is_last_denoiser=args.is_last_denoiser,
    )
