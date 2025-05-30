import argparse
import os
import pickle
from pathlib import Path

import torch
from tqdm import tqdm

from taba.metrics.angles_distances import calc_angles, calc_distances
from taba.models.adm.models import get_ls_cifar10, get_ls_imagenet
from taba.models.adm.ddim import generate_latents, generate_noises, generate_samples
from taba.exp_pipelines.noise_classification.distance_classification import (
    get_noise_sample_by_distance_classification,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


N_SAMPLES = 2048
BATCH_SIZE = 128
T = 100


def load_stuff(model: str, seed: int, st_idx: int = 0):
    if model == "imagenet64":
        work_dir = Path("experiments/trainsteps/imgnet_64_last").resolve()
        samples_dir = Path("experiments/trainsteps/imgnet_64").resolve()
        checkpoints_dir = Path(f"res/ckpt_models/imagenet_64/seed_{seed}").resolve()
        trainsteps_to_sample = sorted(
            list(
                set(
                    [0, 25, 100]
                    + list(range(0, 10_308, 2500))
                    + list(range(0, 522_500, 2 * 10_307))
                    + list(range(522_500, 1_130_613, 2 * 10_307))
                    + list(range(1_130_613, 1_500_613, 2 * 10_307))
                    # + list(range(0, 1_130_613, 2_500))
                    # + list(range(1_130_613, 1_500_613, 2_500))
                )
            )
        )
        checkpoint_idx_to_compare = 1_500_613
    elif model == "cifar32":
        work_dir = Path("experiments/trainsteps/cifar10_32_last").resolve()
        samples_dir = Path("experiments/trainsteps/cifar10_32").resolve()
        checkpoints_dir = Path(f"res/ckpt_models/cifar10_32/seed_{seed}").resolve()
        trainsteps_to_sample = sorted(
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
        checkpoint_idx_to_compare = 700_000
    else:
        raise NotImplementedError(f"Unknown setup: {model}")

    output_dir = (work_dir / f"s{seed}").resolve()
    samples_dir = (samples_dir / f"s{seed}/samples").resolve()
    stats_output_file = (work_dir / f"stats_s{seed}_stidx{st_idx}.pkl").resolve()

    return checkpoints_dir, output_dir, stats_output_file, trainsteps_to_sample, checkpoint_idx_to_compare, samples_dir


def process_checkpoint(
    n_steps_checkpoint: int, model: str, checkpoints_dir: str, samples_to_compare: torch.Tensor
) -> dict:
    if model == "imagenet64":
        model, diffusion, args = get_ls_imagenet(
            steps=T, model_path=checkpoints_dir / f"ema_0.9999_{n_steps_checkpoint:06}.pt"
        )
    else:
        model, diffusion, args = get_ls_cifar10(
            steps=T, model_path=checkpoints_dir / f"ema_0.9999_{n_steps_checkpoint:06}.pt"
        )
    noises = generate_noises(number_of_samples=N_SAMPLES, diffusion_args=args, device=DEVICE)
    latents = generate_latents(
        ddim_generations=samples_to_compare,
        batch_size=BATCH_SIZE,
        diffusion_pipeline=diffusion,
        ddim_model=model,
        device=DEVICE,
    )
    return noises.cpu(), latents.cpu()


def process_stats(noises, samples, latents):
    angles_stats = calc_angles(noises=noises, samples=samples, latents=latents)
    dist_stats = calc_distances(noise=noises, sample_from_noise=samples, latent=latents)
    distclassification_stats = get_noise_sample_by_distance_classification(
        noises=noises, samples=samples, with_plotting=False
    )
    return {**angles_stats, **dist_stats, **distclassification_stats}


def save_stats_to_file(dct: dict, stats_output_file: str):
    with open(stats_output_file, "wb") as file:
        pickle.dump(dct, file)


def process(model: str, seed: int, start_idx: int = None, stop_idx: int = None):
    checkpoints_dir, output_dir, stats_output_file, trainsteps_to_sample, checkpoint_idx_to_compare, samples_dir = (
        load_stuff(model=model, seed=seed, st_idx=start_idx or 0)
    )

    DIR_NOISES = (output_dir / "noises").resolve()
    DIR_LATENTS = (output_dir / "latents").resolve()
    DIR_SAMPLES = (output_dir / "samples").resolve()

    os.makedirs(DIR_NOISES, exist_ok=True)
    os.makedirs(DIR_LATENTS, exist_ok=True)
    os.makedirs(DIR_SAMPLES, exist_ok=True)

    ranges = trainsteps_to_sample
    if stop_idx is not None:
        ranges = [r for r in ranges if r <= stop_idx]
    if start_idx is not None:
        ranges = [r for r in ranges if r >= start_idx]

    all_metrics = {}
    loop = tqdm(ranges, total=len(ranges))

    samples_to_compare = torch.load(samples_dir / f"{checkpoint_idx_to_compare}.pt")

    for checkpoint_step in loop:
        ckpt_noises, ckpt_latents = process_checkpoint(
            n_steps_checkpoint=checkpoint_step,
            model=model,
            checkpoints_dir=checkpoints_dir,
            samples_to_compare=samples_to_compare,
        )
        torch.save(ckpt_noises, (DIR_NOISES / f"{checkpoint_step}.pt"))
        torch.save(samples_to_compare, (DIR_SAMPLES / f"{checkpoint_step}.pt"))
        torch.save(ckpt_latents, (DIR_LATENTS / f"{checkpoint_step}.pt"))

        checkpoint_metrics = process_stats(noises=ckpt_noises, samples=samples_to_compare, latents=ckpt_latents)
        all_metrics[str(checkpoint_step)] = checkpoint_metrics
        save_stats_to_file(all_metrics, stats_output_file=stats_output_file)


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

    process(model=args.model, seed=args.seed, start_idx=args.start_sampling_idx, stop_idx=args.stop_sampling_idx)
