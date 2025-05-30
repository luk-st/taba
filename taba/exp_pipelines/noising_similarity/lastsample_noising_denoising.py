import argparse
import os
from pathlib import Path

import torch
from tqdm import tqdm

from taba.models.adm.models import get_ls_cifar10, get_ls_imagenet
from taba.models.adm.ddim import generate_latents, generate_noises, generate_samples

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


N_SAMPLES = 2048
BATCH_SIZE = 128
T = 100


def load_stuff(model_name: str, seed: int, st_idx: int = 0, is_last_denoiser: bool = False):
    dirname = f"lastdenoising" if is_last_denoiser else "currentdenoising"
    if model_name == "imagenet_pixel_64":
        work_dir = Path(f"experiments/trainsteps__lastsample_noising_{dirname}/imgnet_64_last").resolve()
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
                )
            )
        )
        checkpoint_idx_to_compare = 1_500_613
    elif model_name == "cifar_pixel_32":
        work_dir = Path(f"experiments/trainsteps__lastsample_noising_{dirname}/cifar10_32_last").resolve()
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
        raise NotImplementedError(f"Unknown setup: {model_name}")

    output_dir = (work_dir / f"s{seed}").resolve()
    samples_dir = (samples_dir / f"s{seed}/samples").resolve()

    return checkpoints_dir, output_dir, trainsteps_to_sample, checkpoint_idx_to_compare, samples_dir


def process_checkpoint(
    n_steps_checkpoint: int,
    model_name: str,
    checkpoints_dir: str,
    samples_to_compare: torch.Tensor,
    checkpoint_idx_to_compare: int,
    is_last_denoiser: bool,
) -> dict:

    denoising_model_checkpoint_idx = checkpoint_idx_to_compare if is_last_denoiser else n_steps_checkpoint
    if model_name == "imagenet_pixel_64":
        noising_model, noising_diffusion_pipeline, noising_args = get_ls_imagenet(
            steps=T, model_path=checkpoints_dir / f"ema_0.9999_{n_steps_checkpoint:06}.pt", device="cpu"
        )
        denoising_model, denoising_diffusion_pipeline, denoising_args = get_ls_imagenet(
            steps=T, model_path=checkpoints_dir / f"ema_0.9999_{denoising_model_checkpoint_idx:06}.pt", device="cpu"
        )
    elif model_name == "cifar_pixel_32":
        noising_model, noising_diffusion_pipeline, noising_args = get_ls_cifar10(
            steps=T, model_path=checkpoints_dir / f"ema_0.9999_{n_steps_checkpoint:06}.pt", device="cpu"
        )
        denoising_model, denoising_diffusion_pipeline, denoising_args = get_ls_cifar10(
            steps=T, model_path=checkpoints_dir / f"ema_0.9999_{denoising_model_checkpoint_idx:06}.pt", device="cpu"
        )
    else:
        raise NotImplementedError(f"Unknown model: {model_name}")

    clean_noises = generate_noises(number_of_samples=N_SAMPLES, diffusion_args=noising_args, device=DEVICE)

    # inverting to samples to latents using the noising model
    noising_model = noising_model.to(DEVICE)
    latents = generate_latents(
        ddim_generations=samples_to_compare,
        batch_size=BATCH_SIZE,
        diffusion_pipeline=noising_diffusion_pipeline,
        ddim_model=noising_model,
        device=DEVICE,
    )
    noising_model = noising_model.to("cpu")

    # reconstructing samples using the denoising model
    denoising_model = denoising_model.to(DEVICE)
    reconstructed_samples = generate_samples(
        random_noises=latents.to(DEVICE),
        number_of_samples=N_SAMPLES,
        batch_size=BATCH_SIZE,
        diffusion_pipeline=denoising_diffusion_pipeline,
        ddim_model=denoising_model,
        diffusion_args=args,
        device=DEVICE,
    )
    denoising_model = denoising_model.to("cpu")

    return clean_noises.cpu(), latents.cpu(), reconstructed_samples.cpu()


def process(model_name: str, seed: int, start_idx: int = None, stop_idx: int = None, is_last_denoiser: bool = False):
    print(
        f"Processing model: {model_name}, seed: {seed}, start_idx: {start_idx}, stop_idx: {stop_idx}, is_last_denoiser: {is_last_denoiser}"
    )
    checkpoints_dir, output_dir, trainsteps_to_sample, checkpoint_idx_to_compare, samples_dir = load_stuff(
        model_name=model_name, seed=seed, st_idx=start_idx or 0, is_last_denoiser=is_last_denoiser
    )

    DIR_NOISES = (output_dir / "noises").resolve()
    DIR_LATENTS = (output_dir / "latents").resolve()
    DIR_SAMPLES = (output_dir / "samples").resolve()
    DIR_RECONSTRUCTED_SAMPLES = (output_dir / "reconstructed_samples").resolve()

    os.makedirs(DIR_NOISES, exist_ok=True)
    os.makedirs(DIR_LATENTS, exist_ok=True)
    os.makedirs(DIR_SAMPLES, exist_ok=True)
    os.makedirs(DIR_RECONSTRUCTED_SAMPLES, exist_ok=True)

    ranges = trainsteps_to_sample
    if stop_idx is not None:
        ranges = [r for r in ranges if r <= stop_idx]
    if start_idx is not None:
        ranges = [r for r in ranges if r >= start_idx]

    loop = tqdm(ranges, total=len(ranges))

    samples_to_compare = torch.load(samples_dir / f"{checkpoint_idx_to_compare}.pt")

    for checkpoint_step in loop:
        ckpt_noises, ckpt_latents, ckpt_samples = process_checkpoint(
            n_steps_checkpoint=checkpoint_step,
            model_name=model_name,
            checkpoints_dir=checkpoints_dir,
            samples_to_compare=samples_to_compare,
            checkpoint_idx_to_compare=checkpoint_idx_to_compare,
            is_last_denoiser=is_last_denoiser,
        )
        torch.save(ckpt_noises, (DIR_NOISES / f"{checkpoint_step}.pt"))
        torch.save(samples_to_compare, (DIR_SAMPLES / f"{checkpoint_step}.pt"))
        torch.save(ckpt_latents, (DIR_LATENTS / f"{checkpoint_step}.pt"))
        torch.save(ckpt_samples, (DIR_RECONSTRUCTED_SAMPLES / f"{checkpoint_step}.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process train steps.")
    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Model seed", choices=["cifar_pixel_32", "imagenet_pixel_64"]
    )
    parser.add_argument("-s", "--seed", type=int, required=True, help="Model seed", choices=[0, 10, 42])
    parser.add_argument(
        "-start", "--start_sampling_idx", type=int, required=False, help="Start index of train steps to sample"
    )
    parser.add_argument(
        "-stop", "--stop_sampling_idx", type=int, required=False, help="Stop index of train steps to sample"
    )
    parser.add_argument(
        "-is_last_denoiser",
        "--is_last_denoiser",
        default=False,
        action="store_true",
        help="Whether to use last denoiser",
    )
    args = parser.parse_args()

    process(
        model_name=args.model,
        seed=args.seed,
        start_idx=args.start_sampling_idx,
        stop_idx=args.stop_sampling_idx,
        is_last_denoiser=args.is_last_denoiser,
    )
