import argparse
import os

import torch

from taba.models.ldms.models import get_ldm_celeba, get_scheduler
from taba.models.ldms.sample_ldm import generate_samples as ldm_generate_samples
from taba.models.adm.models get_ddpm_imagenet256, get_openai_cifar, get_openai_imagenet
from taba.models.adm.ddim import generate_samples as pixel_generate_samples

SAVE_DIR_PATH = "experiments/latent_sample/{model_name}/T_{T}/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 64
SEED = 42

MODELS_MAP = {
    "cifar_pixel_32": "cifar_pixel_32",
    "imagenet_pixel_64": "imagenet_pixel_64",
    "imagenet_pixel_256": "imagenet_pixel_256",
    "celeba_ldm_256": "celeba_ldm_256",
    "imagenet_dit_256": "imagenet_dit_256",
}

MODELS_TO_TS = {
    "cifar_pixel_32": 4000,
    "imagenet_pixel_64": 4000,
    "imagenet_pixel_256": 1000,
    "celeba_ldm_256": 1000,
    "imagenet_dit_256": 1000,
}

PIXEL_MODELS_GET_FUNCTIONS = {
    "cifar_pixel_32": get_openai_cifar,
    "imagenet_pixel_64": get_openai_imagenet,
    "imagenet_pixel_256": get_ddpm_imagenet256,
}


def save_results(recons: torch.Tensor, model_name: str, T: int, part: int | None = None):
    if part is not None:
        suffix = f"_{part}"
    else:
        suffix = ""
    dir_path = SAVE_DIR_PATH.format(model_name=model_name, T=T)
    os.makedirs(dir_path, exist_ok=True)
    print(f"Saving results to {dir_path}")
    torch.save(recons, dir_path + f"recons{suffix}.pt")


def sample_ddpm(
    ddpm_model,
    diffusion_pipeline,
    diffusion_args,
    input_noise: torch.Tensor,
    part: int | None = None,
    n_parts: int = 1,
    internal: bool = False,
):
    n_samples = input_noise.shape[0]
    if part is not None:
        noise = input_noise[(part * n_samples) // n_parts : ((part + 1) * n_samples) // n_parts]
        n_samples = n_samples // n_parts

    noise = noise.to(DEVICE)
    outs = pixel_generate_samples(
        random_noises=noise,
        number_of_samples=n_samples,
        batch_size=BATCH_SIZE,
        diffusion_pipeline=diffusion_pipeline,
        ddim_model=ddpm_model,
        diffusion_args=diffusion_args,
        device=DEVICE,
        from_each_t=internal,
    )
    if internal:
        samples, all_t_samples = outs["samples"], outs["all_t_samples"]
    else:
        samples = outs
    return (noise, all_t_samples) if internal else (noise, samples)


def sample_pixel(
    input_noise: torch.Tensor,
    model_name: str,
    T: int,
    internal: bool = False,
    part: int | None = None,
    n_parts: int = 1,
):
    model, diffusion, args = PIXEL_MODELS_GET_FUNCTIONS[model_name](steps=T, device=DEVICE)
    return sample_ddpm(
        ddpm_model=model,
        diffusion_pipeline=diffusion,
        diffusion_args=args,
        input_noise=input_noise,
        part=part,
        n_parts=n_parts,
        internal=internal,
    )


def sample_celeba(
    input_noise: torch.Tensor, T: int, internal: bool = False, part: int | None = None, n_parts: int = 1
):
    n_samples = input_noise.shape[0]
    if part is not None:
        noise = input_noise[(part * n_samples) // n_parts : ((part + 1) * n_samples) // n_parts]

    ldm_unet, _ = get_ldm_celeba(device=DEVICE)
    scheduler = get_scheduler(T=T)
    noise = noise.to(DEVICE)

    if internal:
        samples, all_t_samples = ldm_generate_samples(
            noise=noise,
            diffusion_unet=ldm_unet,
            diffusion_scheduler=scheduler,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            from_each_t=True,
        )
    else:
        samples = ldm_generate_samples(
            noise=noise,
            diffusion_unet=ldm_unet,
            diffusion_scheduler=scheduler,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            from_each_t=False,
        )

    return (noise, all_t_samples) if internal else (noise, samples)


def run_sampling(
    input_noise_path: str, model_name: str, T: int, part: int | None = None, n_parts: int = 1, internal: bool = False
):
    print(
        f"INFO | Sampling from {model_name} with T={T} and part={part} and n_parts={n_parts} and internal={internal}"
    )
    input_noise = torch.load(input_noise_path)
    if model_name == "imagenet_dit_256":
        pass
    elif "pixel" in model_name:
        noise, samples = sample_pixel(
            input_noise=input_noise, model_name=model_name, T=T, internal=internal, part=part, n_parts=n_parts
        )
    elif model_name == "celeba_ldm_256":
        noise, samples = sample_celeba(input_noise=input_noise, T=T, internal=internal, part=part, n_parts=n_parts)
    else:
        raise ValueError(f"Unknown dataset name: {model_name}")
    save_results(recons=samples, T=T, model_name=model_name, part=part)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, choices=list(MODELS_MAP.keys()))
    parser.add_argument("--T", type=int, required=True)
    parser.add_argument("--part", type=int, default=None)
    parser.add_argument("--n_parts", type=int, default=1)
    parser.add_argument("--internal", type=bool, default=False)
    parser.add_argument("--input_noise_path", type=str, required=True)
    args = parser.parse_args()
    run_sampling(
        input_noise_path=args.input_noise_path,
        model_name=args.model_name,
        T=args.T,
        part=args.part,
        n_parts=args.n_parts,
        internal=args.internal,
    )
