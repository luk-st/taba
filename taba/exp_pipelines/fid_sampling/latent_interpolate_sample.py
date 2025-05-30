import argparse
import os

import torch

from taba.interpolations.interpolations import slerp_interpolation
from taba.models.adm.ddim import generate_samples
from taba.models.adm.models import get_ddpm_imagenet256, get_openai_cifar, get_openai_imagenet
from taba.models.ldms.models import get_ldm_celeba, get_scheduler
from taba.models.ldms.sample_ldm import decode_image
from taba.models.ldms.sample_ldm import generate_samples as ldm_generate_samples

SAVE_DIR_PATH = "experiments/fid_latent_interpolate/{model_name}/alpha_{interpolation_alpha}/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUMBER_OF_SAMPLES = 1024 * 5
BATCH_SIZE = 32
T = 100
INTERNAL = False

LEFT_LATENT_PATH = "experiments/outputs_fid/noise_interpolate/{model_name}/alpha_0.0/alpha_0.0latents.pt"
RIGHT_LATENT_PATH = "experiments/outputs_fid/noise_interpolate/{model_name}/alpha_1.0/alpha_1.0latents.pt"


def save_results(outs, model_name, interpolation_alpha: float, part: int | None = None):
    if part is not None:
        suffix = f"_{part}"
    else:
        suffix = ""
    dir_path = SAVE_DIR_PATH.format(model_name=model_name, interpolation_alpha=interpolation_alpha)
    os.makedirs(dir_path, exist_ok=True)
    print(f"Saving results to {dir_path}")
    for name, data in outs.items():
        torch.save(data, dir_path + f"{name}{suffix}.pt")


def sample_ddpm(
    model_name: str,
    ddpm_model,
    diffusion_pipeline,
    diffusion_args,
    interpolation_alpha: float,
    part: int | None = None,
    n_parts: int = 1,
):
    left_latents = torch.load(LEFT_LATENT_PATH.format(model_name=model_name), map_location="cpu")
    right_latents = torch.load(RIGHT_LATENT_PATH.format(model_name=model_name), map_location="cpu")
    latents = slerp_interpolation(left_latents, right_latents, interpolation_alpha).to(DEVICE)

    n_samples = NUMBER_OF_SAMPLES
    if part is not None:
        latents = latents[(part * NUMBER_OF_SAMPLES) // n_parts : ((part + 1) * NUMBER_OF_SAMPLES) // n_parts]
        n_samples = NUMBER_OF_SAMPLES // n_parts

    outs = generate_samples(
        random_noises=latents,
        number_of_samples=n_samples,
        batch_size=BATCH_SIZE,
        diffusion_pipeline=diffusion_pipeline,
        ddim_model=ddpm_model,
        diffusion_args=diffusion_args,
        device=DEVICE,
        from_each_t=INTERNAL,
    )

    return {"latents": latents, "samples": outs}


def sample_ldm(model_name: str, interpolation_alpha: float, part: int | None = None, n_parts: int = 1):
    ldm_unet, ldm_vae = get_ldm_celeba(device=DEVICE)
    scheduler = get_scheduler(T=T)

    left_latents = torch.load(LEFT_LATENT_PATH.format(model_name=model_name), map_location="cpu")
    right_latents = torch.load(RIGHT_LATENT_PATH.format(model_name=model_name), map_location="cpu")
    latents = slerp_interpolation(left_latents, right_latents, interpolation_alpha).to(DEVICE)

    n_samples = NUMBER_OF_SAMPLES
    if part is not None:
        latents = latents[(part * NUMBER_OF_SAMPLES) // n_parts : ((part + 1) * NUMBER_OF_SAMPLES) // n_parts]
        n_samples = NUMBER_OF_SAMPLES // n_parts

    outs = ldm_generate_samples(
        noise=latents,
        diffusion_unet=ldm_unet,
        diffusion_scheduler=scheduler,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        from_each_t=INTERNAL,
    )
    samples_decoded = decode_image(
        unet_out=outs,
        vqvae=ldm_vae,
        batch_size=BATCH_SIZE,
        device=DEVICE,
    )

    return {"latents": latents, "samples": outs.cpu(), "samples_decoded": samples_decoded.cpu()}


def sample_cifar10(model_name: str, interpolation_alpha: float):
    model, diffusion, args = get_openai_cifar(steps=T, device=DEVICE)
    return sample_ddpm(
        model_name=model_name,
        ddpm_model=model,
        diffusion_pipeline=diffusion,
        diffusion_args=args,
        interpolation_alpha=interpolation_alpha,
    )


def sample_imagenet(model_name: str, interpolation_alpha: float):
    model, diffusion, args = get_openai_imagenet(steps=T, device=DEVICE)
    return sample_ddpm(
        model_name=model_name,
        ddpm_model=model,
        diffusion_pipeline=diffusion,
        diffusion_args=args,
        interpolation_alpha=interpolation_alpha,
    )


def sample_ddpm256(model_name: str, interpolation_alpha: float, part: int | None = None, n_parts: int = 1):
    model, diffusion, args = get_ddpm_imagenet256(steps=T, device=DEVICE)
    return sample_ddpm(
        model_name=model_name,
        ddpm_model=model,
        diffusion_pipeline=diffusion,
        diffusion_args=args,
        part=part,
        n_parts=n_parts,
        interpolation_alpha=interpolation_alpha,
    )


def main(model_name: str, interpolation_alpha: float, part: int | None = None, n_parts: int = 1):
    assert 0 <= interpolation_alpha <= 1, "interpolation_alpha must be between 0 and 1"
    print(f"INFO | Running {model_name} with interpolation_alpha={interpolation_alpha}. Part={part} n_parts={n_parts}")

    if model_name == "cifar_pixel_32":
        outs = sample_cifar10(model_name=model_name, interpolation_alpha=interpolation_alpha)
    elif model_name == "imagenet_pixel_64":
        outs = sample_imagenet(model_name=model_name, interpolation_alpha=interpolation_alpha)
    elif model_name == "imagenet_pixel_256":
        outs = sample_ddpm256(
            model_name=model_name,
            interpolation_alpha=interpolation_alpha,
            part=part,
            n_parts=n_parts,
        )
    elif model_name == "celeba_ldm_256":
        outs = sample_ldm(
            model_name=model_name,
            interpolation_alpha=interpolation_alpha,
            part=part,
            n_parts=n_parts,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    save_results(
        outs=outs,
        model_name=model_name,
        interpolation_alpha=interpolation_alpha,
        part=part,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--interpolation_alpha", type=float, required=True)
    parser.add_argument("--part", type=int, default=None)
    parser.add_argument("--n_parts", type=int, default=1)
    args = parser.parse_args()
    main(
        model_name=args.model_name,
        interpolation_alpha=args.interpolation_alpha,
        part=args.part,
        n_parts=args.n_parts,
    )
