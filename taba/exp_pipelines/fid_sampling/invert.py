import argparse
import os

import torch

from taba.models.adm.ddim import generate_latents
from taba.models.adm.models import get_ddpm_imagenet256, get_openai_cifar, get_openai_imagenet
from taba.models.ldms.models import get_inv_scheduler, get_ldm_celeba
from taba.models.ldms.sample_ldm import generate_latents as ldm_generate_latents

SAVE_DIR_PATH = "experiments/fid_noise_interpolate/{model_name}"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
SEED = 42
INTERNAL = False


def save_results(outs, model_name: str, part: int | None = None):
    if part is not None:
        suffix = f"_{part}"
    else:
        suffix = ""
    dir_path = SAVE_DIR_PATH.format(model_name=model_name)
    os.makedirs(dir_path, exist_ok=True)
    print(f"Saving results to {dir_path}")
    for name, data in outs.items():
        torch.save(data, dir_path + "/" + f"{name}{suffix}.pt")


def load_samples(model_name):
    samples_0 = torch.load(f"experiments/fid_noise_interpolate/{model_name}/alpha_0_00/alpha_0_00samples.pt")
    samples_1 = torch.load(f"experiments/fid_noise_interpolate/{model_name}/alpha_1_00/alpha_1_00samples.pt")
    return samples_0, samples_1


def sample_ddpm(model_name, ddpm_model, diffusion_pipeline, diffusion_args, part: int | None = None, n_parts: int = 1):
    samples_0, samples_1 = load_samples(model_name)
    n_samples = samples_0.shape[0]
    if part is not None:
        samples_0 = samples_0[(part * n_samples) // n_parts : ((part + 1) * n_samples) // n_parts]
        samples_1 = samples_1[(part * n_samples) // n_parts : ((part + 1) * n_samples) // n_parts]
        n_samples = n_samples // n_parts

    outs_noising_0 = generate_latents(
        ddim_generations=samples_0,
        batch_size=BATCH_SIZE,
        diffusion_pipeline=diffusion_pipeline,
        ddim_model=ddpm_model,
        device=DEVICE,
        from_each_t=INTERNAL,
    )
    outs_noising_1 = generate_latents(
        ddim_generations=samples_1,
        batch_size=BATCH_SIZE,
        diffusion_pipeline=diffusion_pipeline,
        ddim_model=ddpm_model,
        device=DEVICE,
        from_each_t=INTERNAL,
    )
    return {
        "alpha_0_00latents": outs_noising_0,
        "alpha_1_00latents": outs_noising_1,
    }


def sample_cifar10(model_name: str, T: int):
    model, diffusion, args = get_openai_cifar(steps=T, device=DEVICE)
    return sample_ddpm(model_name=model_name, ddpm_model=model, diffusion_pipeline=diffusion, diffusion_args=args)


def sample_imagenet(model_name: str, T: int):
    model, diffusion, args = get_openai_imagenet(steps=T, device=DEVICE)
    return sample_ddpm(model_name=model_name, ddpm_model=model, diffusion_pipeline=diffusion, diffusion_args=args)


def sample_ddpm256(model_name: str, T: int, part: int | None = None, n_parts: int = 1):
    model, diffusion, args = get_ddpm_imagenet256(steps=T, device=DEVICE)
    return sample_ddpm(
        model_name=model_name,
        ddpm_model=model,
        diffusion_pipeline=diffusion,
        diffusion_args=args,
        part=part,
        n_parts=n_parts,
    )


def sample_celeba(model_name: str, T: int):
    ldm_unet, _ = get_ldm_celeba(device=DEVICE)
    inv_scheduler = get_inv_scheduler(T=T)

    samples_0, samples_1 = load_samples(model_name)
    latents_0 = ldm_generate_latents(
        samples=samples_0,
        diffusion_unet=ldm_unet,
        diffusion_scheduler=inv_scheduler,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        from_each_t=False,
    )
    latents_1 = ldm_generate_latents(
        samples=samples_1,
        diffusion_unet=ldm_unet,
        diffusion_scheduler=inv_scheduler,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        from_each_t=False,
    )
    return {
        "alpha_0_00latents": latents_0,
        "alpha_1_00latents": latents_1,
    }


def main(ds_name: str, T: int, part: int | None = None, n_parts: int = 1):
    if ds_name == "cifar_pixel_32":
        outputs = sample_cifar10(model_name=ds_name, T=T)
    elif ds_name == "imagenet_pixel_64":
        outputs = sample_imagenet(model_name=ds_name, T=T)
    elif ds_name == "celeba_ldm_256":
        outputs = sample_celeba(model_name=ds_name, T=T)
    elif ds_name == "imagenet_pixel_256":
        outputs = sample_ddpm256(model_name=ds_name, T=T, part=part, n_parts=n_parts)
    else:
        raise ValueError(f"Unknown dataset name: {ds_name}")
    save_results(outs=outputs, model_name=ds_name, part=part)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_name", type=str, required=True)
    parser.add_argument("--T", type=int, required=True)
    parser.add_argument("--part", type=int, default=None)
    parser.add_argument("--n_parts", type=int, default=1)
    args = parser.parse_args()
    main(args.ds_name, args.T, args.part, args.n_parts)
