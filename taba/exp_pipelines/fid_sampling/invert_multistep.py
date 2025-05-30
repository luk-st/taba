import argparse
import os

import torch

from taba.models.ldms.models import get_inv_scheduler, get_ldm_celeba
from taba.models.ldms.sample_ldm import generate_latents as ldm_generate_latents
from taba.models.adm.models import get_ddpm_imagenet256, get_openai_cifar, get_openai_imagenet
from taba.models.adm.ddim import generate_latents

SAVE_DIR_PATH = "experiments/latents_swapped/{model_name}/{T}"
LOAD_MULTISTEP_DIR_PATH = "experiments/outputs_per_T_internal/{model_name}/T_{T}/{file_name}"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
SEED = 42
INTERNAL = True


def save_results(outs, model_name: str, T: int):
    dir_path = SAVE_DIR_PATH.format(model_name=model_name, T=T)
    os.makedirs(dir_path, exist_ok=True)
    print(f"Saving results to {dir_path}")
    for name, data in outs.items():
        torch.save(data, dir_path + "/" + f"{name}.pt")


def load_samples(model_name: str, T: int, use_multistep: bool):
    samples = torch.load(LOAD_MULTISTEP_DIR_PATH.format(model_name=model_name, T=T, file_name="samples.pt"))
    if use_multistep:
        eps = torch.load(LOAD_MULTISTEP_DIR_PATH.format(model_name=model_name, T=T, file_name="t_eps_denoising.pt"))
        return {"samples": samples, "eps": eps}
    return {"samples": samples}


def sample_ddpm(
    model_name,
    ddpm_model,
    diffusion_pipeline,
    diffusion_args,
    T: int,
    use_multistep: bool,
):
    samples = load_samples(model_name, T=T, use_multistep=use_multistep)
    if use_multistep:
        images = samples["samples"]
        swap_eps = samples["eps"].to("cpu")
        swap_eps = {0: swap_eps[-1]}

    else:
        images = samples["samples"]
        swap_eps = {}

    latents = generate_latents(
        ddim_generations=images,
        batch_size=BATCH_SIZE,
        diffusion_pipeline=diffusion_pipeline,
        ddim_model=ddpm_model,
        device=DEVICE,
        from_each_t=INTERNAL,
        swap_eps=swap_eps,
    )
    return latents


def sample_cifar10(model_name: str, T: int, use_multistep: bool):
    model, diffusion, args = get_openai_cifar(steps=T, device=DEVICE)
    return sample_ddpm(
        model_name=model_name,
        ddpm_model=model,
        diffusion_pipeline=diffusion,
        diffusion_args=args,
        T=T,
        use_multistep=use_multistep,
    )


def sample_imagenet(model_name: str, T: int, use_multistep: bool):
    model, diffusion, args = get_openai_imagenet(steps=T, device=DEVICE)
    return sample_ddpm(
        model_name=model_name,
        ddpm_model=model,
        diffusion_pipeline=diffusion,
        diffusion_args=args,
        T=T,
        use_multistep=use_multistep,
    )


def sample_ddpm256(model_name: str, T: int, use_multistep: bool):
    model, diffusion, args = get_ddpm_imagenet256(steps=T, device=DEVICE)
    return sample_ddpm(
        model_name=model_name,
        ddpm_model=model,
        diffusion_pipeline=diffusion,
        diffusion_args=args,
        T=T,
        use_multistep=use_multistep,
    )


def sample_celeba(model_name: str, T: int, use_multistep: bool):
    ldm_unet, _ = get_ldm_celeba(device=DEVICE)
    inv_scheduler = get_inv_scheduler(T=T)

    samples = load_samples(model_name, T=T, use_multistep=use_multistep)
    images = samples["samples"]
    latents = ldm_generate_latents(
        samples=images,
        diffusion_unet=ldm_unet,
        diffusion_scheduler=inv_scheduler,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        from_each_t=False,
    )

    return {"latents": latents}


def main(ds_name: str, T: int, use_multistep: bool = False):
    if ds_name == "cifar_pixel_32":
        outputs = sample_cifar10(model_name=ds_name, T=T, use_multistep=use_multistep)
    elif ds_name == "imagenet_pixel_64":
        outputs = sample_imagenet(model_name=ds_name, T=T, use_multistep=use_multistep)
    elif ds_name == "celeba_ldm_256":
        outputs = sample_celeba(model_name=ds_name, T=T, use_multistep=use_multistep)
    elif ds_name == "imagenet_pixel_256":
        outputs = sample_ddpm256(model_name=ds_name, T=T, use_multistep=use_multistep)
    else:
        raise ValueError(f"Unknown dataset name: {ds_name}")
    save_results(outs=outputs, model_name=ds_name, T=T)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_name", type=str, required=True)
    parser.add_argument("--T", type=int, required=True)
    parser.add_argument("--use_multistep", type=bool, default=False)
    args = parser.parse_args()
    main(args.ds_name, args.T, args.use_multistep)
