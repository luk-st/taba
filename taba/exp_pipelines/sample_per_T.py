import argparse
import os

import torch

from taba.models.adm.ddim import generate_latents, generate_noises, generate_samples
from taba.models.adm.models import get_ddpm_imagenet256, get_openai_cifar, get_openai_imagenet
from taba.models.dit.constants import DIT_IMAGENET_CLASSES_SMALL
from taba.models.dit.dit import CustomDiTPipeline
from taba.models.dit.sample_per_T import generate_noises_classes
from taba.models.ldms.models import get_inv_scheduler, get_ldm_celeba, get_scheduler
from taba.models.ldms.sample_ldm import decode_image
from taba.models.ldms.sample_ldm import generate_latents as ldm_generate_latents
from taba.models.ldms.sample_ldm import generate_samples as ldm_generate_samples
from taba.models.ldms.sample_ldm import get_noises as ldm_generate_noises

SAVE_DIR_PATH = "experiments/outputs_per_T{internal}/{seed}/{ds_name}/T_{T}/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUMBER_OF_SAMPLES = 8 * 1024
DEFAULT_BATCH_SIZE = 32


def save_results(outputs, T, ds_name, internal: bool, seed: int, part: int | None = None):
    if part is not None:
        suffix = f"_part{part}"
    else:
        suffix = ""
    internal_format = "_internal" if internal else ""
    dir_path = SAVE_DIR_PATH.format(internal=internal_format, seed=seed, ds_name=ds_name, T=T)
    os.makedirs(dir_path, exist_ok=True)
    print(f"Saving results to {dir_path}")

    for file_name, data in outputs.items():
        torch.save(data, dir_path + f"{file_name}{suffix}.pt")


def sample_ddpm(
    ddpm_model,
    diffusion_pipeline,
    diffusion_args,
    seed: int,
    part: int | None = None,
    n_parts: int = 1,
    internal: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    noise = generate_noises(NUMBER_OF_SAMPLES, diffusion_args, seed=seed)
    n_samples = NUMBER_OF_SAMPLES
    if part is not None:
        noise = noise[(part * NUMBER_OF_SAMPLES) // n_parts : ((part + 1) * NUMBER_OF_SAMPLES) // n_parts]
        n_samples = NUMBER_OF_SAMPLES // n_parts

    outs = generate_samples(
        random_noises=noise,
        number_of_samples=n_samples,
        batch_size=batch_size,
        diffusion_pipeline=diffusion_pipeline,
        ddim_model=ddpm_model,
        diffusion_args=diffusion_args,
        device=DEVICE,
        from_each_t=internal,
    )
    if internal:
        samples, xts_denoising, t_eps_denoising, t_predxstart_denoising = (
            outs["samples"],
            outs["all_t_samples"],
            outs["all_t_eps_samples"],
            outs["all_t_pred_xstart_samples"],
        )
    else:
        samples = outs

    outs_noising = generate_latents(
        ddim_generations=samples,
        batch_size=batch_size,
        diffusion_pipeline=diffusion_pipeline,
        ddim_model=ddpm_model,
        device=DEVICE,
        from_each_t=internal,
    )
    if internal:
        latents, xts_noising, t_eps_noising, t_predxstart_noising = (
            outs_noising["latents"],
            outs_noising["all_t_latents"],
            outs_noising["all_t_eps_samples"],
            outs_noising["all_t_pred_xstart_samples"],
        )
    else:
        latents = outs_noising
    if internal:
        return {
            "noise": noise,
            "samples": samples,
            "latents": latents,
            "xts_denoising": xts_denoising,
            "t_eps_denoising": t_eps_denoising,
            # "t_predxstart_denoising": t_predxstart_denoising,
            "xts_noising": xts_noising,
            "t_eps_noising": t_eps_noising,
            # "t_predxstart_noising": t_predxstart_noising,
        }
    else:
        return {
            "noise": noise,
            "samples": samples,
            "latents": latents,
        }


def sample_cifar10(
    T: int,
    seed: int,
    internal: bool = False,
    part: int | None = None,
    n_parts: int = 1,
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    model, diffusion, args = get_openai_cifar(steps=T, device=DEVICE)
    return sample_ddpm(
        ddpm_model=model,
        diffusion_pipeline=diffusion,
        diffusion_args=args,
        part=part,
        n_parts=n_parts,
        internal=internal,
        batch_size=batch_size,
        seed=seed,
    )


def sample_imagenet(
    T: int,
    seed: int,
    internal: bool = False,
    part: int | None = None,
    n_parts: int = 1,
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    model, diffusion, args = get_openai_imagenet(steps=T, device=DEVICE)
    return sample_ddpm(
        ddpm_model=model,
        diffusion_pipeline=diffusion,
        diffusion_args=args,
        part=part,
        n_parts=n_parts,
        internal=internal,
        batch_size=batch_size,
        seed=seed,
    )


def sample_ddpm256(
    T: int,
    seed: int,
    part: int | None = None,
    n_parts: int = 1,
    internal: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    model, diffusion, args = get_ddpm_imagenet256(steps=T, device=DEVICE)
    return sample_ddpm(
        ddpm_model=model,
        diffusion_pipeline=diffusion,
        diffusion_args=args,
        part=part,
        n_parts=n_parts,
        internal=internal,
        batch_size=batch_size,
        seed=seed,
    )


def sample_celeba(
    T: int,
    seed: int,
    internal: bool = False,
    part: int | None = None,
    n_parts: int = 1,
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    ldm_unet, ldm_vae = get_ldm_celeba(device=DEVICE)
    scheduler = get_scheduler(T=T)
    inv_scheduler = get_inv_scheduler(T=T)

    noises = ldm_generate_noises(n_samples=NUMBER_OF_SAMPLES, seed=seed)
    if part is not None:
        noises = noises[(part * NUMBER_OF_SAMPLES) // n_parts : ((part + 1) * NUMBER_OF_SAMPLES) // n_parts]

    denoising_outs = ldm_generate_samples(
        noise=noises,
        diffusion_unet=ldm_unet,
        diffusion_scheduler=scheduler,
        batch_size=batch_size,
        device=DEVICE,
        from_each_t=internal,
    )

    if internal:
        samples, xts_denoising, t_eps_denoising, t_predxstart_denoising = (
            denoising_outs["samples"],
            denoising_outs["all_t_samples"],
            denoising_outs["all_t_eps_samples"],
            denoising_outs["all_t_pred_xstart_samples"],
        )
    else:
        samples = denoising_outs

    samples_decoded = decode_image(unet_out=samples, vqvae=ldm_vae, batch_size=batch_size, device=DEVICE)

    noising_outs = ldm_generate_latents(
        samples=samples,
        diffusion_unet=ldm_unet,
        diffusion_scheduler=inv_scheduler,
        batch_size=batch_size,
        device=DEVICE,
        from_each_t=internal,
    )

    if internal:
        latents, xts_noising, t_eps_noising, t_predxstart_noising = (
            noising_outs["latents"],
            noising_outs["all_t_latents"],
            noising_outs["all_t_eps_samples"],
            noising_outs["all_t_pred_xstart_samples"],
        )
    else:
        latents = noising_outs
    if internal:
        return {
            "noise": noises,
            "samples": samples,
            "samples_decoded": samples_decoded,
            "latents": latents,
            "xts_denoising": xts_denoising,
            "t_eps_denoising": t_eps_denoising,
            # "t_predxstart_denoising": t_predxstart_denoising,
            "xts_noising": xts_noising,
            "t_eps_noising": t_eps_noising,
            # "t_predxstart_noising": t_predxstart_noising,
        }
    else:
        return {
            "noise": noises,
            "samples": samples,
            "samples_decoded": samples_decoded,
            "latents": latents,
        }


def sample_dit(
    T: int,
    seed: int,
    internal: bool = False,
    part: int | None = None,
    n_parts: int = 1,
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    dit_pipeline = CustomDiTPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.float16)
    dit_pipeline = dit_pipeline.to("cuda")
    noises, class_ids = generate_noises_classes(
        dit_pipeline, seed=seed, n_samples=NUMBER_OF_SAMPLES, all_classes=DIT_IMAGENET_CLASSES_SMALL
    )

    if part is not None:
        noises = noises[(part * NUMBER_OF_SAMPLES) // n_parts : ((part + 1) * NUMBER_OF_SAMPLES) // n_parts]
        class_ids = class_ids[(part * NUMBER_OF_SAMPLES) // n_parts : ((part + 1) * NUMBER_OF_SAMPLES) // n_parts]

    denoising_outs = dit_pipeline.ddim(
        latents_x_T=noises,
        guidance_scale=1.0,
        batch_size=batch_size,
        num_inference_steps=T,
        class_labels=class_ids,
        from_each_t=internal,
    )

    if internal:
        samples, xts_denoising, t_eps_denoising, t_predxstart_denoising = (
            denoising_outs["samples"],
            denoising_outs["all_t_samples"],
            denoising_outs["all_t_eps_samples"],
            denoising_outs["all_t_pred_xstart_samples"],
        )
    else:
        samples = denoising_outs

    samples_decoded = dit_pipeline.vae_decode(latents=samples, batch_size=batch_size)

    noising_outs = dit_pipeline.ddim_inverse(
        latents_x_0=samples,
        guidance_scale=1.0,
        batch_size=batch_size,
        num_inference_steps=T,
        class_labels=class_ids,
        from_each_t=internal,
    )

    if internal:
        latents, xts_noising, t_eps_noising, t_predxstart_noising = (
            noising_outs["latents"],
            noising_outs["all_t_latents"],
            noising_outs["all_t_eps_samples"],
            noising_outs["all_t_pred_xstart_samples"],
        )
    else:
        latents = noising_outs
    if internal:
        return {
            "noise": noises,
            "samples": samples,
            "samples_decoded": samples_decoded,
            "latents": latents,
            "xts_denoising": xts_denoising,
            "t_eps_denoising": t_eps_denoising,
            # "t_predxstart_denoising": t_predxstart_denoising,
            "xts_noising": xts_noising,
            "t_eps_noising": t_eps_noising,
            # "t_predxstart_noising": t_predxstart_noising,
            "class_ids": class_ids,
        }
    else:
        return {
            "noise": noises,
            "samples": samples,
            "samples_decoded": samples_decoded,
            "latents": latents,
            "class_ids": class_ids,
        }


def main(
    ds_name: str,
    T: int,
    seed: int,
    part: int | None = None,
    n_parts: int = 1,
    internal: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    print(
        f"Sampling {ds_name} with T={T}, part={part}, n_parts={n_parts}, internal={internal}, batch_size={batch_size}, seed={seed}"
    )
    if ds_name == "cifar_pixel_32":
        outputs = sample_cifar10(T=T, internal=internal, part=part, n_parts=n_parts, batch_size=batch_size, seed=seed)
    elif ds_name == "imagenet_pixel_64":
        outputs = sample_imagenet(T=T, internal=internal, part=part, n_parts=n_parts, batch_size=batch_size, seed=seed)
    elif ds_name == "celeba_ldm_256":
        outputs = sample_celeba(T=T, internal=internal, part=part, n_parts=n_parts, batch_size=batch_size, seed=seed)
    elif ds_name == "imagenet_pixel_256":
        outputs = sample_ddpm256(T=T, part=part, n_parts=n_parts, internal=internal, batch_size=batch_size, seed=seed)
    elif ds_name == "imagenet_dit_256":
        outputs = sample_dit(T=T, part=part, n_parts=n_parts, internal=internal, batch_size=batch_size, seed=seed)
    else:
        raise ValueError(f"Unknown dataset name: {ds_name}")
    save_results(outputs=outputs, T=T, ds_name=ds_name, internal=internal, part=part, seed=seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_name", type=str, required=True)
    parser.add_argument("--T", type=int, required=True)
    parser.add_argument("--part", type=int, default=None)
    parser.add_argument("--n_parts", type=int, default=1)
    parser.add_argument("--internal", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    main(
        ds_name=args.ds_name,
        T=args.T,
        part=args.part,
        n_parts=args.n_parts,
        internal=args.internal,
        batch_size=args.batch_size,
        seed=args.seed,
    )
