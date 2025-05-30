import argparse
import os
from pathlib import Path

import torch

from taba.interpolations.interpolations import slerp_interpolation
from taba.models.adm.ddim import generate_samples
from taba.models.adm.models import get_ddpm_imagenet256, get_openai_cifar, get_openai_imagenet
from taba.models.dit.dit import CustomDiTPipeline
from taba.models.ldms.models import get_ldm_celeba, get_scheduler
from taba.models.ldms.sample_ldm import decode_image
from taba.models.ldms.sample_ldm import generate_samples as ldm_generate_samples

SAVE_DIR_PATH = "experiments/noise_latent_interpolate/{seed_path}/{info}/{ds_name}/T_{T}/alpha_{interpolation_alpha}/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_BATCH_SIZE = 32
INTERNAL = False


def save_results(
    outputs, T, ds_name, interpolation_alpha: float, info: str, part: int | None = None, seed_path: str = None
):
    if part is not None:
        suffix = f"_part{part}"
    else:
        suffix = ""
    dir_path = SAVE_DIR_PATH.format(
        ds_name=ds_name, T=T, interpolation_alpha=interpolation_alpha, info=info, seed_path=seed_path
    )
    os.makedirs(dir_path, exist_ok=True)
    print(f"Saving results to {dir_path}")

    for file_name, data in outputs.items():
        torch.save(data, dir_path + f"{file_name}{suffix}.pt")


def sample_ddpm(
    ddpm_model,
    diffusion_pipeline,
    diffusion_args,
    interpolation_alpha: float,
    left_latent_path: str,
    right_latent_path: str,
    part: int | None = None,
    n_parts: int = 1,
    internal: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    left_noise = torch.load(left_latent_path, map_location="cpu", weights_only=False)
    right_noise = torch.load(right_latent_path, map_location="cpu", weights_only=False)
    noise = slerp_interpolation(left_noise, right_noise, interpolation_alpha)
    n_samples = noise.shape[0]
    if part is not None:
        noise = noise[(part * n_samples) // n_parts : ((part + 1) * n_samples) // n_parts]
        n_samples = n_samples // n_parts
    noise = noise.to(DEVICE)

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

    if internal:
        return {
            "noise": noise,
            "samples": samples,
            "xts_denoising": xts_denoising,
            "t_eps_denoising": t_eps_denoising,
            # "t_predxstart_denoising": t_predxstart_denoising,
        }
    else:
        return {
            "noise": noise,
            "samples": samples,
        }


def sample_cifar10(
    T: int,
    interpolation_alpha: float,
    left_latent_path: str,
    right_latent_path: str,
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
        interpolation_alpha=interpolation_alpha,
        left_latent_path=left_latent_path,
        right_latent_path=right_latent_path,
        part=part,
        n_parts=n_parts,
        internal=internal,
        batch_size=batch_size,
    )


def sample_imagenet(
    T: int,
    interpolation_alpha: float,
    left_latent_path: str,
    right_latent_path: str,
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
        interpolation_alpha=interpolation_alpha,
        left_latent_path=left_latent_path,
        right_latent_path=right_latent_path,
        part=part,
        n_parts=n_parts,
        internal=internal,
        batch_size=batch_size,
    )


def sample_ddpm256(
    T: int,
    interpolation_alpha: float,
    left_latent_path: str,
    right_latent_path: str,
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
        interpolation_alpha=interpolation_alpha,
        left_latent_path=left_latent_path,
        right_latent_path=right_latent_path,
        part=part,
        n_parts=n_parts,
        internal=internal,
        batch_size=batch_size,
    )


def sample_celeba(
    T: int,
    interpolation_alpha: float,
    left_latent_path: str,
    right_latent_path: str,
    internal: bool = False,
    part: int | None = None,
    n_parts: int = 1,
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    ldm_unet, ldm_vae = get_ldm_celeba(device=DEVICE)
    scheduler = get_scheduler(T=T)

    left_noise = torch.load(left_latent_path, map_location="cpu", weights_only=False)
    right_noise = torch.load(right_latent_path, map_location="cpu", weights_only=False)
    noise = slerp_interpolation(left_noise, right_noise, interpolation_alpha)
    n_samples = noise.shape[0]
    if part is not None:
        noise = noise[(part * n_samples) // n_parts : ((part + 1) * n_samples) // n_parts]
        n_samples = n_samples // n_parts
    noise = noise.to(DEVICE)

    outs = ldm_generate_samples(
        noise=noise,
        diffusion_unet=ldm_unet,
        diffusion_scheduler=scheduler,
        batch_size=batch_size,
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

    samples_decoded = decode_image(unet_out=samples, vqvae=ldm_vae, batch_size=batch_size, device=DEVICE)

    if internal:
        return {
            "noise": noise,
            "samples": samples,
            "samples_decoded": samples_decoded,
            "xts_denoising": xts_denoising,
            "t_eps_denoising": t_eps_denoising,
            # "t_predxstart_denoising": t_predxstart_denoising,
        }
    else:
        return {
            "noise": noise,
            "samples": samples,
            "samples_decoded": samples_decoded,
        }


def sample_dit(
    T: int,
    interpolation_alpha: float,
    left_latent_path: str,
    right_latent_path: str,
    internal: bool = False,
    part: int | None = None,
    n_parts: int = 1,
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    dit_pipeline = CustomDiTPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.float16)
    dit_pipeline = dit_pipeline.to("cuda")
    left_noise = torch.load(left_latent_path, map_location="cpu", weights_only=False)
    right_noise = torch.load(right_latent_path, map_location="cpu", weights_only=False)

    left_class_ids = torch.load(
        (Path(left_latent_path).parent / "class_ids.pt"), map_location="cpu", weights_only=False
    )
    right_class_ids = torch.load(
        (Path(right_latent_path).parent / "class_ids.pt"), map_location="cpu", weights_only=False
    )
    assert torch.equal(left_class_ids, right_class_ids), "Left and right class ids are not the same"

    noise = slerp_interpolation(left_noise, right_noise, interpolation_alpha)
    n_samples = noise.shape[0]
    if part is not None:
        noise = noise[(part * n_samples) // n_parts : ((part + 1) * n_samples) // n_parts]
        class_ids = left_class_ids[(part * n_samples) // n_parts : ((part + 1) * n_samples) // n_parts]
        n_samples = n_samples // n_parts
    noise = noise.to(DEVICE)
    class_ids = class_ids.to(DEVICE)

    outs = dit_pipeline.ddim(
        latents_x_T=noise,
        guidance_scale=1.0,
        batch_size=batch_size,
        num_inference_steps=T,
        class_labels=class_ids,
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

    samples_decoded = dit_pipeline.vae_decode(latents=samples, batch_size=batch_size)

    if internal:
        return {
            "noise": noise,
            "samples": samples,
            "samples_decoded": samples_decoded,
            "xts_denoising": xts_denoising,
            "t_eps_denoising": t_eps_denoising,
            "class_ids": class_ids,
        }
    else:
        return {
            "noise": noise,
            "samples": samples,
            "samples_decoded": samples_decoded,
            "class_ids": class_ids,
        }


def main(
    ds_name: str,
    T: int,
    interpolation_alpha: float,
    left_latent_path: str,
    right_latent_path: str,
    info: str,
    part: int | None = None,
    n_parts: int = 1,
    internal: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed_path: str = None,
):
    print(
        f"Sampling {ds_name} with T={T}, part={part}, n_parts={n_parts}, internal={internal}, batch_size={batch_size}, interpolation_alpha={interpolation_alpha}, left_latent_path={left_latent_path}, right_latent_path={right_latent_path}, seed_path={seed_path}"
    )
    assert os.path.exists(left_latent_path), f"Left latent path {left_latent_path} does not exist"
    assert os.path.exists(right_latent_path), f"Right latent path {right_latent_path} does not exist"
    assert 0 <= interpolation_alpha <= 1, "interpolation_alpha must be between 0 and 1"

    if ds_name == "cifar_pixel_32":
        outputs = sample_cifar10(
            T=T,
            interpolation_alpha=interpolation_alpha,
            left_latent_path=left_latent_path,
            right_latent_path=right_latent_path,
            internal=internal,
            part=part,
            n_parts=n_parts,
            batch_size=batch_size,
        )
    elif ds_name == "imagenet_pixel_64":
        outputs = sample_imagenet(
            T=T,
            interpolation_alpha=interpolation_alpha,
            left_latent_path=left_latent_path,
            right_latent_path=right_latent_path,
            internal=internal,
            part=part,
            n_parts=n_parts,
            batch_size=batch_size,
        )
    elif ds_name == "celeba_ldm_256":
        outputs = sample_celeba(
            T=T,
            interpolation_alpha=interpolation_alpha,
            left_latent_path=left_latent_path,
            right_latent_path=right_latent_path,
            internal=internal,
            part=part,
            n_parts=n_parts,
            batch_size=batch_size,
        )
    elif ds_name == "imagenet_pixel_256":
        outputs = sample_ddpm256(
            T=T,
            interpolation_alpha=interpolation_alpha,
            left_latent_path=left_latent_path,
            right_latent_path=right_latent_path,
            part=part,
            n_parts=n_parts,
            internal=internal,
            batch_size=batch_size,
        )
    elif ds_name == "imagenet_dit_256":
        outputs = sample_dit(
            T=T,
            interpolation_alpha=interpolation_alpha,
            left_latent_path=left_latent_path,
            right_latent_path=right_latent_path,
            part=part,
            n_parts=n_parts,
            internal=internal,
            batch_size=batch_size,
        )
    else:
        raise ValueError(f"Unknown dataset name: {ds_name}")
    save_results(
        outputs=outputs,
        T=T,
        ds_name=ds_name,
        interpolation_alpha=interpolation_alpha,
        info=info,
        part=part,
        seed_path=seed_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_name", type=str, required=True)
    parser.add_argument("--T", type=int, required=True)
    parser.add_argument("--part", type=int, default=None)
    parser.add_argument("--n_parts", type=int, default=1)
    parser.add_argument("--internal", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--interpolation_alpha", type=float, required=True)
    parser.add_argument("--left_latent_path", type=str, required=True)
    parser.add_argument("--right_latent_path", type=str, required=True)
    parser.add_argument("--info", type=str, required=True)
    parser.add_argument("--seed_path", type=str, required=True)

    args = parser.parse_args()
    main(
        ds_name=args.ds_name,
        T=args.T,
        part=args.part,
        n_parts=args.n_parts,
        internal=args.internal,
        batch_size=args.batch_size,
        interpolation_alpha=args.interpolation_alpha,
        left_latent_path=args.left_latent_path,
        right_latent_path=args.right_latent_path,
        info=args.info,
        seed_path=args.seed_path,
    )
