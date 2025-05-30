import argparse
import os

import torch

from taba.models.dit.dit import CustomDiTPipeline
from taba.models.ldms.models import get_inv_scheduler, get_ldm_celeba, get_scheduler
from taba.models.ldms.sample_ldm import generate_latents as ldm_generate_latents
from taba.models.adm.models import get_ddpm_imagenet256, get_openai_cifar, get_openai_imagenet
from taba.models.adm.ddim import generate_latents

SAVE_DIR_PATH = "experiments/{swap_dir}invert{internal}{swap}/S{seed}/{ds_name}/T_{T}/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_BATCH_SIZE = 32
DEFAULT_SWAP_STEPS = lambda n: 2 if n in ["celeba_ldm_256", "imagenet_dit_256"] else 1


def save_results(
    outputs,
    T,
    ds_name,
    internal: bool,
    seed: int,
    part: int | None = None,
    swap_path: str | None = None,
    swap_type: str | None = None,
    swap_steps: int | None = None,
):
    if part is not None:
        suffix = f"_part{part}"
    else:
        suffix = ""
    internal_format = "_internal" if internal else ""
    swap_suffix = "_swap" if swap_path is not None else ""
    swap_dir = "swap_xts/" if swap_type == "xts" else "swap_eps/" if swap_type == "eps" else "no_swap/"
    if swap_path is not None:
        swap_suffix += f"{swap_steps}"
    dir_path = SAVE_DIR_PATH.format(
        swap_dir=swap_dir, internal=internal_format, swap=swap_suffix, seed=seed, ds_name=ds_name, T=T
    )
    os.makedirs(dir_path, exist_ok=True)
    print(f"Saving results to {dir_path}")

    for file_name, data in outputs.items():
        torch.save(data, dir_path + f"{file_name}{suffix}.pt")


def sample_ddpm(
    ddpm_model,
    diffusion_pipeline,
    diffusion_args,
    samples_path: str,
    part: int | None = None,
    n_parts: int = 1,
    internal: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    swap_eps_path: str | None = None,
    swap_steps: int | None = None,
    swap_xt_path: str | None = None,
):
    samples = torch.load(samples_path, weights_only=False)
    n_samples = samples.shape[0]
    if part is not None:
        samples = samples[(part * n_samples) // n_parts : ((part + 1) * n_samples) // n_parts]
        n_samples = samples.shape[0]
    samples = samples.to(DEVICE)

    if swap_xt_path is not None:
        swap_xt_all = torch.load(swap_xt_path, weights_only=False)
        swap_xt = {idx: swap_xt_all[-(idx + 1)] for idx in range(swap_steps)}
    else:
        swap_xt = {}

    if swap_eps_path is not None:
        swap_eps_all = torch.load(swap_eps_path, weights_only=False)
        swap_eps = {idx: swap_eps_all[-(idx + 1)] for idx in range(swap_steps)}
    else:
        swap_eps = {}

    outs_noising = generate_latents(
        ddim_generations=samples,
        batch_size=batch_size,
        diffusion_pipeline=diffusion_pipeline,
        ddim_model=ddpm_model,
        device=DEVICE,
        from_each_t=internal,
        swap_eps=swap_eps,
        swap_xt=swap_xt,
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
            "samples": samples,
            "latents": latents,
            "xts_noising": xts_noising,
            "t_eps_noising": t_eps_noising,
            "t_predxstart_noising": t_predxstart_noising,
        }
    else:
        return {
            "samples": samples,
            "latents": latents,
        }


def sample_cifar10(
    T: int,
    samples_path: str,
    internal: bool = False,
    part: int | None = None,
    n_parts: int = 1,
    batch_size: int = DEFAULT_BATCH_SIZE,
    swap_eps_path: str | None = None,
    swap_xt_path: str | None = None,
    swap_steps: int | None = None,
):
    model, diffusion, args = get_openai_cifar(steps=T, device=DEVICE)
    return sample_ddpm(
        ddpm_model=model,
        diffusion_pipeline=diffusion,
        diffusion_args=args,
        samples_path=samples_path,
        part=part,
        n_parts=n_parts,
        internal=internal,
        batch_size=batch_size,
        swap_eps_path=swap_eps_path,
        swap_xt_path=swap_xt_path,
        swap_steps=swap_steps,
    )


def sample_imagenet(
    T: int,
    samples_path: str,
    internal: bool = False,
    part: int | None = None,
    n_parts: int = 1,
    batch_size: int = DEFAULT_BATCH_SIZE,
    swap_eps_path: str | None = None,
    swap_xt_path: str | None = None,
    swap_steps: int | None = None,
):
    model, diffusion, args = get_openai_imagenet(steps=T, device=DEVICE)
    return sample_ddpm(
        ddpm_model=model,
        diffusion_pipeline=diffusion,
        diffusion_args=args,
        samples_path=samples_path,
        part=part,
        n_parts=n_parts,
        internal=internal,
        batch_size=batch_size,
        swap_eps_path=swap_eps_path,
        swap_xt_path=swap_xt_path,
        swap_steps=swap_steps,
    )


def sample_ddpm256(
    T: int,
    samples_path: str,
    part: int | None = None,
    n_parts: int = 1,
    internal: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    swap_eps_path: str | None = None,
    swap_xt_path: str | None = None,
    swap_steps: int | None = None,
):
    model, diffusion, args = get_ddpm_imagenet256(steps=T, device=DEVICE)
    return sample_ddpm(
        ddpm_model=model,
        diffusion_pipeline=diffusion,
        diffusion_args=args,
        samples_path=samples_path,
        part=part,
        n_parts=n_parts,
        internal=internal,
        batch_size=batch_size,
        swap_eps_path=swap_eps_path,
        swap_xt_path=swap_xt_path,
        swap_steps=swap_steps,
    )


def sample_celeba(
    T: int,
    samples_path: str,
    internal: bool = False,
    part: int | None = None,
    n_parts: int = 1,
    batch_size: int = DEFAULT_BATCH_SIZE,
    swap_eps_path: str | None = None,
    swap_steps: int | None = None,
):
    ldm_unet, _ = get_ldm_celeba(device=DEVICE)
    inv_scheduler = get_inv_scheduler(T=T)

    samples = torch.load(samples_path, weights_only=False)
    n_samples = samples.shape[0]
    if part is not None:
        samples = samples[(part * n_samples) // n_parts : ((part + 1) * n_samples) // n_parts]
        n_samples = samples.shape[0]
    samples = samples.to(DEVICE)

    if swap_eps_path is not None:
        swap_eps_all = torch.load(swap_eps_path, weights_only=False)
        swap_eps = {idx: swap_eps_all[-(idx + 1)] for idx in range(swap_steps)}
    else:
        swap_eps = {}

    noising_outs = ldm_generate_latents(
        samples=samples,
        diffusion_unet=ldm_unet,
        diffusion_scheduler=inv_scheduler,
        batch_size=batch_size,
        device=DEVICE,
        from_each_t=internal,
        swap_eps=swap_eps,
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
            "samples": samples,
            "latents": latents,
            "xts_noising": xts_noising,
            "t_eps_noising": t_eps_noising,
            "t_predxstart_noising": t_predxstart_noising,
        }
    else:
        return {
            "samples": samples,
            "latents": latents,
        }


def sample_dit(
    T: int,
    samples_path: str,
    internal: bool = False,
    part: int | None = None,
    n_parts: int = 1,
    batch_size: int = DEFAULT_BATCH_SIZE,
    swap_eps_path: str | None = None,
    swap_steps: int | None = None,
):
    dit_pipeline = CustomDiTPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.float16)
    dit_pipeline = dit_pipeline.to("cuda")
    samples = torch.load(samples_path, weights_only=False)
    class_ids = torch.load(samples_path.replace("samples.pt", "class_ids.pt"), weights_only=False)
    n_samples = samples.shape[0]
    if part is not None:
        samples = samples[(part * n_samples) // n_parts : ((part + 1) * n_samples) // n_parts]
        class_ids = class_ids[(part * n_samples) // n_parts : ((part + 1) * n_samples) // n_parts]
        n_samples = samples.shape[0]
    samples = samples.to(DEVICE)
    class_ids = class_ids.to(DEVICE)

    if swap_eps_path is not None:
        swap_eps_all = torch.load(swap_eps_path, weights_only=False)
        swap_eps = {idx: swap_eps_all[-(idx + 1)] for idx in range(swap_steps)}
    else:
        swap_eps = {}

    noising_outs = dit_pipeline.ddim_inverse(
        latents_x_0=samples,
        guidance_scale=1.0,
        batch_size=batch_size,
        num_inference_steps=T,
        class_labels=class_ids,
        from_each_t=internal,
        swap_eps=swap_eps,
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
            "samples": samples,
            "latents": latents,
            "xts_noising": xts_noising,
            "t_eps_noising": t_eps_noising,
            "t_predxstart_noising": t_predxstart_noising,
            "class_ids": class_ids,
        }
    else:
        return {
            "samples": samples,
            "latents": latents,
            "class_ids": class_ids,
        }


def main(
    ds_name: str,
    T: int,
    samples_path: str,
    seed: int,
    part: int | None = None,
    n_parts: int = 1,
    internal: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    swap_eps_path: str | None = None,
    swap_xt_path: str | None = None,
    swap_steps: int | None = None,
):
    if swap_eps_path is None and swap_xt_path is None:
        swap_steps = 0
    elif swap_steps is None:
        swap_steps = DEFAULT_SWAP_STEPS(ds_name)
    print(
        f"Sampling {ds_name} with T={T}, part={part}, n_parts={n_parts}, internal={internal}, batch_size={batch_size}, swap_eps_path={swap_eps_path}, swap_xt_path={swap_xt_path}, swap_steps={swap_steps}, seed={seed}"
    )
    if ds_name == "cifar_pixel_32":
        outputs = sample_cifar10(
            T=T,
            samples_path=samples_path,
            internal=internal,
            part=part,
            n_parts=n_parts,
            batch_size=batch_size,
            swap_eps_path=swap_eps_path,
            swap_xt_path=swap_xt_path,
            swap_steps=swap_steps,
        )
    elif ds_name == "imagenet_pixel_64":
        outputs = sample_imagenet(
            T=T,
            samples_path=samples_path,
            internal=internal,
            part=part,
            n_parts=n_parts,
            batch_size=batch_size,
            swap_eps_path=swap_eps_path,
            swap_xt_path=swap_xt_path,
            swap_steps=swap_steps,
        )
    elif ds_name == "imagenet_pixel_256":
        outputs = sample_ddpm256(
            T=T,
            samples_path=samples_path,
            part=part,
            n_parts=n_parts,
            internal=internal,
            batch_size=batch_size,
            swap_eps_path=swap_eps_path,
            swap_xt_path=swap_xt_path,
            swap_steps=swap_steps,
        )
    elif ds_name == "celeba_ldm_256":
        outputs = sample_celeba(
            T=T,
            samples_path=samples_path,
            internal=internal,
            part=part,
            n_parts=n_parts,
            batch_size=batch_size,
            swap_eps_path=swap_eps_path,
            swap_steps=swap_steps,
        )
    elif ds_name == "imagenet_dit_256":
        outputs = sample_dit(
            T=T,
            samples_path=samples_path,
            internal=internal,
            part=part,
            n_parts=n_parts,
            batch_size=batch_size,
            swap_eps_path=swap_eps_path,
            swap_steps=swap_steps,
        )
    else:
        raise ValueError(f"Unknown dataset name: {ds_name}")

    swap_type = "eps" if swap_eps_path is not None else "xts" if swap_xt_path is not None else None

    save_results(
        outputs=outputs,
        T=T,
        ds_name=ds_name,
        internal=internal,
        part=part,
        swap_path=swap_eps_path or swap_xt_path or None,
        swap_type=swap_type,
        seed=seed,
        swap_steps=swap_steps,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_name", type=str, required=True)
    parser.add_argument("--T", type=int, required=True)
    parser.add_argument("--samples_path", type=str, required=True)
    parser.add_argument("--part", type=int, default=None)
    parser.add_argument("--n_parts", type=int, default=1)
    parser.add_argument("--internal", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--swap_eps_path", type=str, default=None)
    parser.add_argument("--swap_xt_path", type=str, default=None)
    parser.add_argument("--swap_steps", type=int, default=None)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    main(
        ds_name=args.ds_name,
        T=args.T,
        samples_path=args.samples_path,
        seed=args.seed,
        part=args.part,
        n_parts=args.n_parts,
        internal=args.internal,
        batch_size=args.batch_size,
        swap_eps_path=args.swap_eps_path,
        swap_xt_path=args.swap_xt_path,
        swap_steps=args.swap_steps,
    )
