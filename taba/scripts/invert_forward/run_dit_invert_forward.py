import argparse
import json
import logging
import os
import random
import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Dict

import torch
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, gather_object
from diffusers.training_utils import set_seed
from diffusers.utils.logging import disable_progress_bar, enable_progress_bar

from taba.metrics.angles_distances import reconstruction_error
from taba.metrics.correlation import get_top_k_corr_in_patches
from taba.metrics.normality import stats_from_tensor
from taba.models.dit.constants import DIT_IMAGENET_MAP
from taba.models.dit.dit import CustomDiTPipeline

torch.backends.cuda.matmul.allow_tf32 = True
torch._inductor.config.conv_1x1_as_mm = True  # type: ignore
torch._inductor.config.coordinate_descent_tuning = True  # type: ignore
torch._inductor.config.epilogue_fusion = False  # type: ignore
torch._inductor.config.coordinate_descent_check_all_directions = True  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

START_TIME = time.strftime("%Y%m%d_%H%M%S")
SEED = 42
NOISES_PER_PROMPT_DEFAULT = 4
N_PROMPTS_DEFAULT = 128
BATCH_SIZE_DEFAULT = 64
NUM_INFERENCE_STEPS_DEFAULT = 50
GUIDANCE_SCALE_DEFAULT = 1.0
COND_SEED_DEFAULT = 10


def load_cond(n_conds: int, n_noises_per_cond: int, dit_model: CustomDiTPipeline, cond_seed: int) -> torch.Tensor:
    random.seed(cond_seed)
    classes_cond = random.choices(list(DIT_IMAGENET_MAP.values()), k=n_conds)
    class_ids = torch.tensor(
        dit_model.get_label_ids(label=[class_cond for class_cond in classes_cond for _ in range(n_noises_per_cond)])
    )
    return class_ids


def save_tensor(accelerator: Accelerator, tensors: Dict[str, torch.Tensor], path_dir: str):
    if accelerator.is_main_process:
        for name, tensor in tensors.items():
            torch.save(tensor.cpu(), (Path(path_dir) / f"{name}.pt").resolve())
    accelerator.wait_for_everyone()


def denoise_invert_multigpu(
    latent: torch.Tensor,
    conds: torch.Tensor,
    pipeline: CustomDiTPipeline,
    accelerator: Accelerator,
    batch_size: int,
    num_inference_steps: int,
    guidance_scale: float,
    run_inversion: bool,
    from_each_t: bool,
    swaps_per_t: dict[int, torch.Tensor] = {},
    swap_type: str | None = None,
    forward_before_t: int | None = None,
    forward_seed: int = 42,
):
    conds_indices = list(range(len(conds)))
    all_outputs = defaultdict(list)
    with accelerator.split_between_processes(conds_indices) as device_indices:
        fixed_noise_generator = (
            torch.Generator(device=accelerator.device).manual_seed(forward_seed + accelerator.process_index)
            if forward_before_t is not None
            else None
        )
        conds_device = torch.stack([conds[i] for i in device_indices])
        latent_device = torch.stack([latent[i] for i in device_indices])
        swaps_per_t_device = (
            {t: torch.stack([swaps_per_t[t][i] for i in device_indices]) for t in swaps_per_t.keys()}
            if swaps_per_t != {}
            else {}
        )
        if accelerator.is_main_process:
            print(f"Rank {accelerator.process_index} | {(len(device_indices)/len(conds_indices)):.2f}")
        if run_inversion:
            swap_xt, swap_eps = {}, {}
            if swap_type == "xt":
                swap_xt = swaps_per_t_device
            elif swap_type == "eps":
                swap_eps = swaps_per_t_device

            outputs = pipeline.ddim_inverse(
                latents_x_0=latent_device,
                guidance_scale=guidance_scale,
                batch_size=batch_size,
                num_inference_steps=num_inference_steps,
                class_labels=conds_device,
                from_each_t=from_each_t,
                swap_eps=swap_eps,
                swap_xt=swap_xt,
                forward_before_t=forward_before_t,
                fixed_noise_generator=fixed_noise_generator,
            )
        else:
            outputs = pipeline.ddim(
                latents_x_T=latent_device,
                guidance_scale=guidance_scale,
                batch_size=batch_size,
                num_inference_steps=num_inference_steps,
                class_labels=conds_device,
                from_each_t=from_each_t,
            )
        for key, value in outputs.items():
            all_outputs[key].append(value.cpu())
    accelerator.wait_for_everyone()
    for obj_name, obj_value in all_outputs.items():
        vals = gather_object(obj_value)
        all_outputs[obj_name] = torch.cat(vals, dim=1 if obj_name.startswith("all_t") else 0)
    return all_outputs


def main(
    num_inference_steps: int,
    input_image_path: str,
    input_cond_path: str,
    with_reconstruction: bool,
    seed: int,
    batch_size: int,
    internal: bool,
    guidance_scale: float,
    swap_path: str | None,
    swap_before_t: int | None,
    start_time: str = START_TIME,
    swap_type: str | None = None,
    forward_before_t: int | None = None,
    forward_seed: int = 42,
    save_dir: str | None = None,
):
    if save_dir is None:
        save_dir = (
            f"experiments/dit/invert/forward_seed{seed}/forward_before{forward_before_t}/"
            f"{start_time}_"
            f"seed_{seed}_"
            f"forward_seed_{forward_seed}_"
            f"T_{num_inference_steps}_"
            f"batch_size_{batch_size}_"
            f"guidance_scale_{guidance_scale}_"
            f"with_reconstruction_{with_reconstruction}_"
            f"internal_{internal}"
        )

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800 * 6))
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
        logging.info(f"Seed: {seed}")
        logging.info(f"Device: {accelerator.device}")
        logging.info(f"Num inference steps: {num_inference_steps}")
        logging.info(f"Batch size: {batch_size}")
        logging.info(f"Save dir: {save_dir}")
        logging.info(f"Input image path: {input_image_path}")
        logging.info(f"Input cond path: {input_cond_path}")
        logging.info(f"Swap path: {swap_path}")
        logging.info(f"Swap before t: {swap_before_t}")
        logging.info(f"Swap type: {swap_type}")
        logging.info(f"Forward before t: {forward_before_t}")
        logging.info(f"Forward seed: {forward_seed}")
        logging.info(f"With reconstruction: {with_reconstruction}")
        logging.info(f"Internal: {internal}")
    set_seed(seed)

    disable_progress_bar()
    pipe: CustomDiTPipeline = CustomDiTPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.float16)
    enable_progress_bar()
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to(accelerator.device)

    if input_image_path is None or not os.path.exists(input_image_path):
        raise ValueError(f"Input image path {input_image_path} does not exist")
    else:
        image = torch.load(input_image_path, weights_only=False)
    if input_cond_path is None or not os.path.exists(input_cond_path):
        raise ValueError(f"Input cond path {input_cond_path} does not exist")
    else:
        conds = torch.load(input_cond_path, weights_only=False)
    if swap_before_t is not None and swap_before_t > 0:
        assert swap_type is not None, "swap_type must be provided"
        assert swap_path is not None and os.path.exists(
            swap_path
        ), f"Swap path {swap_path} does not exist, but {swap_before_t=} is greater than 0"
        swap_tensor = torch.load(swap_path, weights_only=False)
        swap_per_t_inc = 1 if swap_type == "eps" else 2
        swaps_per_t = {idx: swap_tensor[-(idx + swap_per_t_inc)] for idx in range(swap_before_t)}
    else:
        swaps_per_t = {}

    if accelerator.is_main_process:
        logging.info(f"Image shape: {image.shape}")
        logging.info(f"Conds shape: {conds.shape}")
    save_tensor(accelerator=accelerator, tensors={"samples": image, "conds": conds}, path_dir=save_dir)

    decoded_samples = None
    if accelerator.is_main_process:
        logging.info("Decoding samples")
        decoded_samples = pipe.vae_decode(latents=image, batch_size=batch_size)
    accelerator.wait_for_everyone()
    save_tensor(accelerator=accelerator, tensors={"decoded_samples": decoded_samples}, path_dir=save_dir)

    if accelerator.is_main_process:
        logging.info("Inversion image -> latent")
    inversion_outputs = denoise_invert_multigpu(
        latent=image,
        conds=conds,
        pipeline=pipe,
        accelerator=accelerator,
        batch_size=batch_size,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        from_each_t=internal,
        run_inversion=True,
        swaps_per_t=swaps_per_t,
        swap_type=swap_type,
        forward_before_t=forward_before_t,
        forward_seed=forward_seed,
    )
    save_tensor(accelerator=accelerator, tensors=inversion_outputs, path_dir=save_dir)

    if with_reconstruction:
        if accelerator.is_main_process:
            logging.info("Reconstruction latent -> image2")
        denoising_outputs2 = denoise_invert_multigpu(
            latent=inversion_outputs["latents"],
            conds=conds,
            pipeline=pipe,
            accelerator=accelerator,
            batch_size=batch_size,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            from_each_t=internal,
            run_inversion=False,
        )
        denoising_outputs2 = {k.replace("samples", "samples2"): v for k, v in denoising_outputs2.items()}
        save_tensor(accelerator=accelerator, tensors=denoising_outputs2, path_dir=save_dir)

        decoded_samples2 = None
        if accelerator.is_main_process:
            logging.info("Decoding reconstructions")
            decoded_samples2 = pipe.vae_decode(latents=denoising_outputs2["samples2"], batch_size=batch_size)
        accelerator.wait_for_everyone()
        save_tensor(accelerator=accelerator, tensors={"decoded_samples2": decoded_samples2}, path_dir=save_dir)

    # calculate metrics
    if accelerator.is_main_process:
        logging.info("Calculating metrics")

        latents = inversion_outputs["latents"].cpu()
        images = decoded_samples.cpu()

        metrics = {}
        metrics["_params"] = {}
        metrics["_params"]["seed"] = seed
        metrics["_params"]["batch_size"] = batch_size
        metrics["_params"]["num_inference_steps"] = num_inference_steps
        metrics["_params"]["guidance_scale"] = guidance_scale
        metrics["_params"]["input_image_path"] = input_image_path
        metrics["_params"]["input_cond_path"] = input_cond_path
        metrics["_params"]["swap_path"] = swap_path
        metrics["_params"]["swap_before_t"] = swap_before_t
        metrics["_params"]["swap_type"] = swap_type
        metrics["_params"]["with_reconstruction"] = with_reconstruction
        metrics["_params"]["internal"] = internal
        metrics["_params"]["forward_before_t"] = forward_before_t
        metrics["_params"]["forward_seed"] = forward_seed

        metrics["metrics"] = {}
        metrics["metrics"]["correlation"] = get_top_k_corr_in_patches(latents, top_k=20)
        metrics["metrics"]["latent_stats"] = stats_from_tensor(latents)

        if with_reconstruction:
            images2 = decoded_samples2.cpu()
            metrics["metrics"]["reconstruction_error_images"] = reconstruction_error(images, images2)

        logging.info(metrics)
        with open(Path(save_dir) / "metrics.json", "w") as f:
            json.dump(metrics, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_inference_steps", type=int, default=NUM_INFERENCE_STEPS_DEFAULT)
    parser.add_argument("--input_image_path", type=str, required=True)
    parser.add_argument("--input_cond_path", type=str, required=True)
    parser.add_argument("--with_reconstruction", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument("--internal", action="store_true")
    parser.add_argument("--swap_path", type=str, default=None)
    parser.add_argument("--swap_before_t", type=int, default=None)
    parser.add_argument("--swap_type", type=str, choices=["eps", "xt"], default=None)
    parser.add_argument("--guidance_scale", type=float, default=GUIDANCE_SCALE_DEFAULT)
    parser.add_argument("--forward_before_t", type=int, default=None)
    parser.add_argument("--forward_seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()
    main(
        num_inference_steps=args.num_inference_steps,
        input_image_path=args.input_image_path,
        input_cond_path=args.input_cond_path,
        with_reconstruction=args.with_reconstruction,
        seed=args.seed,
        batch_size=args.batch_size,
        internal=args.internal,
        guidance_scale=args.guidance_scale,
        swap_path=args.swap_path,
        swap_before_t=args.swap_before_t,
        swap_type=args.swap_type,
        forward_before_t=args.forward_before_t,
        forward_seed=args.forward_seed,
        save_dir=args.save_dir,
    )
