import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Dict

import torch
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, gather_object
from diffusers.training_utils import set_seed

from taba.metrics.angles_distances import reconstruction_error
from taba.metrics.correlation import get_top_k_corr_in_patches
from taba.metrics.normality import kl_div, stats_from_tensor
from taba.models.ldms.models import get_inv_scheduler, get_ldm_celeba, get_scheduler
from taba.models.ldms.sample_ldm import decode_image
from taba.models.ldms.sample_ldm import generate_latents as ldm_generate_latents
from taba.models.ldms.sample_ldm import generate_samples as ldm_generate_samples
from taba.models.ldms.sample_ldm import get_noises as ldm_get_noises

torch.backends.cuda.matmul.allow_tf32 = True
torch._inductor.config.conv_1x1_as_mm = True  # type: ignore
torch._inductor.config.coordinate_descent_tuning = True  # type: ignore
torch._inductor.config.epilogue_fusion = False  # type: ignore
torch._inductor.config.coordinate_descent_check_all_directions = True  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

START_TIME = time.strftime("%Y%m%d_%H%M%S")
SEED = 42
BATCH_SIZE_DEFAULT = 64
NUM_INFERENCE_STEPS_DEFAULT = 100
N_SAMPLES_DEFAULT = 1024


def save_tensor(accelerator: Accelerator, tensors: Dict[str, torch.Tensor], path_dir: str):
    if accelerator.is_main_process:
        for name, tensor in tensors.items():
            torch.save(tensor.cpu(), (Path(path_dir) / f"{name}.pt").resolve())
    accelerator.wait_for_everyone()


def sample_ldm(
    input_obj: torch.Tensor,
    T: int,
    device: torch.device,
    batch_size: int,
    do_inversion: bool,
    from_each_t: bool,
    forward_before_t: int | None = None,
    fixed_noise_generator: torch.Generator | None = None,
):
    ldm_unet, vae = get_ldm_celeba(device=device)
    scheduler = get_scheduler(T=T) if not do_inversion else get_inv_scheduler(T=T)

    if not do_inversion:
        output = ldm_generate_samples(
            noise=input_obj,
            diffusion_unet=ldm_unet,
            diffusion_scheduler=scheduler,
            batch_size=batch_size,
            device=device,
            from_each_t=from_each_t,
        )
        samples_decoded = decode_image(
            unet_out=output["samples"],
            vqvae=vae,
            batch_size=batch_size,
            device=device,
        )
        output["samples_decoded"] = samples_decoded
    else:
        output = ldm_generate_latents(
            samples=input_obj,
            diffusion_unet=ldm_unet,
            diffusion_scheduler=scheduler,
            batch_size=batch_size,
            device=device,
            from_each_t=from_each_t,
            forward_before_t=forward_before_t,
            fixed_noise_generator=fixed_noise_generator,
        )
    return output


def sample_multigpu(
    input_obj: torch.Tensor,
    T: int,
    batch_size: int,
    accelerator: Accelerator,
    do_inversion: bool,
    from_each_t: bool,
    forward_before_t: int | None = None,
    forward_seed: int = 42,
):
    all_input = list(range(input_obj.shape[0]))
    all_outputs = defaultdict(list)
    with accelerator.split_between_processes(all_input) as device_indices:
        fixed_noise_generator = (
            torch.Generator(device=accelerator.device).manual_seed(forward_seed + accelerator.process_index)
            if forward_before_t is not None
            else None
        )
        input_obj_device = torch.stack([input_obj[i] for i in device_indices])
        if accelerator.is_main_process:
            print(f"Rank {accelerator.process_index} | {(len(device_indices)/len(all_input)):.2f}")
        outputs = sample_ldm(
            input_obj=input_obj_device,
            T=T,
            device=accelerator.device,
            batch_size=batch_size,
            do_inversion=do_inversion,
            from_each_t=from_each_t,
            forward_before_t=forward_before_t,
            fixed_noise_generator=fixed_noise_generator,
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
    input_image_path: str | None,
    with_reconstruction: bool,
    seed: int,
    batch_size: int,
    internal: bool = False,
    start_time: str = START_TIME,
    forward_before_t: int | None = None,
    forward_seed: int = 42,
    save_dir: str | None = None,
):
    cmd = " ".join(sys.argv)
    if save_dir is None:
        save_dir = (
            f"experiments/celeba_ldm_256/invert/forward_seed{forward_seed}/forward_before{forward_before_t}/"
            f"{start_time}_"
            f"seed_{seed}_"
            f"T_{num_inference_steps}_"
            f"batch_size_{batch_size}_"
            f"_with_reconstruction_{with_reconstruction}"
            f"_internal_{internal}"
        )

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800 * 6))
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
        logging.info(f"T: {num_inference_steps}")
        logging.info(f"Seed: {seed}")
        logging.info(f"Device: {accelerator.device}")
        logging.info(f"Save dir: {save_dir}")
        logging.info(f"Input image path: {input_image_path}")
        logging.info(f"Forward before t: {forward_before_t}")
        logging.info(f"Forward seed: {forward_seed}")
        logging.info(f"Internal: {internal}")
        logging.info(f"With reconstruction: {with_reconstruction}")
        logging.info(f"Batch size: {batch_size}")
        logging.info(f"Command: {cmd}")
    set_seed(seed)

    if input_image_path is None or not os.path.exists(input_image_path):
        raise ValueError(f"Input image path {input_image_path} does not exist")
    else:
        image = torch.load(input_image_path, weights_only=False)

    save_tensor(accelerator=accelerator, tensors={"samples": image}, path_dir=save_dir)

    if accelerator.is_main_process:
        logging.info("Inversion: samples -> latents")
    inversion_outputs = sample_multigpu(
        input_obj=image,
        T=num_inference_steps,
        batch_size=batch_size,
        accelerator=accelerator,
        do_inversion=True,
        from_each_t=internal,
        forward_before_t=forward_before_t,
        forward_seed=forward_seed,
    )
    save_tensor(accelerator=accelerator, tensors=inversion_outputs, path_dir=save_dir)

    if with_reconstruction:
        if accelerator.is_main_process:
            logging.info("Reconstruction: latents -> image2")
        reconstruction_outputs = sample_multigpu(
            input_obj=inversion_outputs["latents"],
            T=num_inference_steps,
            batch_size=batch_size,
            accelerator=accelerator,
            do_inversion=False,
            from_each_t=internal,
        )
        reconstruction_outputs = {k.replace("samples", "samples2"): v for k, v in reconstruction_outputs.items()}
        save_tensor(accelerator=accelerator, tensors=reconstruction_outputs, path_dir=save_dir)

    # calculate metrics
    if accelerator.is_main_process:
        logging.info("Calculating metrics")
        metrics = {}
        metrics["_params"] = {}
        metrics["_params"]["seed"] = seed
        metrics["_params"]["num_inference_steps"] = num_inference_steps
        metrics["_params"]["batch_size"] = batch_size
        metrics["_params"]["n_samples"] = image.shape[0]
        metrics["_params"]["with_reconstruction"] = with_reconstruction
        metrics["_params"]["input_image_path"] = input_image_path
        metrics["_params"]["forward_before_t"] = forward_before_t
        metrics["_params"]["forward_seed"] = forward_seed
        metrics["_params"]["internal"] = internal
        metrics["_params"]["save_dir"] = save_dir
        metrics["_params"]["cmd"] = cmd
        metrics["metrics"] = {}

        metrics["metrics"]["correlation"] = get_top_k_corr_in_patches(inversion_outputs["latents"], top_k=20)
        metrics["metrics"]["latent_stats"] = stats_from_tensor(inversion_outputs["latents"])
        if with_reconstruction:
            metrics["metrics"]["reconstruction_error_images"] = reconstruction_error(
                reconstruction_outputs["samples2_decoded"], inversion_outputs["samples_decoded"]
            )
        logging.info(metrics)

        with open(Path(save_dir) / "metrics.json", "w") as f:
            json.dump(metrics, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_inference_steps", type=int, default=NUM_INFERENCE_STEPS_DEFAULT)
    parser.add_argument("--input_image_path", type=str, default=None)
    parser.add_argument("--with_reconstruction", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument("--internal", action="store_true")
    parser.add_argument("--forward_before_t", type=int, default=None)
    parser.add_argument("--forward_seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()
    main(
        num_inference_steps=args.num_inference_steps,
        input_image_path=args.input_image_path,
        with_reconstruction=args.with_reconstruction,
        seed=args.seed,
        batch_size=args.batch_size,
        forward_before_t=args.forward_before_t,
        forward_seed=args.forward_seed,
        internal=args.internal,
        save_dir=args.save_dir,
    )
