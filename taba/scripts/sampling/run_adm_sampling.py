import argparse
import json
import logging
import os
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
from taba.models.adm.ddim import generate_latents, generate_noises, generate_samples
from taba.models.adm.models import get_ddpm_imagenet256, get_openai_cifar, get_openai_imagenet

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

ADM_INIT_FUNCTIONS = {
    "cifar_pixel_32": get_openai_cifar,
    "imagenet_pixel_64": get_openai_imagenet,
    "imagenet_pixel_256": get_ddpm_imagenet256,
}


def save_tensor(accelerator: Accelerator, tensors: Dict[str, torch.Tensor], path_dir: str):
    if accelerator.is_main_process:
        for name, tensor in tensors.items():
            torch.save(tensor.cpu(), (Path(path_dir) / f"{name}.pt").resolve())
    accelerator.wait_for_everyone()


def sample_adm(
    model_name: str,
    input_obj: torch.Tensor,
    T: int,
    device: torch.device,
    batch_size: int,
    do_inversion: bool,
    from_each_t: bool,
):
    model, pipeline, args = ADM_INIT_FUNCTIONS[model_name](steps=T, device=device)
    n_samples = input_obj.shape[0]

    if not do_inversion:
        output = generate_samples(
            random_noises=input_obj,
            number_of_samples=n_samples,
            batch_size=batch_size,
            diffusion_pipeline=pipeline,
            ddim_model=model,
            diffusion_args=args,
            device=device,
            from_each_t=from_each_t,
        )
    else:
        output = generate_latents(
            ddim_generations=input_obj,
            batch_size=batch_size,
            diffusion_pipeline=pipeline,
            ddim_model=model,
            device=device,
            from_each_t=from_each_t,
        )
    return output


def sample_multigpu(
    model_name: str,
    input_obj: torch.Tensor,
    T: int,
    batch_size: int,
    accelerator: Accelerator,
    do_inversion: bool,
    from_each_t: bool,
):
    all_input = list(range(input_obj.shape[0]))
    all_outputs = defaultdict(list)
    with accelerator.split_between_processes(all_input) as device_indices:
        input_obj_device = torch.stack([input_obj[i] for i in device_indices])
        if accelerator.is_main_process:
            print(f"Rank {accelerator.process_index} | {(len(device_indices)/len(all_input)):.2f}")
        outputs = sample_adm(
            model_name=model_name,
            input_obj=input_obj_device,
            T=T,
            device=accelerator.device,
            batch_size=batch_size,
            do_inversion=do_inversion,
            from_each_t=from_each_t,
        )
        for key, value in outputs.items():
            all_outputs[key].append(value.cpu())
    accelerator.wait_for_everyone()
    for obj_name, obj_value in all_outputs.items():
        vals = gather_object(obj_value)
        all_outputs[obj_name] = torch.cat(vals, dim=0)
    return all_outputs


def main(
    model_name: str,
    num_inference_steps: int,
    input_noise_path: str | None,
    input_image_path: str | None,
    with_inversion: bool,
    with_reconstruction: bool,
    seed: int,
    batch_size: int,
    n_samples: int,
    start_time: str = START_TIME,
    internal: bool = False,
    n_parts: int = 1,
    part_idx: int = 0,
    save_dir: str | None = None,
):
    if with_reconstruction is True and with_inversion is False:
        raise ValueError("Reconstruction cannot be done without inversion")

    INTERNAL_DIR = "internal/" if internal else ""
    if save_dir is None:
        save_dir = (
            f"experiments/{model_name}/{INTERNAL_DIR}sampling_invert_reconstruction/"
            f"{start_time}_"
            f"seed_{seed}_"
            f"T_{num_inference_steps}_"
            f"batch_size_{batch_size}_"
            f"n_samples_{n_samples}_"
            f"_with_inversion_{with_inversion}_"
            f"_with_reconstruction_{with_reconstruction}"
        )
    if n_parts > 1:
        save_dir = save_dir + f"_part_{part_idx}"

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800 * 6))
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
        logging.info(f"Model name: {model_name}")
        logging.info(f"T: {num_inference_steps}")
        logging.info(f"Seed: {seed}")
        logging.info(f"Device: {accelerator.device}")
        logging.info(f"Save dir: {save_dir}")
        logging.info(f"Input noise path: {input_noise_path}")
        logging.info(f"Input image path: {input_image_path}")
        logging.info(f"With inversion: {with_inversion}")
        logging.info(f"With reconstruction: {with_reconstruction}")
        logging.info(f"Batch size: {batch_size}")
        logging.info(f"N samples: {n_samples}")
        logging.info(f"N parts: {n_parts}")
        logging.info(f"Part idx: {part_idx}")
    set_seed(seed)

    if input_noise_path is not None and os.path.exists(input_noise_path):
        noise = torch.load(input_noise_path, weights_only=False)
        logging.warning("Using input noise path, n_samples will be ignored")
    elif input_noise_path is not None:
        raise ValueError(f"Input noise path {input_noise_path} does not exist")
    else:
        _, _, args = ADM_INIT_FUNCTIONS[model_name](steps=num_inference_steps, device="cpu")
        noise = generate_noises(number_of_samples=n_samples, seed=seed, device="cpu", diffusion_args=args)

    if input_image_path is not None and os.path.exists(input_image_path):
        sampling_outputs = {"samples": torch.load(input_image_path, weights_only=False)}
        logging.warning("Using input image path, n_samples will be ignored")
    elif input_image_path is not None:
        raise ValueError(f"Input image path {input_image_path} does not exist")
    else:
        sampling_outputs = None

    if n_parts > 1:
        assert n_samples % n_parts == 0, "n_samples must be divisible by n_parts"
        n_per_part = n_samples // n_parts
        noise = noise[part_idx * n_per_part : (part_idx + 1) * n_per_part]
        if sampling_outputs is not None:
            sampling_outputs["samples"] = sampling_outputs["samples"][part_idx * n_per_part : (part_idx + 1) * n_per_part]

    if accelerator.is_main_process:
        logging.info(f"Noise shape: {noise.shape}")
    save_tensor(accelerator=accelerator, tensors={"noise": noise}, path_dir=save_dir)

    # sampling noise -> image
    if sampling_outputs is None:
        if accelerator.is_main_process:
            logging.info("Sampling: noise -> image")
        sampling_outputs = sample_multigpu(
            model_name=model_name,
            input_obj=noise,
            T=num_inference_steps,
            batch_size=batch_size,
            accelerator=accelerator,
            do_inversion=False,
            from_each_t=internal,
        )
    save_tensor(accelerator=accelerator, tensors=sampling_outputs, path_dir=save_dir)

    if with_inversion:
        if accelerator.is_main_process:
            logging.info("Inversion: noise -> latents")
        inversion_outputs = sample_multigpu(
            model_name=model_name,
            input_obj=sampling_outputs["samples"],
            T=num_inference_steps,
            batch_size=batch_size,
            accelerator=accelerator,
            do_inversion=True,
            from_each_t=internal,
        )
        save_tensor(accelerator=accelerator, tensors=inversion_outputs, path_dir=save_dir)

    if with_reconstruction:
        if accelerator.is_main_process:
            logging.info("Reconstruction: latents -> image2")
        reconstruction_outputs = sample_multigpu(
            model_name=model_name,
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
        metrics["_params"]["input_image_path"] = input_image_path
        metrics["_params"]["output_dir"] = save_dir
        metrics["_params"]["batch_size"] = batch_size
        metrics["_params"]["n_samples"] = n_samples
        metrics["_params"]["with_inversion"] = with_inversion
        metrics["_params"]["with_reconstruction"] = with_reconstruction
        metrics["_params"]["input_noise_path"] = input_noise_path
        metrics["_params"]["n_parts"] = n_parts
        metrics["_params"]["part_idx"] = part_idx
        metrics["_params"]["model_name"] = model_name
        metrics["metrics"] = {}

        if with_inversion:
            metrics["metrics"]["kl_div_noise_latent"] = kl_div(noise, inversion_outputs["latents"])
            metrics["metrics"]["reconstruction_error_noise"] = reconstruction_error(
                noise, inversion_outputs["latents"]
            )
            metrics["metrics"]["correlation"] = get_top_k_corr_in_patches(inversion_outputs["latents"], top_k=20)
            metrics["metrics"]["latent_stats"] = stats_from_tensor(inversion_outputs["latents"])
            metrics["metrics"]["noise_stats"] = stats_from_tensor(noise)

        if with_reconstruction:
            metrics["metrics"]["reconstruction_error_images"] = reconstruction_error(
                reconstruction_outputs["samples2"], sampling_outputs["samples"]
            )
        logging.info(metrics)

        with open(Path(save_dir) / "metrics.json", "w") as f:
            json.dump(metrics, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, choices=ADM_INIT_FUNCTIONS.keys())
    parser.add_argument("--num_inference_steps", type=int, default=NUM_INFERENCE_STEPS_DEFAULT)
    parser.add_argument("--input_noise_path", type=str, default=None)
    parser.add_argument("--input_image_path", type=str, default=None)
    parser.add_argument("--with_inversion", action="store_true")
    parser.add_argument("--with_reconstruction", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument("--n_samples", type=int, default=N_SAMPLES_DEFAULT)
    parser.add_argument("--internal", action="store_true")
    parser.add_argument("--n_parts", type=int, default=1)
    parser.add_argument("--part_idx", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()
    main(
        model_name=args.model_name,
        num_inference_steps=args.num_inference_steps,
        input_noise_path=args.input_noise_path,
        input_image_path=args.input_image_path,
        with_inversion=args.with_inversion,
        with_reconstruction=args.with_reconstruction,
        seed=args.seed,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        internal=args.internal,
        n_parts=args.n_parts,
        part_idx=args.part_idx,
        save_dir=args.save_dir,
    )
