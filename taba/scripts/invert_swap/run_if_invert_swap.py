import argparse
import json
import logging
import os
import pickle
import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, gather_object
from diffusers.training_utils import set_seed
from diffusers.utils.logging import disable_progress_bar, enable_progress_bar
from einops import rearrange
from tqdm import tqdm

from taba.metrics.angles_distances import reconstruction_error
from taba.metrics.correlation import get_top_k_corr_in_patches
from taba.metrics.normality import kl_div, stats_from_tensor
from taba.models.deepfloyd_if.pipeline import CustomIFPipeline, initialize_if_ddim_pipeline

torch.backends.cuda.matmul.allow_tf32 = True
torch._inductor.config.conv_1x1_as_mm = True  # type: ignore
torch._inductor.config.coordinate_descent_tuning = True  # type: ignore
torch._inductor.config.epilogue_fusion = False  # type: ignore
torch._inductor.config.coordinate_descent_check_all_directions = True  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

START_TIME = time.strftime("%Y%m%d_%H%M%S")
SEED = 42
BATCH_SIZE_DEFAULT = 16
NUM_INFERENCE_STEPS_DEFAULT = 50
GUIDANCE_SCALE_DEFAULT = 1.0


def _rearrange_t_to_b(tensor: torch.Tensor) -> torch.Tensor:
    return rearrange(tensor, "t b c h w -> b t c h w")


def _rearrange_b_to_t(tensor: torch.Tensor) -> torch.Tensor:
    return rearrange(tensor, "b t c h w -> t b c h w")


def save_prompts_pkl(prompts: list[str], path_dir: str):
    with open(Path(path_dir) / "prompts.pkl", "wb") as f:
        pickle.dump(prompts, f)


def load_prompts_pkl(path_pkl: str):
    with open(Path(path_pkl), "rb") as f:
        prompts = pickle.load(f)
    return prompts


def prepare_noise(n_all_prompts: int, pipeline: CustomIFPipeline, seed: int):
    noises = pipeline.prepare_latents(n_images=n_all_prompts, generator=torch.manual_seed(seed))
    return noises


def save_tensor(accelerator: Accelerator, tensors: Dict[str, torch.Tensor], path_dir: str):
    if accelerator.is_main_process:
        for name, tensor in tensors.items():
            torch.save(tensor.cpu(), (Path(path_dir) / f"{name}.pt").resolve())
    accelerator.wait_for_everyone()


def noise_denoise_batched(
    pipeline: CustomIFPipeline,
    prompts: list[str],
    noise: torch.Tensor,
    batch_size: int,
    num_inference_steps: int,
    device: torch.device,
    guidance_scale: float,
    run_inversion: bool = False,
    from_each_t: bool = False,
    swap_xt: dict[int, torch.Tensor] = {},
    swap_eps: dict[int, torch.Tensor] = {},
):
    all_outputs = defaultdict(list)
    with tqdm(total=len(prompts)) as pbar:
        for batch_num, batch_start in enumerate(range(0, len(prompts), batch_size)):
            prompt = prompts[batch_start : batch_start + batch_size]
            latent = noise[batch_start : batch_start + batch_size].to(device)
            swap_xt_batch = {t: swap_xt[t][batch_start : batch_start + batch_size] for t in swap_xt.keys()}
            swap_eps_batch = {t: swap_eps[t][batch_start : batch_start + batch_size] for t in swap_eps.keys()}

            if run_inversion:
                outputs = pipeline.invert(
                    prompt=prompt,
                    image=latent,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    eta=0.0,
                    clean_caption=False,
                    from_each_t=from_each_t,
                    swap_xt=swap_xt_batch,
                    swap_eps=swap_eps_batch,
                )
            else:
                outputs = pipeline.sample(
                    prompt=prompt,
                    noise=latent,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    eta=0.0,
                    clean_caption=False,
                    from_each_t=from_each_t,
                )
            for key, value in outputs.items():
                value = _rearrange_t_to_b(value) if key.startswith("all_t") else value
                all_outputs[key].append(value)
            pbar.update(len(prompt))
    for key, value in all_outputs.items():
        value = torch.cat(value, dim=0)
        value = _rearrange_b_to_t(value) if key.startswith("all_t") else value
        all_outputs[key] = value
    return all_outputs


def denoise_invert_multigpu(
    latent: torch.Tensor,
    prompts: list[str],
    pipeline: CustomIFPipeline,
    accelerator: Accelerator,
    batch_size: int,
    num_inference_steps: int,
    guidance_scale: float,
    run_inversion: bool,
    from_each_t: bool,
    swaps_per_t: dict[int, torch.Tensor] = {},
    swap_type: str | None = None,
):
    prompts_indices = list(range(len(prompts)))
    all_outputs = defaultdict(list)
    with accelerator.split_between_processes(prompts_indices) as device_indices:
        prompts_device = [prompts[i] for i in device_indices]
        latent_device = torch.stack([latent[i] for i in device_indices])
        swaps_per_t_device = (
            {t: torch.stack([swaps_per_t[t][i] for i in device_indices]) for t in swaps_per_t.keys()}
            if swaps_per_t != {}
            else {}
        )
        if accelerator.is_main_process:
            print(f"Rank {accelerator.process_index} | {(len(device_indices)/len(prompts_indices)):.2f}")
        swap_xt, swap_eps = {}, {}
        if run_inversion is True:
            if swap_type == "xt":
                swap_xt = swaps_per_t_device
            elif swap_type == "eps":
                swap_eps = swaps_per_t_device

        outputs = noise_denoise_batched(
            pipeline=pipeline,
            prompts=prompts_device,
            noise=latent_device,
            batch_size=batch_size,
            num_inference_steps=num_inference_steps,
            device=accelerator.device,
            guidance_scale=guidance_scale,
            run_inversion=run_inversion,
            from_each_t=from_each_t,
            swap_xt=swap_xt,
            swap_eps=swap_eps,
        )
        for key, value in outputs.items():
            all_outputs[key].append(value.cpu())
    accelerator.wait_for_everyone()
    for obj_name, obj_value in all_outputs.items():
        vals = gather_object(obj_value)
        all_outputs[obj_name] = torch.cat(vals, dim=1 if obj_name.startswith("all_t") else 0)
    return all_outputs


def main(
    seed: int,
    batch_size: int,
    num_inference_steps: int,
    guidance_scale: float,
    input_samples_path: str,
    input_prompts_path: str,
    internal: bool,
    with_reconstruction: bool,
    swap_path: str | None,
    swap_before_t: int | None,
    swap_type: str | None = None,
    start_time: str = START_TIME,
    save_dir: str | None = None,
):
    if save_dir is None:
        save_dir = (
            f"experiments/deepfloyd_if/invert/swap_{swap_type}_before{swap_before_t}/"
            f"{start_time}_"
            f"seed_{seed}_"
            f"batch_size_{batch_size}_"
            f"num_inference_steps_{num_inference_steps}_"
            f"guidance_scale_{guidance_scale}_"
            f"internal_{internal}"
            f"with_reconstruction_{with_reconstruction}"
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
        logging.info(f"Input samples path: {input_samples_path}")
        logging.info(f"Input prompts path: {input_prompts_path}")
        logging.info(f"With reconstruction: {with_reconstruction}")
        logging.info(f"Swap path: {swap_path}")
        logging.info(f"Swap before t: {swap_before_t}")
        logging.info(f"Swap type: {swap_type}")
        logging.info(f"Internal: {internal}")
        logging.info(f"Start time: {start_time}")
    set_seed(seed)

    disable_progress_bar()
    pipe = initialize_if_ddim_pipeline()
    enable_progress_bar()
    pipe.set_progress_bar_config(disable=True)

    assert os.path.exists(input_samples_path), f"Input samples path {input_samples_path} does not exist"
    assert os.path.exists(input_prompts_path), f"Input prompts path {input_prompts_path} does not exist"

    prompts = load_prompts_pkl(input_prompts_path)
    samples = torch.load(input_samples_path, weights_only=False)
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
        logging.info(f"Number of prompts: {len(prompts)}")
        logging.info(f"Samples shape: {samples.shape}")
    save_tensor(accelerator=accelerator, tensors={"samples": samples}, path_dir=save_dir)
    save_prompts_pkl(prompts, save_dir)

    # inversion image -> latent
    if accelerator.is_main_process:
        logging.info("Inversion image -> latent")
    inversion_outputs = denoise_invert_multigpu(
        latent=samples,
        prompts=prompts,
        pipeline=pipe,
        accelerator=accelerator,
        batch_size=batch_size,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        run_inversion=True,
        from_each_t=internal,
        swaps_per_t=swaps_per_t,
        swap_type=swap_type,
    )
    save_tensor(accelerator=accelerator, tensors=inversion_outputs, path_dir=save_dir)

    if not with_reconstruction:
        if accelerator.is_main_process:
            logging.info("Done")
        return
    if accelerator.is_main_process:
        logging.info("Reconstruction latent -> image2")
    images2 = denoise_invert_multigpu(
        latent=inversion_outputs["latents"],
        prompts=prompts,
        pipeline=pipe,
        accelerator=accelerator,
        batch_size=batch_size,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        from_each_t=internal,
        run_inversion=False,
    )
    images2 = {k.replace("samples", "samples2"): v for k, v in images2.items()}
    save_tensor(accelerator=accelerator, tensors=images2, path_dir=save_dir)

    # calculate metrics
    if accelerator.is_main_process:
        logging.info("Calculating metrics")
        metrics = {}
        metrics["_params"] = {}
        metrics["_params"]["seed"] = seed
        metrics["_params"]["batch_size"] = batch_size
        metrics["_params"]["num_inference_steps"] = num_inference_steps
        metrics["_params"]["guidance_scale"] = guidance_scale
        metrics["_params"]["input_samples_path"] = input_samples_path
        metrics["_params"]["input_prompts_path"] = input_prompts_path
        metrics["_params"]["internal"] = internal
        metrics["_params"]["with_reconstruction"] = with_reconstruction

        metrics["metrics"] = {}
        latents = inversion_outputs["latents"]
        metrics["metrics"]["correlation"] = get_top_k_corr_in_patches(latents, top_k=20)
        metrics["metrics"]["latent_stats"] = stats_from_tensor(latents)
        if with_reconstruction:
            images = samples
            images2 = images2["samples2"]
            metrics["metrics"]["reconstruction_error_images"] = reconstruction_error(images, images2)

        logging.info(metrics)
        with open(Path(save_dir) / "metrics.json", "w") as f:
            json.dump(metrics, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument("--num_inference_steps", type=int, default=NUM_INFERENCE_STEPS_DEFAULT)
    parser.add_argument("--guidance_scale", type=float, default=GUIDANCE_SCALE_DEFAULT)
    parser.add_argument("--input_samples_path", type=str)
    parser.add_argument("--input_prompts_path", type=str)
    parser.add_argument("--internal", action="store_true")
    parser.add_argument("--with_reconstruction", action="store_true")
    parser.add_argument("--swap_path", type=str, default=None)
    parser.add_argument("--swap_before_t", type=int, default=None)
    parser.add_argument("--swap_type", type=str, choices=["eps", "xt"], default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()
    main(
        seed=args.seed,
        batch_size=args.batch_size,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        input_samples_path=args.input_samples_path,
        input_prompts_path=args.input_prompts_path,
        internal=args.internal,
        with_reconstruction=args.with_reconstruction,
        swap_path=args.swap_path,
        swap_before_t=args.swap_before_t,
        swap_type=args.swap_type,
        save_dir=args.save_dir,
    )
