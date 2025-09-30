import argparse
import json
import logging
import os
import pickle
import sys
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
NOISES_PER_PROMPT_DEFAULT = 4
N_PROMPTS_DEFAULT = 128
BATCH_SIZE_DEFAULT = 16
NUM_INFERENCE_STEPS_DEFAULT = 50
NULL_PROMPT = ""
HOBBIT_PROMPT = "hobbit standing next to its house"
GUIDANCE_SCALE_DEFAULT = 1.0
COND_SEED_DEFAULT = 10


def _rearrange_t_to_b(tensor: torch.Tensor) -> torch.Tensor:
    return rearrange(tensor, "t b c h w -> b t c h w")


def _rearrange_b_to_t(tensor: torch.Tensor) -> torch.Tensor:
    return rearrange(tensor, "b t c h w -> t b c h w")


def load_prompts(n_prompts: int, n_noises_per_prompt: int, prompts_dataset: str, cond_seed: int):
    if prompts_dataset == "null":
        prompts = [NULL_PROMPT] * n_prompts * n_noises_per_prompt
    elif prompts_dataset == "hobbit":
        prompts = [HOBBIT_PROMPT] * n_prompts * n_noises_per_prompt
    elif prompts_dataset == "dataset":
        from datasets import load_dataset

        ds = load_dataset("UCSC-VLAA/Recap-COCO-30K", columns=["recaption"])
        ds_prompts = pd.Series(ds.data.get("train")[0]).sample(n_prompts, random_state=cond_seed).to_list()
        prompts = [ds_prompt for ds_prompt in ds_prompts for _ in range(n_noises_per_prompt)]
    else:
        raise ValueError(f"Invalid prompts dataset: {prompts_dataset}")
    return prompts


def save_prompts_pkl(prompts: list[str], path_dir: str):
    with open(Path(path_dir) / "prompts.pkl", "wb") as f:
        pickle.dump(prompts, f)


def load_prompts_pkl(path_dir: str):
    with open(Path(path_dir) / "prompts.pkl", "rb") as f:
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
):
    all_outputs = defaultdict(list)
    with tqdm(total=len(prompts)) as pbar:
        for batch_num, batch_start in enumerate(range(0, len(prompts), batch_size)):
            prompt = prompts[batch_start : batch_start + batch_size]
            latent = noise[batch_start : batch_start + batch_size].to(device)

            if run_inversion:
                outputs = pipeline.invert(
                    prompt=prompt,
                    image=latent,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    eta=0.0,
                    clean_caption=False,
                    from_each_t=from_each_t,
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
):
    prompts_indices = list(range(len(prompts)))
    all_outputs = defaultdict(list)
    with accelerator.split_between_processes(prompts_indices) as device_indices:
        prompts_device = [prompts[i] for i in device_indices]
        latent_device = torch.stack([latent[i] for i in device_indices])
        if accelerator.is_main_process:
            print(f"Rank {accelerator.process_index} | {(len(device_indices)/len(prompts_indices)):.2f}")
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
    noises_per_prompt: int,
    n_prompts: int,
    batch_size: int,
    num_inference_steps: int,
    guidance_scale: float,
    input_path: str | None,
    cond_seed: int,
    internal: bool,
    with_inversion: bool,
    with_reconstruction: bool,
    prompts_dataset: str,
    start_time: str = START_TIME,
    save_dir: str | None = None,
):
    cmd = " ".join(sys.argv)
    INTERNAL_DIR = "internal/" if internal else ""
    if save_dir is None:
        save_dir = (
            f"experiments/deepfloyd_if/{INTERNAL_DIR}sampling_invert_reconstruction/"
            f"{prompts_dataset}_"
            f"{start_time}_"
            f"seed_{seed}_"
            f"noises_per_prompt_{noises_per_prompt}_"
            f"n_prompts_{n_prompts}_"
            f"batch_size_{batch_size}_"
            f"num_inference_steps_{num_inference_steps}_"
            f"guidance_scale_{guidance_scale}"
            f"cond_seed_{cond_seed}"
            f"_with_inversion_{with_inversion}"
            f"_with_reconstruction_{with_reconstruction}"
        )
        save_dir = save_dir + "_with_input" if input_path is not None else save_dir

    if with_reconstruction and not with_inversion:
        raise ValueError("Reconstruction requires inversion")

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800 * 6))
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
        logging.info(f"Seed: {seed}")
        logging.info(f"Device: {accelerator.device}")
        logging.info(f"Num inference steps: {num_inference_steps}")
        logging.info(f"Batch size: {batch_size}")
        logging.info(f"Save dir: {save_dir}")
        logging.info(f"Input path: {input_path}")
        logging.info(f"With inversion: {with_inversion}")
        logging.info(f"With reconstruction: {with_reconstruction}")
        logging.info(f"Prompts dataset: {prompts_dataset}")
        logging.info(f"Internal: {internal}")
        logging.info(f"Cond seed: {cond_seed}")
        logging.info(f"Start time: {start_time}")
        logging.info(f"Command: {cmd}")
    accelerator.wait_for_everyone()
    set_seed(seed)

    disable_progress_bar()
    pipe = initialize_if_ddim_pipeline()
    enable_progress_bar()
    pipe.set_progress_bar_config(disable=True)

    prompts = load_prompts(n_prompts, noises_per_prompt, prompts_dataset, cond_seed)
    save_prompts_pkl(prompts, save_dir)

    if input_path is not None:
        assert os.path.exists(input_path), f"Input path {input_path} does not exist"
        noise = torch.load(input_path, map_location="cpu", weights_only=False)
        assert noise.shape[0] == len(
            prompts
        ), f"Noise shape {noise.shape} does not match number of prompts {len(prompts)}"
    else:
        noise = prepare_noise(len(prompts), pipe, seed)
    if accelerator.is_main_process:
        logging.info(f"Number of prompts: {len(prompts)}")
        logging.info(f"Noise shape: {noise.shape}")
    save_tensor(accelerator=accelerator, tensors={"noise": noise}, path_dir=save_dir)

    # sampling noise -> image
    if accelerator.is_main_process:
        logging.info("Sampling noise -> image")
    denoising_outputs = denoise_invert_multigpu(
        latent=noise,
        prompts=prompts,
        pipeline=pipe,
        accelerator=accelerator,
        batch_size=batch_size,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        from_each_t=internal,
        run_inversion=False,
    )
    save_tensor(accelerator=accelerator, tensors=denoising_outputs, path_dir=save_dir)

    if not with_inversion:
        if accelerator.is_main_process:
            logging.info("Done")
        return
    if accelerator.is_main_process:
        logging.info("Inversion image -> latent")
    inversion_outputs = denoise_invert_multigpu(
        latent=denoising_outputs["samples"],
        prompts=prompts,
        pipeline=pipe,
        accelerator=accelerator,
        batch_size=batch_size,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        from_each_t=internal,
        run_inversion=True,
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
        noise = noise.cpu()
        latents = inversion_outputs["latents"].cpu()
        images = denoising_outputs["samples"].cpu()
        images2 = images2["samples2"].cpu()

        logging.info("Calculating metrics")
        metrics = {}
        metrics["_params"] = {}
        metrics["_params"]["seed"] = seed
        metrics["_params"]["noises_per_prompt"] = noises_per_prompt
        metrics["_params"]["n_prompts"] = n_prompts
        metrics["_params"]["batch_size"] = batch_size
        metrics["_params"]["num_inference_steps"] = num_inference_steps
        metrics["_params"]["guidance_scale"] = guidance_scale
        metrics["_params"]["prompts_dataset"] = prompts_dataset
        metrics["_params"]["cond_seed"] = cond_seed
        metrics["_params"]["input_path"] = input_path
        metrics["_params"]["internal"] = internal
        metrics["_params"]["with_inversion"] = with_inversion
        metrics["_params"]["with_reconstruction"] = with_reconstruction
        metrics["_params"]["save_dir"] = save_dir
        metrics["_params"]["cmd"] = cmd
        metrics["metrics"] = {}
        metrics["metrics"]["kl_div_noise_latent"] = kl_div(noise, latents)
        metrics["metrics"]["reconstruction_error_images"] = reconstruction_error(images, images2)
        metrics["metrics"]["reconstruction_error_noise"] = reconstruction_error(noise, latents)
        metrics["metrics"]["correlation"] = get_top_k_corr_in_patches(latents, top_k=20)
        metrics["metrics"]["latent_stats"] = stats_from_tensor(latents)
        metrics["metrics"]["noise_stats"] = stats_from_tensor(noise)

        logging.info(metrics)
        with open(Path(save_dir) / "metrics.json", "w") as f:
            json.dump(metrics, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--noises_per_prompt", type=int, default=NOISES_PER_PROMPT_DEFAULT)
    parser.add_argument("--n_prompts", type=int, default=N_PROMPTS_DEFAULT)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument("--num_inference_steps", type=int, default=NUM_INFERENCE_STEPS_DEFAULT)
    parser.add_argument("--guidance_scale", type=float, default=GUIDANCE_SCALE_DEFAULT)
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--cond_seed", type=int, default=COND_SEED_DEFAULT)
    parser.add_argument("--internal", action="store_true")
    parser.add_argument("--with_inversion", action="store_true")
    parser.add_argument("--with_reconstruction", action="store_true")
    parser.add_argument("--prompts", type=str, choices=["null", "hobbit", "dataset"], default="dataset")
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()
    main(
        seed=args.seed,
        noises_per_prompt=args.noises_per_prompt,
        n_prompts=args.n_prompts,
        batch_size=args.batch_size,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        input_path=args.input_path,
        cond_seed=args.cond_seed,
        internal=args.internal,
        with_inversion=args.with_inversion,
        with_reconstruction=args.with_reconstruction,
        prompts_dataset=args.prompts,
        save_dir=args.save_dir,
    )
