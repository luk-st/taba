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

from taba.metrics.angles_distances import reconstruction_error
from taba.metrics.correlation import get_top_k_corr_in_patches
from taba.metrics.normality import kl_div, stats_from_tensor
from taba.models.dit.constants import DIT_IMAGENET_MAP
from taba.models.dit.dit import CustomDiTPipeline
from diffusers.utils.logging import disable_progress_bar, enable_progress_bar


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


def prepare_noise(n_all_prompts: int, dit_pipeline: CustomDiTPipeline, seed: int):
    noises = dit_pipeline.prepare_latents(n_samples=n_all_prompts, generator=torch.manual_seed(seed))
    return noises


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
):
    conds_indices = list(range(len(conds)))
    all_outputs = defaultdict(list)
    with accelerator.split_between_processes(conds_indices) as device_indices:
        conds_device = torch.stack([conds[i] for i in device_indices])
        latent_device = torch.stack([latent[i] for i in device_indices])
        if accelerator.is_main_process:
            print(f"Rank {accelerator.process_index} | {(len(device_indices)/len(conds_indices)):.2f}")
        if run_inversion:
            outputs = pipeline.ddim_inverse(
                latents_x_0=latent_device,
                guidance_scale=guidance_scale,
                batch_size=batch_size,
                num_inference_steps=num_inference_steps,
                class_labels=conds_device,
                from_each_t=from_each_t,
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
    seed: int,
    noises_per_prompt: int,
    n_prompts: int,
    batch_size: int,
    num_inference_steps: int,
    guidance_scale: float,
    input_path: str | None,
    input_cond_path: str | None,
    cond_seed: int,
    internal: bool,
    with_inversion: bool,
    with_reconstruction: bool,
    start_time: str = START_TIME,
    save_dir: str | None = None,
):
    INTERNAL_DIR = "internal/" if internal else ""
    if save_dir is None:
        save_dir = (
            f"experiments/dit/{INTERNAL_DIR}sampling_invert_reconstruction/"
            f"{start_time}_"
            f"seed_{seed}_"
            f"noises_per_prompt_{noises_per_prompt}_"
            f"n_prompts_{n_prompts}_"
            f"batch_size_{batch_size}_"
            f"num_inference_steps_{num_inference_steps}_"
            f"guidance_scale_{guidance_scale}_"
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
        logging.info(f"Internal: {internal}")
        logging.info(f"Input cond path: {input_cond_path}")
    set_seed(seed)

    disable_progress_bar()
    pipe: CustomDiTPipeline = CustomDiTPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.float16)
    enable_progress_bar()
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to(accelerator.device)

    if input_cond_path is not None:
        assert os.path.exists(input_cond_path), f"Input cond path {input_cond_path} does not exist"
        conds = torch.load(input_cond_path, weights_only=False)
    else:
        conds = load_cond(n_prompts, noises_per_prompt, pipe, cond_seed)
    save_tensor(accelerator=accelerator, tensors={"conds": conds}, path_dir=save_dir)

    if input_path is not None:
        assert os.path.exists(input_path), f"Input path {input_path} does not exist"
        noise = torch.load(input_path, map_location="cpu", weights_only=False)
        assert noise.shape[0] == len(conds), f"Noise shape {noise.shape} does not match number of prompts {len(conds)}"
    else:
        noise = prepare_noise(len(conds), pipe, seed)
    if accelerator.is_main_process:
        logging.info(f"Number of prompts: {len(conds)}")
        logging.info(f"Noise shape: {noise.shape}")
    save_tensor(accelerator=accelerator, tensors={"noise": noise}, path_dir=save_dir)

    # sampling noise -> image
    if accelerator.is_main_process:
        logging.info("Sampling noise -> image")
    denoising_outputs = denoise_invert_multigpu(
        latent=noise,
        conds=conds,
        pipeline=pipe,
        accelerator=accelerator,
        batch_size=batch_size,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        from_each_t=internal,
        run_inversion=False,
    )
    save_tensor(accelerator=accelerator, tensors=denoising_outputs, path_dir=save_dir)

    decoded_samples = None
    if accelerator.is_main_process:
        logging.info("Decoding samples")
        decoded_samples = pipe.vae_decode(latents=denoising_outputs["samples"], batch_size=batch_size)
    accelerator.wait_for_everyone()
    save_tensor(accelerator=accelerator, tensors={"decoded_samples": decoded_samples}, path_dir=save_dir)

    if with_inversion:
        # inversion image -> latent
        if accelerator.is_main_process:
            logging.info("Inversion image -> latent")
        inversion_outputs = denoise_invert_multigpu(
            latent=denoising_outputs["samples"],
            conds=conds,
            pipeline=pipe,
            accelerator=accelerator,
            batch_size=batch_size,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            from_each_t=internal,
            run_inversion=True,
        )
        save_tensor(accelerator=accelerator, tensors=inversion_outputs, path_dir=save_dir)

    if with_reconstruction:
        # reconstruction latent -> image2
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

        metrics = {}
        metrics["_params"] = {}
        metrics["_params"]["seed"] = seed
        metrics["_params"]["noises_per_prompt"] = noises_per_prompt
        metrics["_params"]["n_prompts"] = n_prompts
        metrics["_params"]["batch_size"] = batch_size
        metrics["_params"]["num_inference_steps"] = num_inference_steps
        metrics["_params"]["guidance_scale"] = guidance_scale
        metrics["_params"]["cond_seed"] = cond_seed
        metrics["_params"]["input_path"] = input_path
        metrics["_params"]["with_inversion"] = with_inversion
        metrics["_params"]["with_reconstruction"] = with_reconstruction
        metrics["_params"]["internal"] = internal
        metrics["_params"]["save_dir"] = save_dir

        metrics["metrics"] = {}

        if with_inversion:
            noise = noise.cpu()
            latents = inversion_outputs["latents"].cpu()
            metrics["metrics"]["kl_div_noise_latent"] = kl_div(noise, latents)
            metrics["metrics"]["reconstruction_error_noise"] = reconstruction_error(noise, latents)
            metrics["metrics"]["correlation"] = get_top_k_corr_in_patches(latents, top_k=20)
            metrics["metrics"]["latent_stats"] = stats_from_tensor(latents)
            metrics["metrics"]["noise_stats"] = stats_from_tensor(noise)

        if with_reconstruction:
            images = decoded_samples.cpu()
            images2 = decoded_samples2.cpu()
            metrics["metrics"]["reconstruction_error_images"] = reconstruction_error(images, images2)

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
    parser.add_argument("--input_cond_path", type=str, default=None)
    parser.add_argument("--cond_seed", type=int, default=COND_SEED_DEFAULT)
    parser.add_argument("--internal", action="store_true")
    parser.add_argument("--with_inversion", action="store_true")
    parser.add_argument("--with_reconstruction", action="store_true")
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
        input_cond_path=args.input_cond_path,
        cond_seed=args.cond_seed,
        internal=args.internal,
        with_inversion=args.with_inversion,
        with_reconstruction=args.with_reconstruction,
        save_dir=args.save_dir,
    )
