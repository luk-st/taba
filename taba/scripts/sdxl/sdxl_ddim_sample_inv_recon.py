import json
import logging
import pickle
from datetime import timedelta
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, gather_object
from diffusers.utils.logging import disable_progress_bar, enable_progress_bar
from diffusers.utils.torch_utils import randn_tensor
from fire import Fire
from tqdm import tqdm

from taba.ddim.schedulers import AdvancedDDIMInverseScheduler, AdvancedDDIMScheduler
from taba.models.sdxl.sdxl_pipeline import CustomStableDiffusionXLImg2ImgPipeline

torch.backends.cuda.matmul.allow_tf32 = True
torch._inductor.config.conv_1x1_as_mm = True  # type: ignore
torch._inductor.config.coordinate_descent_tuning = True  # type: ignore
torch._inductor.config.epilogue_fusion = False  # type: ignore
torch._inductor.config.coordinate_descent_check_all_directions = True  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 1.0
BATCH_SIZE = 4
PATH_DIR = "experiments/sdxl/ddim"
N_PROMPTS = 512
N_NOISES_PER_PROMPT = 4
COND_SEED = 11
SEED = 88
WITH_FORWARD = False
FORWARD_BEFORE_T = 2
FORWARD_SEED = 113


def create_noises(n_noises, generator=None, device="cpu"):
    img_size = (1024, 1024)
    VQAE_SCALE, N_CHANNELS = 8, 4
    latents_size = (n_noises, N_CHANNELS, img_size[0] // VQAE_SCALE, img_size[1] // VQAE_SCALE)
    return randn_tensor(latents_size, dtype=torch.float16, device=torch.device(device), generator=generator)


def load_prompts(n_prompts: int, n_noises_per_prompt: int, cond_seed: int):
    from datasets import load_dataset

    ds = load_dataset("UCSC-VLAA/Recap-COCO-30K", columns=["recaption"])
    ds_prompts = pd.Series(ds.data.get("train")[0]).sample(n_prompts, random_state=cond_seed).to_list()
    prompts = [ds_prompt for ds_prompt in ds_prompts for _ in range(n_noises_per_prompt)]
    return prompts


def get_model(device):
    disable_progress_bar()
    pipe_custom_sdxl = CustomStableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        safety_checker=None,
    )
    enable_progress_bar()
    pipe_custom_sdxl = pipe_custom_sdxl.to(device)
    return pipe_custom_sdxl


def save_tensor(
    accelerator: Accelerator,
    tensors: Dict[str, torch.Tensor],
    with_forward: bool,
    forward_before_t: int | None = None,
    forward_seed: int | None = None,
):
    path_dir = PATH_DIR
    if with_forward:
        path_dir += f"_forward{forward_before_t}_s{forward_seed}"
    Path(path_dir).mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        for name, tensor in tensors.items():
            torch.save(tensor.cpu(), (Path(path_dir) / f"{name}.pt").resolve())
    accelerator.wait_for_everyone()


def save_config(
    accelerator: Accelerator,
    config: Dict,
    with_forward: bool,
    forward_before_t: int | None = None,
    forward_seed: int | None = None,
):
    path_dir = PATH_DIR
    if with_forward:
        path_dir += f"_forward{forward_before_t}_s{forward_seed}"
    Path(path_dir).mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        with open(Path(path_dir) / "config.json", "w") as f:
            json.dump(config, f)


def sample_noise_images(
    pipeline,
    latents_or_images,
    prompts,
    accelerator,
    is_sampling: bool = True,
    forward_before_t: int | None = None,
    forward_seed: int | None = None,
    with_forward: bool = WITH_FORWARD,
    batch_size: int = BATCH_SIZE,
    num_inference_steps: int = NUM_INFERENCE_STEPS,
    guidance_scale: int = GUIDANCE_SCALE,
):
    images_ids = list(range(len(latents_or_images)))
    all_outputs = []
    with accelerator.split_between_processes(images_ids) as device_indices:
        images_device = torch.stack([latents_or_images[i] for i in device_indices])
        prompts_device = [prompts[i] for i in device_indices]
        fixed_noise_generator = (
            torch.Generator(device=accelerator.device).manual_seed(forward_seed + accelerator.process_index)
            if with_forward is True
            else None
        )

        all_outputs_device = []
        with tqdm(total=len(prompts_device)) as pbar:
            for batch_start in range(0, len(prompts_device), batch_size):
                imgs = images_device[batch_start : batch_start + batch_size].to(accelerator.device)
                prompts_batch = prompts_device[batch_start : batch_start + batch_size]

                if is_sampling:
                    outputs_batch = pipeline.denoise(
                        prompt=prompts_batch,
                        image=imgs,
                        latents=imgs,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        strength=1.0,
                        output_type="pt",
                    ).images.cpu()
                else:
                    outputs_batch = pipeline.invert(
                        prompt=prompts_batch,
                        image=imgs,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        strength=1.0,
                        output_type="pt",
                        forward_before_t=forward_before_t,
                        fixed_noise_generator=fixed_noise_generator,
                        is_first_batch=(batch_start == 0),
                    ).images.cpu()
                all_outputs_device.append(outputs_batch)
                pbar.update(len(prompts_batch))

            all_outputs_device = torch.cat(all_outputs_device, dim=0).cpu()
    torch.cuda.empty_cache()
    all_outputs.append(all_outputs_device)
    accelerator.wait_for_everyone()
    all_outputs = gather_object(all_outputs)
    all_outputs = torch.cat(all_outputs, dim=0)
    return all_outputs


def main(
    n_prompts: int = N_PROMPTS,
    n_noises_per_prompt: int = N_NOISES_PER_PROMPT,
    cond_seed: int = COND_SEED,
    seed: int = SEED,
    with_forward: bool = WITH_FORWARD,
    forward_before_t: int | None = FORWARD_BEFORE_T,
    forward_seed: int | None = FORWARD_SEED,
    batch_size: int = BATCH_SIZE,
    num_inference_steps: int = NUM_INFERENCE_STEPS,
    guidance_scale: int = GUIDANCE_SCALE,
):
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800 * 6))
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    pipe_custom_sdxl = get_model(accelerator.device)

    if with_forward is False:
        forward_before_t = None
        forward_seed = None

    prompts = load_prompts(n_prompts=n_prompts, n_noises_per_prompt=n_noises_per_prompt, cond_seed=cond_seed)
    generator = torch.manual_seed(seed)
    noise = create_noises(n_noises=len(prompts), generator=generator, device=accelerator.device)
    save_tensor(
        accelerator,
        {"noise": noise},
        with_forward=with_forward,
        forward_before_t=forward_before_t,
        forward_seed=forward_seed,
    )

    pipe_custom_sdxl.scheduler = AdvancedDDIMScheduler.from_config(pipe_custom_sdxl.scheduler.config)
    samples = sample_noise_images(
        pipe_custom_sdxl,
        noise,
        prompts,
        accelerator,
        is_sampling=True,
        with_forward=with_forward,
        forward_before_t=forward_before_t,
        forward_seed=forward_seed,
        batch_size=batch_size,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
    save_tensor(
        accelerator,
        {"samples": samples},
        with_forward=with_forward,
        forward_before_t=forward_before_t,
        forward_seed=forward_seed,
    )

    pipe_custom_sdxl.scheduler = AdvancedDDIMInverseScheduler.from_config(pipe_custom_sdxl.scheduler.config)
    latents = sample_noise_images(
        pipe_custom_sdxl,
        samples,
        prompts,
        accelerator,
        is_sampling=False,
        forward_before_t=forward_before_t,
        forward_seed=forward_seed,
        batch_size=batch_size,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        with_forward=with_forward,
    )
    save_tensor(
        accelerator,
        {"latents": latents},
        with_forward=with_forward,
        forward_before_t=forward_before_t,
        forward_seed=forward_seed,
    )

    pipe_custom_sdxl.scheduler = AdvancedDDIMScheduler.from_config(pipe_custom_sdxl.scheduler.config)
    recons = sample_noise_images(
        pipe_custom_sdxl,
        latents,
        prompts,
        accelerator,
        is_sampling=True,
        with_forward=with_forward,
        forward_before_t=forward_before_t,
        forward_seed=forward_seed,
        batch_size=batch_size,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
    save_tensor(
        accelerator,
        {"recons": recons},
        with_forward=with_forward,
        forward_before_t=forward_before_t,
        forward_seed=forward_seed,
    )
    save_config(
        accelerator,
        {
            "n_prompts": n_prompts,
            "n_noises_per_prompt": n_noises_per_prompt,
            "cond_seed": cond_seed,
            "seed": seed,
            "with_forward": with_forward,
            "forward_before_t": forward_before_t,
            "forward_seed": forward_seed,
            "batch_size": batch_size,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "PATH_DIR": PATH_DIR,
        },
        with_forward=with_forward,
        forward_before_t=forward_before_t,
        forward_seed=forward_seed,
    )


if __name__ == "__main__":
    Fire(main)
