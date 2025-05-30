import argparse
import logging
import os
import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, gather_object
from diffusers.training_utils import set_seed

from taba.models.adm.ddim import generate_samples as adm_generate_samples
from taba.models.adm.models import get_ddpm_imagenet256, get_openai_cifar, get_openai_imagenet
from taba.models.ldms.models import get_ldm_celeba, get_scheduler
from taba.models.ldms.sample_ldm import decode_image
from taba.models.ldms.sample_ldm import generate_samples as ldm_generate_samples

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
ADD_NOISE_SEED_DEFAULT = 10
N_VARIATIONS_DEFAULT = 20

MODELS_MAP = {
    "cifar_pixel_32": "cifar_pixel_32",
    "imagenet_pixel_64": "imagenet_pixel_64",
    "imagenet_pixel_256": "imagenet_pixel_256",
    "celeba_ldm_256": "celeba_ldm_256",
    "imagenet_dit_256": "imagenet_dit_256",
}

ADM_INIT_FUNCTIONS = {
    "cifar_pixel_32": get_openai_cifar,
    "imagenet_pixel_64": get_openai_imagenet,
    "imagenet_pixel_256": get_ddpm_imagenet256,
}


def prepare_noise(n_all_prompts: int, single_noise_shape: tuple[int, int, int], seed: int):
    noise_shape = (n_all_prompts, *single_noise_shape)
    noises = torch.randn(noise_shape, generator=torch.manual_seed(seed))
    return noises


def interpolate_noise(noise: torch.Tensor, alpha: float, added_noise_seed: int, n_variations: int):
    generator = torch.Generator(device=noise.device).manual_seed(added_noise_seed)
    added_noise = (
        torch.normal(
            mean=0,
            std=np.sqrt(1),
            size=(n_variations, noise.shape[1], noise.shape[2], noise.shape[3]),
            generator=generator,
        )
        .unsqueeze(0)
        .repeat(noise.shape[0], 1, 1, 1, 1)
    )
    starting_noise = noise.unsqueeze(1).repeat(1, n_variations, 1, 1, 1)
    final_noise = torch.sqrt(torch.tensor(1 - alpha)) * starting_noise + torch.sqrt(torch.tensor(alpha)) * added_noise
    final_noise = final_noise.view(-1, starting_noise.shape[2], starting_noise.shape[3], starting_noise.shape[4])
    return final_noise


def save_tensor(accelerator: Accelerator, tensors: Dict[str, torch.Tensor], path_dir: str):
    if accelerator.is_main_process:
        for name, tensor in tensors.items():
            torch.save(tensor.cpu(), (Path(path_dir) / f"{name}.pt").resolve())
    accelerator.wait_for_everyone()


def sample_adm(input_noise: torch.Tensor, model_name: str, T: int, device: torch.device, batch_size: int):
    model, pipeline, args = ADM_INIT_FUNCTIONS[model_name](steps=T, device=device)

    n_samples = input_noise.shape[0]
    outs = adm_generate_samples(
        random_noises=input_noise,
        number_of_samples=n_samples,
        batch_size=batch_size,
        diffusion_pipeline=pipeline,
        ddim_model=model,
        diffusion_args=args,
        device=device,
        from_each_t=False,
    )
    return {"samples": outs}


def sample_celeba(input_noise: torch.Tensor, model_name: str, T: int, device: torch.device, batch_size: int):
    ldm_unet, vae = get_ldm_celeba(device=device)
    scheduler = get_scheduler(T=T)

    samples = ldm_generate_samples(
        noise=input_noise,
        diffusion_unet=ldm_unet,
        diffusion_scheduler=scheduler,
        batch_size=batch_size,
        device=device,
        from_each_t=False,
    )
    samples_decoded = decode_image(
        unet_out=samples,
        vqvae=vae,
        batch_size=batch_size,
        device=device,
    )

    return {"samples": samples, "samples_decoded": samples_decoded}


def sample_multigpu(input_noise: torch.Tensor, model_name: str, T: int, batch_size: int, accelerator: Accelerator):
    all_noise = list(range(input_noise.shape[0]))
    all_outputs = defaultdict(list)
    with accelerator.split_between_processes(all_noise) as device_indices:
        noise_device = torch.stack([input_noise[i] for i in device_indices])
        if accelerator.is_main_process:
            print(f"Rank {accelerator.process_index} | {(len(device_indices)/len(all_noise)):.2f}")
        if "pixel" in model_name:
            outputs = sample_adm(
                input_noise=noise_device,
                model_name=model_name,
                T=T,
                device=accelerator.device,
                batch_size=batch_size,
            )
        elif model_name == "celeba_ldm_256":
            outputs = sample_celeba(
                input_noise=noise_device,
                model_name=model_name,
                T=T,
                device=accelerator.device,
                batch_size=batch_size,
            )
        for key, value in outputs.items():
            all_outputs[key].append(value)
    accelerator.wait_for_everyone()
    for obj_name, obj_value in all_outputs.items():
        vals = gather_object(obj_value)
        all_outputs[obj_name] = torch.cat(vals, dim=0)
    return all_outputs


def main(
    model_name: str,
    T: int,
    input_path: str,
    input_type: str,
    var: float,
    seed: int,
    add_noise_seed: int,
    n_variations: int,
    batch_size: int,
    start_time: str = START_TIME,
):
    SAVE_DIR = (
        f"experiments/diversity/{model_name}/noise_adding/"
        f"{start_time}_"
        f"seed_{seed}_"
        f"add_noise_seed_{add_noise_seed}_"
        f"var_{var}_"
        f"T_{T}_"
        f"input_type_{input_type}_"
        f"batch_size_{batch_size}_"
        f"n_variations_{n_variations}"
    )

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800 * 6))
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    if accelerator.is_main_process:
        os.makedirs(SAVE_DIR, exist_ok=True)
        logging.info(f"Seed: {seed}")
        logging.info(f"Device: {accelerator.device}")
        logging.info(f"Save dir: {SAVE_DIR}")
    set_seed(seed)

    assert os.path.exists(input_path), f"Input path {input_path} does not exist"
    start_noise = torch.load(input_path, map_location="cpu", weights_only=False)
    interpolated_noise = interpolate_noise(
        noise=start_noise, alpha=var, added_noise_seed=add_noise_seed, n_variations=n_variations
    )
    if accelerator.is_main_process:
        logging.info(f"New noise shape: {interpolated_noise.shape}")
    save_tensor(accelerator=accelerator, tensors={"interpolated_noise": interpolated_noise}, path_dir=SAVE_DIR)

    # sampling noise -> image
    if accelerator.is_main_process:
        logging.info("Sampling noise -> image")
    outputs = sample_multigpu(
        input_noise=interpolated_noise,
        model_name=model_name,
        T=T,
        batch_size=batch_size,
        accelerator=accelerator,
    )
    save_tensor(accelerator=accelerator, tensors=outputs, path_dir=SAVE_DIR)

    # calculate metrics
    if accelerator.is_main_process:
        logging.info("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, choices=list(MODELS_MAP.keys()))
    parser.add_argument("--T", type=int, default=NUM_INFERENCE_STEPS_DEFAULT)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--input_type", type=str, choices=["noise", "latent"], required=True)
    parser.add_argument("--var", type=float, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--add_noise_seed", type=int, default=ADD_NOISE_SEED_DEFAULT)
    parser.add_argument("--n_variations", type=int, default=N_VARIATIONS_DEFAULT)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE_DEFAULT)
    args = parser.parse_args()
    main(
        model_name=args.model_name,
        T=args.T,
        input_path=args.input_path,
        input_type=args.input_type,
        var=args.var,
        seed=args.seed,
        add_noise_seed=args.add_noise_seed,
        n_variations=args.n_variations,
        batch_size=args.batch_size,
    )
