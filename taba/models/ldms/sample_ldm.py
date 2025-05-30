from typing import List

import numpy as np
import PIL.Image
import torch
from accelerate import Accelerator
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.autoencoders.vq_model import VQModel
from diffusers.models.unets.unet_2d import UNet2DModel
from einops import rearrange
from tqdm import tqdm

from taba.ddim.schedulers import (
    AdvancedDDIMInverseScheduler,
    AdvancedDDIMScheduler,
    AdvancedDDIMSchedulerOutput,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 64
DEFAULT_SEED = 420


def _rearrange_samples_t_to_b(tensors: List[torch.Tensor]) -> torch.Tensor:
    t = torch.stack(tensors).cpu()
    t = rearrange(t, "t b c h w -> b t c h w")
    return t


def _rearrange_samples_b_to_t(tensors: List[torch.Tensor]) -> torch.Tensor:
    t = torch.cat(tensors, dim=0).cpu()
    t = rearrange(t, "b t c h w -> t b c h w")
    return t


def get_noises(n_samples, seed=DEFAULT_SEED):
    generator = torch.manual_seed(seed)
    noise = torch.randn(
        (n_samples, 3, 64, 64),
        generator=generator,
    )
    return noise


def generate_samples(
    noise: torch.Tensor,
    diffusion_unet: UNet2DModel,
    diffusion_scheduler: AdvancedDDIMScheduler,
    batch_size: int = BATCH_SIZE,
    device: torch.device = DEVICE,
    from_each_t: bool = False,
    accelerator: Accelerator | None = None,
):
    n = noise.shape[0]
    all_samples = []
    all_t_samples = []
    all_t_eps = []
    # all_t_pred_xstart = []

    if accelerator is not None:
        if accelerator.is_main_process:
            pbar = tqdm(range(0, n, batch_size), desc="DDIM Sampling")
        else:
            pbar = range(0, n, batch_size)
    else:
        pbar = tqdm(range(0, n, batch_size), desc="DDIM Sampling")

    for idx_start in pbar:
        samples = noise[idx_start : idx_start + batch_size].to(device)
        t_samples = []
        t_eps = []
        # t_pred_xstart = []

        timesteps = diffusion_scheduler.timesteps
        timestep_ids = list(range(len(timesteps)))[::-1]
        for t_idx, t in zip(timestep_ids, timesteps):
            with torch.no_grad():
                residual = diffusion_unet(samples, t)["sample"]
            output: AdvancedDDIMSchedulerOutput = diffusion_scheduler.step(
                model_output=residual, timestep=t, sample=samples, timestep_idx=t_idx
            )
            samples = output.prev_sample
            if from_each_t:
                t_samples.append(samples.clone())
                t_eps.append(output.eps.clone())
                # t_pred_xstart.append(output.pred_original_sample.clone())
        all_samples.append(samples.cpu())
        if from_each_t:
            all_t_samples.append(_rearrange_samples_t_to_b(t_samples))
            all_t_eps.append(_rearrange_samples_t_to_b(t_eps))
            # all_t_pred_xstart.append(_rearrange_samples_t_to_b(t_pred_xstart))

    all_samples = torch.cat(all_samples)
    if from_each_t:
        all_t_samples = _rearrange_samples_b_to_t(all_t_samples)
        all_t_eps = _rearrange_samples_b_to_t(all_t_eps)
        # all_t_pred_xstart = _rearrange_samples_b_to_t(all_t_pred_xstart)
        return {
            "samples": all_samples,
            "all_t_samples": all_t_samples,
            "all_t_eps_samples": all_t_eps,
            # "all_t_pred_xstart_samples": all_t_pred_xstart,
        }
    else:
        return {"samples": all_samples}


def generate_latents(
    samples: torch.Tensor,
    diffusion_unet: UNet2DModel,
    diffusion_scheduler: AdvancedDDIMInverseScheduler,
    batch_size: int,
    device: torch.device,
    from_each_t: bool = False,
    swap_eps: dict[int, torch.Tensor] = {},
    swap_xt: dict[int, torch.Tensor] = {},
    accelerator: Accelerator | None = None,
):
    assert swap_eps == {} or swap_xt == {}, "swap_eps and swap_xt cannot both be provided"

    n = samples.shape[0]
    all_latents = []
    all_t_latents = []
    all_t_eps = []
    # all_t_pred_xstart = []

    if accelerator is not None:
        if accelerator.is_main_process:
            pbar = tqdm(range(0, n, batch_size), desc="DDIM Inversion")
        else:
            pbar = range(0, n, batch_size)
    else:
        pbar = tqdm(range(0, n, batch_size), desc="DDIM Inversion")

    for idx_start in pbar:
        latents = samples[idx_start : idx_start + batch_size].to(device)
        sw_eps_curr = {k: swap_eps[k][idx_start : idx_start + batch_size].to(device) for k in swap_eps.keys()}
        sw_xt_curr = {k: swap_xt[k][idx_start : idx_start + batch_size].to(device) for k in swap_xt.keys()}
        t_latents = []
        t_eps = []
        # t_pred_xstart = []
        timesteps = diffusion_scheduler.timesteps
        timestep_ids = list(range(len(timesteps)))

        for t_idx, t in zip(timestep_ids, timesteps):
            with torch.no_grad():
                residual = diffusion_unet(latents, t)["sample"]
            output: AdvancedDDIMSchedulerOutput = diffusion_scheduler.step(
                model_output=residual, timestep=t, sample=latents, timestep_idx=t_idx, swap_eps=sw_eps_curr
            )
            if t_idx in sw_xt_curr:
                output.prev_sample = sw_xt_curr[t_idx]
            latents = output.prev_sample
            if from_each_t:
                t_latents.append(latents.clone())
                t_eps.append(output.eps.clone())
                # t_pred_xstart.append(output.pred_original_sample.clone())
        all_latents.append(latents.cpu())
        if from_each_t:
            all_t_latents.append(_rearrange_samples_t_to_b(t_latents))
            all_t_eps.append(_rearrange_samples_t_to_b(t_eps))
            # all_t_pred_xstart.append(_rearrange_samples_t_to_b(t_pred_xstart))

    all_latents = torch.cat(all_latents)
    if from_each_t:
        all_t_latents = _rearrange_samples_b_to_t(all_t_latents)
        all_t_eps = _rearrange_samples_b_to_t(all_t_eps)
        # all_t_pred_xstart = _rearrange_samples_b_to_t(all_t_pred_xstart)
        return {
            "latents": all_latents,
            "all_t_latents": all_t_latents,
            "all_t_eps_latents": all_t_eps,
            # "all_t_pred_xstart_latents": all_t_pred_xstart,
        }
    else:
        return {"latents": all_latents}


def decode_image(
    unet_out: torch.Tensor,
    vqvae: VQModel,
    batch_size: int,
    device: torch.device,
    accelerator: Accelerator | None = None,
):
    n = unet_out.shape[0]
    outs = []
    if accelerator is not None:
        if accelerator.is_main_process:
            pbar = tqdm(range(0, n, batch_size), desc="VAE Decoding")
        else:
            pbar = range(0, n, batch_size)
    else:
        pbar = tqdm(range(0, n, batch_size), desc="VAE Decoding")

    for idx_start in pbar:
        u_outs = unet_out[idx_start : idx_start + batch_size].to(device)
        with torch.no_grad():
            decoded: DecoderOutput = vqvae.decode(u_outs, return_dict=True)  # type: ignore
            outs.append(decoded.sample.cpu())
    return torch.cat(outs)


def to_image(vqvae_out):
    image_processed = vqvae_out.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.clamp(0, 255).numpy().astype(np.uint8)
    return [PIL.Image.fromarray(img_processed) for img_processed in image_processed]


def from_image(image: PIL.Image.Image):
    image = np.array(image)
    image = image.astype(np.float32) / 127.5 - 1.0
    image = torch.from_numpy(image)
    image = image.permute(2, 0, 1)
    return image
