import random

import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

IN_CHANNELS = 3


def generate_noises(number_of_samples, diffusion_args, seed=420, in_channels=IN_CHANNELS, device="cuda"):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    random_noise = torch.randn(
        (number_of_samples, in_channels, diffusion_args["image_size"], diffusion_args["image_size"])
    )
    random_noise = random_noise.to(device)
    return random_noise


def generate_samples(
    random_noises,
    number_of_samples,
    batch_size,
    diffusion_pipeline,
    ddim_model,
    diffusion_args,
    device="cuda",
    from_each_t=False,
):
    ddim_noise_samples = []
    all_t_samples = []
    all_t_eps_samples = []
    all_t_pred_xstart_samples = []

    for i in tqdm(range(0, number_of_samples // batch_size), desc="Generating samples from noises"):
        noise_in = random_noises[i * batch_size : (i + 1) * batch_size].to(device)
        outs = diffusion_pipeline.ddim_sample_loop(
            ddim_model,
            (batch_size, IN_CHANNELS, diffusion_args["image_size"], diffusion_args["image_size"]),
            clip_denoised=True,
            device=device,
            noise=noise_in,
            from_each_t=from_each_t,
        )

        if from_each_t:
            sample = outs["sample"]

            t_samples = outs["t_samples"]
            t_samples = torch.stack(t_samples).cpu()
            t_samples = rearrange(t_samples, "t b c h w -> b t c h w")
            all_t_samples.append(t_samples)

            eps_samples = outs["eps_samples"]
            eps_samples = torch.stack(eps_samples).cpu()
            eps_samples = rearrange(eps_samples, "t b c h w -> b t c h w")
            all_t_eps_samples.append(eps_samples)

            pred_xstart_samples = outs["pred_xstart_samples"]
            pred_xstart_samples = torch.stack(pred_xstart_samples).cpu()
            pred_xstart_samples = rearrange(pred_xstart_samples, "t b c h w -> b t c h w")
            all_t_pred_xstart_samples.append(pred_xstart_samples)
        else:
            sample = outs

        ddim_noise_samples.append(sample.cpu())

    ddim_samples = torch.cat(ddim_noise_samples)
    if from_each_t:
        all_t_samples = torch.cat(all_t_samples)
        all_t_samples = rearrange(all_t_samples, "b t c h w -> t b c h w")
        all_t_eps_samples = torch.cat(all_t_eps_samples)
        all_t_eps_samples = rearrange(all_t_eps_samples, "b t c h w -> t b c h w")
        all_t_pred_xstart_samples = torch.cat(all_t_pred_xstart_samples)
        all_t_pred_xstart_samples = rearrange(all_t_pred_xstart_samples, "b t c h w -> t b c h w")

        return {
            "samples": ddim_samples,
            "all_t_samples": all_t_samples,
            "all_t_eps_samples": all_t_eps_samples,
            "all_t_pred_xstart_samples": all_t_pred_xstart_samples,
        }
    else:
        return {"samples": ddim_samples}


def generate_latents(
    ddim_generations,
    batch_size,
    diffusion_pipeline,
    ddim_model,
    device="cuda",
    from_each_t=False,
    swap_eps={},
    swap_xt={},
    forward_before_t: int | None = None,
    fixed_noise_generator: torch.Generator | None = None,
):
    x = ddim_generations
    latents = []
    all_t_latents = []
    all_t_eps_samples = []
    all_t_pred_xstart_samples = []

    apply_forward = False
    assert (forward_before_t is None) or (0<= forward_before_t <= diffusion_pipeline.num_timesteps), "forward_before_t must be less than or equal to the number of timesteps when provided"
    if forward_before_t is not None and forward_before_t > 0:
        assert fixed_noise_generator is not None, "fixed_noise_generator must be provided if forward_before_t is provided"
        assert from_each_t is False, "from_each_t not supported if forward_before_t is provided"
        apply_forward=True

    for j in tqdm(range((x.shape[0] // batch_size)), desc="Generating latents from samples"):
        xj = x[j * batch_size : (j + 1) * batch_size]
        if len(swap_eps.keys()) > 0 or len(swap_xt.keys()) > 0:
            swap_eps_j = {}
            swap_xt_j = {}
            for key in swap_eps.keys():
                swap_eps_j[key] = swap_eps[key][j * batch_size : (j + 1) * batch_size]
            for key in swap_xt.keys():
                swap_xt_j[key] = swap_xt[key][j * batch_size : (j + 1) * batch_size]
        else:
            swap_eps_j = {}
            swap_xt_j = {}
        timesteps_t_latents = []
        timesteps_eps_samples = []
        timesteps_pred_xstart_samples = []

        if apply_forward:
            eps = torch.randn(size=xj.shape, generator=fixed_noise_generator, dtype=xj.dtype, device=xj.device)
            t_batch = torch.tensor([forward_before_t] * batch_size, device=xj.device)
            xj = diffusion_pipeline.q_sample(x_start=xj, t=t_batch, noise=eps)
        else:
            forward_before_t = 0

        for i in range(forward_before_t, diffusion_pipeline.num_timesteps):
            with torch.no_grad():
                xj = xj.to(device)
                t = torch.tensor([i] * xj.shape[0], device=device)
                sample = diffusion_pipeline.ddim_reverse_sample(
                    ddim_model,
                    xj,
                    t,
                    clip_denoised=True,
                    swap_eps=swap_eps_j,
                    swap_xt=swap_xt_j,
                )
                xj = sample["sample"]
                if from_each_t:
                    timesteps_t_latents.append(xj.clone().cpu())
                    timesteps_eps_samples.append(sample["eps"].clone().cpu())
                    timesteps_pred_xstart_samples.append(sample["pred_xstart"].clone().cpu())

        if from_each_t:
            timesteps_t_latents = torch.stack(timesteps_t_latents)
            timesteps_t_latents = rearrange(timesteps_t_latents, "t b c h w -> b t c h w")
            timesteps_eps_samples = torch.stack(timesteps_eps_samples)
            timesteps_eps_samples = rearrange(timesteps_eps_samples, "t b c h w -> b t c h w")
            timesteps_pred_xstart_samples = torch.stack(timesteps_pred_xstart_samples)
            timesteps_pred_xstart_samples = rearrange(timesteps_pred_xstart_samples, "t b c h w -> b t c h w")
            all_t_latents.append(timesteps_t_latents)
            all_t_eps_samples.append(timesteps_eps_samples)
            all_t_pred_xstart_samples.append(timesteps_pred_xstart_samples)
        latents.append(xj.cpu())

    latents = torch.cat(latents)

    if from_each_t:
        all_t_latents = torch.cat(all_t_latents)
        all_t_latents = rearrange(all_t_latents, "b t c h w -> t b c h w")
        all_t_eps_samples = torch.cat(all_t_eps_samples)
        all_t_eps_samples = rearrange(all_t_eps_samples, "b t c h w -> t b c h w")
        all_t_pred_xstart_samples = torch.cat(all_t_pred_xstart_samples)
        all_t_pred_xstart_samples = rearrange(all_t_pred_xstart_samples, "b t c h w -> t b c h w")

        return {
            "latents": latents,
            "all_t_latents": all_t_latents,
            "all_t_eps_latents": all_t_eps_samples,
            "all_t_pred_xstart_latents": all_t_pred_xstart_samples,
        }

    else:
        return {"latents": latents}
