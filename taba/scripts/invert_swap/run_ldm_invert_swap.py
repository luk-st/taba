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
from taba.metrics.normality import stats_from_tensor
from taba.models.ldms.models import get_inv_scheduler, get_ldm_celeba, get_scheduler
from taba.models.ldms.sample_ldm import decode_image
from taba.models.ldms.sample_ldm import generate_latents as ldm_generate_latents
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


def save_tensor(accelerator: Accelerator, tensors: Dict[str, torch.Tensor], path_dir: str):
    if accelerator.is_main_process:
        for name, tensor in tensors.items():
            torch.save(tensor.cpu(), (Path(path_dir) / f"{name}.pt").resolve())
    accelerator.wait_for_everyone()


def sample_ldm(
    input_obj: torch.Tensor,
    T: int,
    batch_size: int,
    do_inversion: bool,
    from_each_t: bool,
    swaps_per_t: dict[int, torch.Tensor],
    swap_type: str | None,
    accelerator: Accelerator,
):
    ldm_unet, vae = get_ldm_celeba(device=accelerator.device)
    scheduler = get_scheduler(T=T) if not do_inversion else get_inv_scheduler(T=T)
    if not do_inversion:
        output = ldm_generate_samples(
            noise=input_obj,
            diffusion_unet=ldm_unet,
            diffusion_scheduler=scheduler,
            batch_size=batch_size,
            device=accelerator.device,
            from_each_t=from_each_t,
            accelerator=accelerator,
        )
        samples_decoded = decode_image(
            unet_out=output["samples"],
            vqvae=vae,
            batch_size=batch_size,
            device=accelerator.device,
            accelerator=accelerator,
        )
        output["samples_decoded"] = samples_decoded
    else:

        swap_xt, swap_eps = {}, {}
        if swap_type == "xt":
            swap_xt = swaps_per_t
        elif swap_type == "eps":
            swap_eps = swaps_per_t

        output = ldm_generate_latents(
            samples=input_obj,
            diffusion_unet=ldm_unet,
            diffusion_scheduler=scheduler,
            batch_size=batch_size,
            device=accelerator.device,
            from_each_t=from_each_t,
            swap_eps=swap_eps,
            swap_xt=swap_xt,
            accelerator=accelerator,
        )
    return output


def sample_multigpu(
    input_obj: torch.Tensor,
    T: int,
    batch_size: int,
    accelerator: Accelerator,
    do_inversion: bool,
    from_each_t: bool,
    swaps_per_t: dict[int, torch.Tensor],
    swap_type: str | None,
):
    assert do_inversion or (not do_inversion and swaps_per_t == {}), f"swaps_per_t should be empty during sampling"

    all_input = list(range(input_obj.shape[0]))
    all_outputs = defaultdict(list)
    with accelerator.split_between_processes(all_input) as device_indices:
        input_obj_device = torch.stack([input_obj[i] for i in device_indices])
        swaps_per_t_device = (
            {t: torch.stack([swaps_per_t[t][i] for i in device_indices]) for t in swaps_per_t.keys()}
            if swaps_per_t != {}
            else {}
        )

        if accelerator.is_main_process:
            print(f"Rank {accelerator.process_index} | {(len(device_indices)/len(all_input)):.2f}")
        outputs = sample_ldm(
            input_obj=input_obj_device,
            T=T,
            batch_size=batch_size,
            do_inversion=do_inversion,
            from_each_t=from_each_t,
            swaps_per_t=swaps_per_t_device,
            swap_type=swap_type,
            accelerator=accelerator,
        )
        for key, value in outputs.items():
            all_outputs[key].append(value.cpu())
    accelerator.wait_for_everyone()
    for obj_name, obj_value in all_outputs.items():
        vals = gather_object(obj_value)
        all_outputs[obj_name] = torch.cat(vals, dim=0)
    return all_outputs


def main(
    num_inference_steps: int,
    input_image_path: str,
    with_reconstruction: bool,
    seed: int,
    batch_size: int,
    internal: bool,
    swap_path: str | None,
    swap_before_t: int | None,
    start_time: str = START_TIME,
    swap_type: str | None = None,
    n_parts: int = 1,
    part_idx: int = 0,
    save_dir: str | None = None,
):
    if save_dir is None:
        save_dir = (
            f"experiments/celeba_ldm_256/invert/swap_{swap_type}_before{swap_before_t}/"
            f"{start_time}_"
            f"seed_{seed}_"
            f"T_{num_inference_steps}_"
            f"batch_size_{batch_size}_"
            f"with_reconstruction_{with_reconstruction}_"
            f"internal_{internal}"
        )
    if n_parts > 1:
        save_dir += f"_part_{part_idx}"

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800 * 6))
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
        logging.info(f"T: {num_inference_steps}")
        logging.info(f"Seed: {seed}")
        logging.info(f"Device: {accelerator.device}")
        logging.info(f"Save dir: {save_dir}")
        logging.info(f"Batch size: {batch_size}")
        logging.info(f"Swap path: {swap_path}")
        logging.info(f"Input image path: {input_image_path}")
        logging.info(f"With reconstruction: {with_reconstruction}")
        logging.info(f"Internal: {internal}")
        logging.info(f"Swap type: {swap_type}")
        logging.info(f"Swap before t: {swap_before_t}")
        logging.info(f"N parts: {n_parts}")
        logging.info(f"Part idx: {part_idx}")
    set_seed(seed)

    if input_image_path is None or not os.path.exists(input_image_path):
        raise ValueError(f"Input image path {input_image_path} does not exist")
    else:
        image = torch.load(input_image_path, weights_only=False)

    if swap_before_t is not None and swap_before_t > 0:
        assert swap_type is not None, "swap_type must be provided"
        assert swap_path is not None and os.path.exists(
            swap_path
        ), f"Swap path {swap_path} does not exist, but {swap_before_t=} is greater than 0"
        swap_tensor = torch.load(swap_path, weights_only=False)
        swap_inc = 1 if swap_type == "eps" else 2
        swaps_per_t = {idx: swap_tensor[-(idx + swap_inc)] for idx in range(swap_before_t)}
    else:
        swaps_per_t = {}

    if n_parts > 1:
        assert image.shape[0] % n_parts == 0, f"Image shape {image.shape[0]} is not divisible by {n_parts}"
        objects_per_part = image.shape[0] // n_parts
        image = image[objects_per_part * part_idx : objects_per_part * (part_idx + 1)]
        swaps_per_t = {
            t: swaps_per_t[t][objects_per_part * part_idx : objects_per_part * (part_idx + 1)] for t in swaps_per_t
        }

    if accelerator.is_main_process:
        logging.info(f"Image shape: {image.shape}")
    save_tensor(accelerator=accelerator, tensors={"samples": image}, path_dir=save_dir)

    if accelerator.is_main_process:
        logging.info("Inversion: image -> latents")
    inversion_outputs = sample_multigpu(
        input_obj=image,
        T=num_inference_steps,
        batch_size=batch_size,
        accelerator=accelerator,
        do_inversion=True,
        from_each_t=internal,
        swaps_per_t=swaps_per_t,
        swap_type=swap_type,
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
            swaps_per_t={},
            swap_type=None,
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
        metrics["_params"]["with_reconstruction"] = with_reconstruction
        metrics["_params"]["input_image_path"] = input_image_path
        metrics["_params"]["swap_path"] = swap_path
        metrics["_params"]["internal"] = internal
        metrics["_params"]["swap_type"] = swap_type
        metrics["_params"]["swap_before_t"] = swap_before_t
        metrics["_params"]["n_parts"] = n_parts
        metrics["_params"]["part_idx"] = part_idx
        metrics["_params"]["save_dir"] = save_dir
        metrics["metrics"] = {}
        metrics["metrics"]["correlation"] = get_top_k_corr_in_patches(inversion_outputs["latents"], top_k=20)
        metrics["metrics"]["latent_stats"] = stats_from_tensor(inversion_outputs["latents"])

        # if with_reconstruction:
        #     metrics["metrics"]["reconstruction_error_images"] = reconstruction_error(
        #         reconstruction_outputs["samples2_decoded"], inversion_outputs["samples_decoded"]
        #     )
        logging.info(metrics)

        with open(Path(save_dir) / "metrics.json", "w") as f:
            json.dump(metrics, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_inference_steps", type=int, default=NUM_INFERENCE_STEPS_DEFAULT)
    parser.add_argument("--input_image_path", type=str, required=True)
    parser.add_argument("--with_reconstruction", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument("--internal", action="store_true")
    parser.add_argument("--swap_path", type=str, default=None)
    parser.add_argument("--swap_before_t", type=int, default=None)
    parser.add_argument("--swap_type", type=str, choices=["eps", "xt"], default=None)
    parser.add_argument("--n_parts", type=int, default=1)
    parser.add_argument("--part_idx", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()
    main(
        num_inference_steps=args.num_inference_steps,
        input_image_path=args.input_image_path,
        with_reconstruction=args.with_reconstruction,
        seed=args.seed,
        batch_size=args.batch_size,
        internal=args.internal,
        swap_path=args.swap_path,
        swap_before_t=args.swap_before_t,
        swap_type=args.swap_type,
        n_parts=args.n_parts,
        part_idx=args.part_idx,
        save_dir=args.save_dir,
    )
