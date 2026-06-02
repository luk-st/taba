import logging
import sys
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
from diffusers import DDIMScheduler
from diffusers.utils import load_image
from diffusers.utils.logging import disable_progress_bar, enable_progress_bar
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

from taba._hydra import CONFIG_DIR
from taba.ext.style_aligned import inversion, sa_handler
from taba.ext.style_aligned.pipe_sdxl import CustomStableDiffusionXLPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SDXL_SIZE = (1024, 1024)
SHARED_SCORE_SCALE = 1.0


def run_single_styledrop_example(
    pipeline: CustomStableDiffusionXLPipeline,
    image_path: str,
    p_source: str,
    ps_target: list[str],
    num_inference_steps: int,
    guidance_scale_inv: float,
    forward_t: int,
    forward_seed: int,
    offset_inv: int,
    gs_sampling: float,
    batch_size: int,
    shared_score_shift: float,
    use_forward_diffusion: bool,
    seed: int,
):
    input_image = load_image(image_path)
    image_size = input_image.size
    x0 = np.array(input_image.resize(SDXL_SIZE))

    torch.cuda.empty_cache()

    if use_forward_diffusion:
        zts_inv = inversion.ddim_inversion_with_forward(
            pipeline, x0, p_source, num_inference_steps, guidance_scale_inv,
            forward_t=forward_t, forward_seed=forward_seed,
        )
    else:
        zts_inv = inversion.ddim_inversion(pipeline, x0, p_source, num_inference_steps, guidance_scale_inv)

    all_images = []
    recon_image = None

    for batch_start_idx in tqdm(range(0, len(ps_target), batch_size), desc="Transferring style"):
        torch.cuda.empty_cache()
        batch_target_prompts = ps_target[batch_start_idx:batch_start_idx + batch_size]
        batch_prompts = [p_source] + batch_target_prompts

        handler = sa_handler.Handler(pipeline)
        sa_args = sa_handler.StyleAlignedArgs(
            share_group_norm=True, share_layer_norm=True, share_attention=True,
            adain_queries=True, adain_keys=True, adain_values=False,
            shared_score_shift=np.log(shared_score_shift), shared_score_scale=SHARED_SCORE_SCALE,)
        handler.register(sa_args)
        zT, inversion_callback = inversion.make_inversion_callback(zts_inv, offset=offset_inv)

        g_cpu = torch.Generator(device='cpu').manual_seed(seed)
        latents = torch.randn(len(batch_prompts), 4, 128, 128, device='cpu', generator=g_cpu,
                              dtype=pipeline.unet.dtype).to('cuda')
        latents[0] = zT

        images = pipeline(batch_prompts, latents=latents, num_inference_steps=num_inference_steps,
                          callback_on_step_end=inversion_callback, guidance_scale=gs_sampling).images
        all_images.extend([
            img.resize(image_size) for img in images[1:]
        ])
        if recon_image is None:
            recon_image = images[0].resize(image_size)
    return recon_image, all_images


def main(
    image_path: str | None,
    p_source: str,
    ps_target: list[str],
    num_inference_steps: int,
    guidance_scale_inv: float,
    forward_t: int,
    forward_seed: int,
    offset_inv: int,
    gs_sampling: float,
    batch_size: int,
    shared_score_shift: float,
    use_forward_diffusion: bool,
    output_dir: str | None,
    seed: int,
):
    cmd = " ".join(sys.argv)
    if image_path is None:
        raise ValueError("image_path must be provided (path to the user image to transfer style from)")

    ps_target = list(OmegaConf.to_container(ps_target, resolve=True)) if OmegaConf.is_config(ps_target) else list(ps_target)

    if output_dir is None:
        curr_datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/style_aligned/{curr_datetime_str}"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_path = output_path.resolve()

    logging.info(f"Image path: {image_path}")
    logging.info(f"Source prompt: {p_source}")
    logging.info(f"Target prompts: {ps_target}")
    logging.info(f"Use forward diffusion: {use_forward_diffusion} (forward_t={forward_t}, forward_seed={forward_seed})")
    logging.info(f"Output dir: {output_path}")
    logging.info(f"Command: {cmd}")

    disable_progress_bar()

    scheduler = DDIMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
        clip_sample=False, set_alpha_to_one=False)

    pipeline = CustomStableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16",
        use_safetensors=True,
        scheduler=scheduler,
    ).to("cuda")
    enable_progress_bar()
    pipeline.set_progress_bar_config(disable=True)

    recon_image, transfered_images = run_single_styledrop_example(
        pipeline=pipeline,
        image_path=image_path,
        p_source=p_source,
        ps_target=ps_target,
        num_inference_steps=num_inference_steps,
        guidance_scale_inv=guidance_scale_inv,
        forward_t=forward_t,
        forward_seed=forward_seed,
        offset_inv=offset_inv,
        gs_sampling=gs_sampling,
        batch_size=batch_size,
        shared_score_shift=shared_score_shift,
        use_forward_diffusion=use_forward_diffusion,
        seed=seed,
    )

    # save reconstruction
    if recon_image is not None:
        recon_image.save((output_path / "reconstruction.png").resolve())
    # save styled outputs
    for idx_image, image in enumerate(transfered_images):
        image.save((output_path / f"image_{idx_image}.png").resolve())
    # save input image (load_image handles both local paths and URLs)
    load_image(image_path).save((output_path / "input.png").resolve())
    logging.info(f"Saved {len(transfered_images)} styled images + reconstruction + input to {output_path}")


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="style_aligned/transfer")
def cli(cfg: DictConfig) -> None:
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))
    main(**cfg)


if __name__ == "__main__":
    cli()
