import logging
import os
from pathlib import Path

import hydra
import torch
import torchvision.transforms as T
from diffusers import DDIMScheduler
from diffusers.utils import load_image as hf_load_image
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision.utils import save_image

from taba._hydra import CONFIG_DIR
from taba.ext.masactrl.diffuser_utils import MasaCtrlPipeline, seed_everything
from taba.ext.masactrl.masactrl import MutualSelfAttentionControl
from taba.ext.masactrl.masactrl_utils import (
    AttentionBase,
    regiter_attention_editor_diffusers,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_image(image, device):
    """Load a real image as a [-1, 1] tensor of shape (1, 3, 512, 512)."""
    if isinstance(image, str):
        image = hf_load_image(image)  # supports local paths and URLs
    if not isinstance(image, Image.Image):
        raise ValueError("input must be a file path or PIL.Image.Image")
    original_size = image.size
    transform = T.Compose(
        [
            T.Lambda(lambda im: im.convert("RGB")),  # drop alpha / ensure 3 channels
            T.Resize((512, 512), interpolation=Image.BICUBIC),
            T.ToTensor(),  # scales to [0, 1]
            T.Lambda(lambda x: x * 2 - 1),  # [0, 1] -> [-1, 1]
        ]
    )
    image = transform(image).unsqueeze(0).to(device)  # (1, 3, 512, 512)
    return image, original_size


def main(
    image_path: str,
    source_prompt: str,
    target_prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    inv_guidance_scale: float,
    forward_t: int,
    forward_seed: int,
    masactrl_step: int,
    masactrl_layer: int,
    model_path: str,
    output_dir: str,
    seed: int,
) -> None:
    if image_path is None:
        raise ValueError("image_path must be provided (path to the image to edit).")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Build the MasaCtrl pipeline from SD1.4 with the notebook's DDIM scheduler.
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )
    model = MasaCtrlPipeline.from_pretrained(model_path, scheduler=scheduler).to(device)

    seed_everything(seed)

    # Load the user image (resize 512x512, to [-1, 1]).
    source_image, original_size = load_image(image_path, device)
    logger.info("Loaded image %s (original size %s)", image_path, original_size)

    prompts = [source_prompt, target_prompt]

    # Invert the source image, replacing the first `forward_t` steps with forward diffusion (our method).
    start_code, latents_list = model.invert(
        source_image,
        source_prompt,
        guidance_scale=inv_guidance_scale,
        num_inference_steps=num_inference_steps,
        return_intermediates=True,
        with_forward=True,
        forward_t=forward_t,
        forward_seed=forward_seed,
    )
    start_code = start_code.expand(len(prompts), -1, -1, -1)

    # Direct synthesis of the target prompt (no mutual self-attention control) for reference.
    editor = AttentionBase()
    regiter_attention_editor_diffusers(model, editor)
    image_fixed = model(
        [target_prompt],
        latents=start_code[-1:],
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )

    # MasaCtrl edit with mutual self-attention control.
    editor = MutualSelfAttentionControl(masactrl_step, masactrl_layer)
    regiter_attention_editor_diffusers(model, editor)
    image_masactrl = model(
        prompts,
        latents=start_code,
        guidance_scale=guidance_scale,
    )

    os.makedirs(output_dir, exist_ok=True)
    out_dir = Path(output_dir)

    # source_image is in [-1, 1]; the model outputs are already in [0, 1].
    save_image(source_image * 0.5 + 0.5, out_dir / "input.png")  # the input image
    save_image(image_masactrl[0:1], out_dir / "reconstruction.png")  # source prompt + MasaCtrl: reconstruction of input
    save_image(image_fixed, out_dir / "edit_without_masactrl.png")  # target prompt, no control: baseline edit
    save_image(image_masactrl[-1:], out_dir / "edited.png")  # target prompt + MasaCtrl: the edit

    out_grid = torch.cat(
        [
            source_image * 0.5 + 0.5,
            image_masactrl[0:1],
            image_fixed,
            image_masactrl[-1:],
        ],
        dim=0,
    )
    save_image(out_grid, out_dir / "grid.png", nrow=4)

    logger.info("Saved edit results to %s", out_dir)


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="masactrl/edit_real")
def cli(cfg: DictConfig) -> None:
    logger.info("Resolved config:\n%s", OmegaConf.to_yaml(cfg))
    main(**cfg)


if __name__ == "__main__":
    cli()
