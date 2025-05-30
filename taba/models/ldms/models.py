import torch
from diffusers.models.autoencoders.vq_model import VQModel
from diffusers.models.unets.unet_2d import UNet2DModel

from taba.ddim.schedulers import AdvancedDDIMInverseScheduler, AdvancedDDIMScheduler

DEFAULT_T = 100
CELEBA_LDM_256 = "CompVis/ldm-celebahq-256"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_ldm_celeba(device: torch.device = DEVICE):
    unet = UNet2DModel.from_pretrained(CELEBA_LDM_256, subfolder="unet")
    vae = VQModel.from_pretrained(CELEBA_LDM_256, subfolder="vqvae")
    return unet.to(device), vae.to(device)  # type: ignore


def get_scheduler(T: int = DEFAULT_T) -> AdvancedDDIMScheduler:
    scheduler = AdvancedDDIMScheduler.from_config(CELEBA_LDM_256, subfolder="scheduler")
    scheduler.set_timesteps(num_inference_steps=T)  # type: ignore
    return scheduler


def get_inv_scheduler(T: int = DEFAULT_T) -> AdvancedDDIMInverseScheduler:
    scheduler = AdvancedDDIMInverseScheduler.from_config(CELEBA_LDM_256, subfolder="scheduler")
    scheduler.set_timesteps(num_inference_steps=T)  # type: ignore
    return scheduler
