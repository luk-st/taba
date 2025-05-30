import math
from typing import Callable, List, Tuple, Union

import lpips
import numpy as np
import PIL
import torch
from diffusers.utils import pt_to_pil
from dreamsim import dreamsim
from tqdm import tqdm


def dreamsim_init(device: torch.device) -> Tuple[torch.nn.Module, Callable]:
    model, preprocess = dreamsim(pretrained=True, device=device, cache_dir="./res/dreamsim")
    return model, preprocess


def dreamsim_calculate_distances(
    tensors1: Union[torch.Tensor, List[PIL.Image.Image]],
    tensors2: Union[torch.Tensor, List[PIL.Image.Image]],
    model: torch.nn.Module,
    preprocessor: Callable,
    batch_size: int = 64,
) -> torch.Tensor:
    if isinstance(tensors1, torch.Tensor):
        tensors1 = pt_to_pil(tensors1)
    if isinstance(tensors2, torch.Tensor):
        tensors2 = pt_to_pil(tensors2)

    images1 = torch.cat([preprocessor(gen) for gen in tensors1])
    images2 = torch.cat([preprocessor(gen) for gen in tensors2])
    n_images = images1.shape[0]

    distances = torch.zeros((n_images), device=torch.device("cpu"))
    for i in tqdm(range(0, math.ceil(n_images / batch_size)), desc="Calculating distances"):
        batch1 = images1[i * batch_size : (i + 1) * batch_size].to(model.device)
        batch2 = images2[i * batch_size : (i + 1) * batch_size].to(model.device)
        dist = model(batch1, batch2)
        distances[i * batch_size : (i + 1) * batch_size] = dist.cpu()
    return distances


def tensor_to_lpips_tensor(tensor: torch.Tensor) -> torch.Tensor:
    tens = pt_to_pil(tensor)
    tens = np.array(tens, dtype=np.uint8)
    tens = torch.tensor((tens / (255.0 / 2.0) - 1.0)[:, :, :].transpose((0, 3, 1, 2))).float()
    return tens


def lpips_calculate_distances(
    tensors1: Union[torch.Tensor, List[PIL.Image.Image]],
    tensors2: Union[torch.Tensor, List[PIL.Image.Image]],
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    loss_fn_alex = lpips.LPIPS(net="alex", version="0.1").to(device)
    tensors1 = tensor_to_lpips_tensor(tensors1)
    tensors2 = tensor_to_lpips_tensor(tensors2)

    with torch.no_grad():
        n_images = tensors1.shape[0]
        dists = torch.zeros((n_images), device="cpu")
        for i in tqdm(range(0, math.ceil(n_images / batch_size)), desc="Calculating distances"):
            batch1 = tensors1[i * batch_size : (i + 1) * batch_size].to(device)
            batch2 = tensors2[i * batch_size : (i + 1) * batch_size].to(device)
            dists[i * batch_size : (i + 1) * batch_size] = loss_fn_alex(batch1, batch2).squeeze().cpu()
    return dists
