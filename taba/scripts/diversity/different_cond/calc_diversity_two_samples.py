import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers.utils import pt_to_pil

from taba.metrics.alignment import calculate_psnr, calculate_ssim
from taba.metrics.clip_dino import get_dino, get_dino_features, get_mean_cosine_sim
from taba.metrics.diversity import (
    dreamsim_calculate_distances,
    dreamsim_init,
    lpips_calculate_distances,
)


def interimage_distance(
    tensor_in: torch.Tensor, tensor_ref: torch.Tensor, dreamsim_model: Any, dreamsim_preprocessor: Any, dino_model: Any
):
    dists = {}

    dist_dreamsim = dreamsim_calculate_distances(
        tensor_in, tensor_ref, dreamsim_model, dreamsim_preprocessor, batch_size=128
    )
    dists["dreamsim"] = {"mean": dist_dreamsim.mean().item(), "std": dist_dreamsim.std().item()}

    dist_lpips = lpips_calculate_distances(tensor_in, tensor_ref, "cuda", batch_size=128)
    dists["lpips"] = {"mean": dist_lpips.mean().item(), "std": dist_lpips.std().item()}

    dist_ssim = np.array(
        [
            calculate_ssim(i1.permute(1, 2, 0).numpy(), i2.permute(1, 2, 0).numpy())
            for i1, i2 in zip(tensor_in, tensor_ref)
        ]
    )
    dists["ssim"] = {"mean": dist_ssim.mean().item(), "std": dist_ssim.std().item()}

    dist_psnr = np.array(
        [
            calculate_psnr(i1.permute(1, 2, 0).numpy(), i2.permute(1, 2, 0).numpy())
            for i1, i2 in zip(tensor_in, tensor_ref)
        ]
    )
    dists["psnr"] = {"mean": dist_psnr.mean().item(), "std": dist_psnr.std().item()}

    dist_dino = get_mean_cosine_sim(
        get_dino_features(pt_to_pil(tensor_in), dino_model, "cuda"),
        get_dino_features(pt_to_pil(tensor_ref), dino_model, "cuda"),
    )
    dists["dino"] = dist_dino.mean().item()

    return dists


def main(in_dir: str, ref_dir: str):
    device = "cuda"
    dreamsim_model, dreamsim_preprocessor = dreamsim_init(device)
    dino_model = get_dino(device)

    assert os.path.exists(in_dir) and os.path.exists(ref_dir), f"Both {in_dir=} and {ref_dir=} must exist"

    decoded = "ldm" in in_dir or "dit" in in_dir

    output_file = Path(in_dir) / "diversity_different_cond.json"

    samples_in = torch.load(
        Path(in_dir) / "decoded_samples.pt" if decoded else Path(in_dir) / "images.pt", weights_only=False
    )
    samples_ref = torch.load(
        Path(ref_dir) / "decoded_samples.pt" if decoded else Path(ref_dir) / "images.pt", weights_only=False
    )

    stats = interimage_distance(
        samples_in,
        samples_ref,
        dreamsim_model=dreamsim_model,
        dreamsim_preprocessor=dreamsim_preprocessor,
        dino_model=dino_model,
    )

    output = {
        "_config": {
            "in_dir": in_dir,
            "ref_dir": ref_dir,
            "decoded": decoded,
        },
        "stats": stats,
    }

    with open(output_file, "w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, required=True)
    parser.add_argument("--ref_dir", type=str, required=True)
    args = parser.parse_args()
    main(args.in_dir, args.ref_dir)
