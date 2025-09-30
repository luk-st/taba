import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch_fidelity
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import json


def ldm_tensor_to_images(ldm_out):
    image_processed = ldm_out.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.clamp(0, 255).numpy().astype(np.uint8)
    return [Image.fromarray(img_processed) for img_processed in image_processed]


def tensors_to_pil_single(tensor):
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()
    return transforms.ToPILImage()(tensor)


def save_image(idx_and_sample, save_dir):
    idx, sample = idx_and_sample
    sample.save(save_dir / f"{idx}.png")


def tens_to_imagedir(tens_path: Path):
    dir_name = tens_path.name[:-3]
    save_dir = (tens_path.parent / (dir_name + "_dir")).resolve()
    if save_dir.exists():
        print(f"Directory {save_dir} already exists")
    else:
        os.makedirs(save_dir, exist_ok=True)
        tensor = torch.load(tens_path, weights_only=False, map_location="cpu")
        if "celeba_ldm_256" in str(tens_path):
            images = ldm_tensor_to_images(tensor)
        else:
            images = [tensors_to_pil_single(tensor[idx]) for idx in range(tensor.shape[0])]

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(
                tqdm(
                    executor.map(partial(save_image, save_dir=save_dir), enumerate(images)),
                    total=len(images),
                    desc="Saving images",
                )
            )

    return save_dir


def process_cifar_metrics(alphas_to_paths):
    metrics_per_alpha = {}
    for alpha, path in tqdm(alphas_to_paths.items(), desc="Processing metrics"):
        print("ALPHA: ", alpha)
        path = tens_to_imagedir(path)
        metrics = torch_fidelity.calculate_metrics(
            input1="cifar10-train",
            input2=str(path),
            cuda=True,
            isc=True,
            fid=True,
            kid=True,
            prc=True,
            verbose=False,
            datasets_root="/net/scratch/hscra/plgrid/plglukaszst/torch_cache",
            cache_root="/net/scratch/hscra/plgrid/plglukaszst/torch_cache"
        )
        # unlink dir
        shutil.rmtree(path)
        metrics_per_alpha[alpha] = metrics
    return metrics_per_alpha


def process_imagenet_metrics(alphas_to_paths, ref_path):
    metrics_per_alpha = {}
    for alpha, path in tqdm(alphas_to_paths.items(), desc="Processing metrics"):
        print("ALPHA: ", alpha)
        path = tens_to_imagedir(path)
        metrics = torch_fidelity.calculate_metrics(
            input1=str(ref_path),
            input2=str(path),
            cuda=True,
            isc=True,
            fid=True,
            kid=True,
            prc=True,
            verbose=False,
            datasets_root="/net/scratch/hscra/plgrid/plglukaszst/torch_cache",
            cache_root="/net/scratch/hscra/plgrid/plglukaszst/torch_cache"
        )
        # unlink dir
        shutil.rmtree(path)
        metrics_per_alpha[alpha] = metrics
    return metrics_per_alpha

if __name__ == "__main__":

    # alphas_to_paths_noise = {
    #     "latent_ours_0000": Path("experiments/interpolations_adms/cifar_pixel_32/s0/invert_forward2_s10/samples2.pt"),
    #     "latent_ours_0167": Path("experiments/interpolations_adms/cifar_pixel_32/interpolate/sample_latents_ours_0167/samples.pt"),
    #     "latent_ours_0333": Path("experiments/interpolations_adms/cifar_pixel_32/interpolate/sample_latents_ours_0333/samples.pt"),
    #     "latent_ours_0500": Path("experiments/interpolations_adms/cifar_pixel_32/interpolate/sample_latents_ours_0500/samples.pt"),
    #     "latent_ours_0667": Path("experiments/interpolations_adms/cifar_pixel_32/interpolate/sample_latents_ours_0667/samples.pt"),
    #     "latent_ours_0833": Path("experiments/interpolations_adms/cifar_pixel_32/interpolate/sample_latents_ours_0833/samples.pt"),
    #     "latent_ours_1000": Path("experiments/interpolations_adms/cifar_pixel_32/s420/invert_forward2_s42/samples2.pt"),
    # }
    # metrics_per_alpha_noise = process_cifar_metrics(alphas_to_paths_noise)
    # json.dump(metrics_per_alpha_noise, open("experiments/interpolations_adms/cifar_pixel_32/interpolate/fid_scores_11092025.json", "w"))

    # alphas_to_paths_noise = {
    #     "latent_ours_0000": Path("experiments/interpolations_adms/imagenet_pixel_64/s0/invert_forward2_s10/samples2.pt"),
    #     "latent_ours_0167": Path("experiments/interpolations_adms/imagenet_pixel_64/interpolate/sample_latents_ours_0167/samples.pt"),
    #     "latent_ours_0333": Path("experiments/interpolations_adms/imagenet_pixel_64/interpolate/sample_latents_ours_0333/samples.pt"),
    #     "latent_ours_0500": Path("experiments/interpolations_adms/imagenet_pixel_64/interpolate/sample_latents_ours_0500/samples.pt"),
    #     "latent_ours_0667": Path("experiments/interpolations_adms/imagenet_pixel_64/interpolate/sample_latents_ours_0667/samples.pt"),
    #     "latent_ours_0833": Path("experiments/interpolations_adms/imagenet_pixel_64/interpolate/sample_latents_ours_0833/samples.pt"),
    #     "latent_ours_1000": Path("experiments/interpolations_adms/imagenet_pixel_64/s420/invert_forward2_s42/samples2.pt"),
    # }
    # metrics_per_alpha_noise = process_imagenet_metrics(alphas_to_paths_noise, Path("data/imagenet64_ref/imagenet64_ref_dir"))
    # json.dump(metrics_per_alpha_noise, open("experiments/interpolations_adms/imagenet_pixel_64/interpolate/fid_scores_11092025.json", "w"))

    # alphas_to_paths_noise = {
    #     "latent_ours_0000": Path("experiments/interpolations_adms/imagenet_pixel_256/s0/invert_forward2_s10/samples2.pt"),
    #     "latent_ours_0167": Path("experiments/interpolations_adms/imagenet_pixel_256/interpolate/sample_latents_ours_0167/samples.pt"),
    #     "latent_ours_0333": Path("experiments/interpolations_adms/imagenet_pixel_256/interpolate/sample_latents_ours_0333/samples.pt"),
    #     "latent_ours_0500": Path("experiments/interpolations_adms/imagenet_pixel_256/interpolate/sample_latents_ours_0500/samples.pt"),
    #     "latent_ours_0667": Path("experiments/interpolations_adms/imagenet_pixel_256/interpolate/sample_latents_ours_0667/samples.pt"),
    #     "latent_ours_0833": Path("experiments/interpolations_adms/imagenet_pixel_256/interpolate/sample_latents_ours_0833/samples.pt"),
    #     "latent_ours_1000": Path("experiments/interpolations_adms/imagenet_pixel_256/s420/invert_forward2_s42/samples2.pt"),
    # }
    # metrics_per_alpha_noise = process_imagenet_metrics(alphas_to_paths_noise, Path("data/imagenet256_ref/imagenet256_ref_dir"))
    # json.dump(metrics_per_alpha_noise, open("experiments/interpolations_adms/imagenet_pixel_256/interpolate/fid_scores_11092025.json", "w"))

    # alphas_to_paths_noise = {
    #     "latent_ours_0000": Path("experiments/dit/interpolate/samples_latents_ours/sample_0000/decoded_samples.pt"),
    #     "latent_ours_0167": Path("experiments/dit/interpolate/samples_latents_ours/sample_0167/decoded_samples.pt"),
    #     "latent_ours_0333": Path("experiments/dit/interpolate/samples_latents_ours/sample_0333/decoded_samples.pt"),
    #     "latent_ours_0500": Path("experiments/dit/interpolate/samples_latents_ours/sample_0500/decoded_samples.pt"),
    #     "latent_ours_0667": Path("experiments/dit/interpolate/samples_latents_ours/sample_0667/decoded_samples.pt"),
    #     "latent_ours_0833": Path("experiments/dit/interpolate/samples_latents_ours/sample_0833/decoded_samples.pt"),
    #     "latent_ours_1000": Path("experiments/dit/interpolate/samples_latents_ours/sample_1000/decoded_samples.pt"),
    # }
    # metrics_per_alpha_noise = process_imagenet_metrics(alphas_to_paths_noise, Path("data/imagenet256_ref/imagenet256_ref_dir"))
    # json.dump(metrics_per_alpha_noise, open("experiments/dit/interpolate/fid_scores_latents_ours_11092025.json", "w"))

    # alphas_to_paths_noise = {
    #     "latent_ours_0000": Path("experiments/dit/interpolate/samples_latents_ddim/sample_0000/decoded_samples.pt"),
    #     "latent_ours_0167": Path("experiments/dit/interpolate/samples_latents_ddim/sample_0167/decoded_samples.pt"),
    #     "latent_ours_0333": Path("experiments/dit/interpolate/samples_latents_ddim/sample_0333/decoded_samples.pt"),
    #     "latent_ours_0500": Path("experiments/dit/interpolate/samples_latents_ddim/sample_0500/decoded_samples.pt"),
    #     "latent_ours_0667": Path("experiments/dit/interpolate/samples_latents_ddim/sample_0667/decoded_samples.pt"),
    #     "latent_ours_0833": Path("experiments/dit/interpolate/samples_latents_ddim/sample_0833/decoded_samples.pt"),
    #     "latent_ours_1000": Path("experiments/dit/interpolate/samples_latents_ddim/sample_1000/decoded_samples.pt"),
    # }
    # metrics_per_alpha_noise = process_imagenet_metrics(alphas_to_paths_noise, Path("data/imagenet256_ref/imagenet256_ref_dir"))
    # json.dump(metrics_per_alpha_noise, open("experiments/dit/interpolate/fid_scores_latents_ddim_11092025.json", "w"))

    # alphas_to_paths_noise = {
    #     "latent_ours_0000": Path("experiments/dit/interpolate/samples_noise/sample_0000/decoded_samples.pt"),
    #     "latent_ours_0167": Path("experiments/dit/interpolate/samples_noise/sample_0167/decoded_samples.pt"),
    #     "latent_ours_0333": Path("experiments/dit/interpolate/samples_noise/sample_0333/decoded_samples.pt"),
    #     "latent_ours_0500": Path("experiments/dit/interpolate/samples_noise/sample_0500/decoded_samples.pt"),
    #     "latent_ours_0667": Path("experiments/dit/interpolate/samples_noise/sample_0667/decoded_samples.pt"),
    #     "latent_ours_0833": Path("experiments/dit/interpolate/samples_noise/sample_0833/decoded_samples.pt"),
    #     "latent_ours_1000": Path("experiments/dit/interpolate/samples_noise/sample_1000/decoded_samples.pt"),
    # }
    # metrics_per_alpha_noise = process_imagenet_metrics(alphas_to_paths_noise, Path("data/imagenet256_ref/imagenet256_ref_dir"))
    # json.dump(metrics_per_alpha_noise, open("experiments/dit/interpolate/fid_scores_noise_11092025.json", "w"))

    # PRIOR = "noise" # latents_ddim, noise, latents_ours

    # alphas_to_paths_noise = {
    #     "latent_ours_0000": Path(f"experiments/celeba_ldm_256/interpolate/samples_{PRIOR}/sample_0000/samples_decoded.pt"),
    #     "latent_ours_0167": Path(f"experiments/celeba_ldm_256/interpolate/samples_{PRIOR}/sample_0167/samples_decoded.pt"),
    #     "latent_ours_0333": Path(f"experiments/celeba_ldm_256/interpolate/samples_{PRIOR}/sample_0333/samples_decoded.pt"),
    #     "latent_ours_0500": Path(f"experiments/celeba_ldm_256/interpolate/samples_{PRIOR}/sample_0500/samples_decoded.pt"),
    #     "latent_ours_0667": Path(f"experiments/celeba_ldm_256/interpolate/samples_{PRIOR}/sample_0667/samples_decoded.pt"),
    #     "latent_ours_0833": Path(f"experiments/celeba_ldm_256/interpolate/samples_{PRIOR}/sample_0833/samples_decoded.pt"),
    #     "latent_ours_1000": Path(f"experiments/celeba_ldm_256/interpolate/samples_{PRIOR}/sample_1000/samples_decoded.pt"),
    # }
    # metrics_per_alpha_noise = process_imagenet_metrics(alphas_to_paths_noise, Path("data/celebahq/celeba_hq_256"))
    # json.dump(metrics_per_alpha_noise, open(f"experiments/celeba_ldm_256/interpolate/fid_scores_{PRIOR}_11092025.json", "w"))

    alphas_to_paths_noise = {
        "latent_ours_0000": Path("experiments/interpolations_adms/imagenet_pixel_256/interpolate/samples_latents_ours/sample_0000/samples.pt"),
        "latent_ours_0167": Path("experiments/interpolations_adms/imagenet_pixel_256/interpolate/samples_latents_ours/sample_0167/samples.pt"),
        "latent_ours_0333": Path("experiments/interpolations_adms/imagenet_pixel_256/interpolate/samples_latents_ours/sample_0333/samples.pt"),
        "latent_ours_0500": Path("experiments/interpolations_adms/imagenet_pixel_256/interpolate/samples_latents_ours/sample_0500/samples.pt"),
        "latent_ours_0667": Path("experiments/interpolations_adms/imagenet_pixel_256/interpolate/samples_latents_ours/sample_0667/samples.pt"),
        "latent_ours_0833": Path("experiments/interpolations_adms/imagenet_pixel_256/interpolate/samples_latents_ours/sample_0833/samples.pt"),
        "latent_ours_1000": Path("experiments/interpolations_adms/imagenet_pixel_256/interpolate/samples_latents_ours/sample_1000/samples.pt"),
    }
    metrics_per_alpha_noise = process_imagenet_metrics(alphas_to_paths_noise, Path("data/imagenet256_ref/imagenet256_ref_dir"))
    json.dump(metrics_per_alpha_noise, open("experiments/interpolations_adms/imagenet_pixel_256/interpolate/samples_latents_ours/fid_scores_11092025_new.json", "w"))