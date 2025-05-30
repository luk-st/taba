import os
from pathlib import Path

import torch

from taba.models.adm.models import get_ddpm_imagenet256, get_openai_cifar, get_openai_imagenet
from taba.models.adm.ddim import generate_latents

BATCH_SIZE = 64
DEVICE = "cuda"

MODELNAME_TO_GET_FUNC = {
    "cifar_pixel_32": get_openai_cifar,
    "imagenet_pixel_64": get_openai_imagenet,
    "imagenet_pixel_256": get_ddpm_imagenet256,
}


def save_results(latents: torch.Tensor, T: int, model_name: str, save_path_dir: str, internal: bool = False):
    save_path = Path(save_path_dir).resolve()
    os.makedirs(save_path, exist_ok=True)
    torch.save(latents, save_path / f"latents_{model_name}_T{T}.pt")


def obtain_latents_pixel(samples: torch.Tensor, T: int, internal: bool = False):
    diffusion_model, diffusion_pipeline, diffusion_args = MODELNAME_TO_GET_FUNC["cifar_pixel_32"](
        steps=T, device=DEVICE
    )
    return invert_to_latents(
        samples=samples,
        ddpm_model=diffusion_model,
        diffusion_pipeline=diffusion_pipeline,
        diffusion_args=diffusion_args,
        internal=internal,
    )


def invert_to_latents(
    samples: torch.Tensor,
    ddpm_model,
    diffusion_pipeline,
    diffusion_args,
    internal: bool = False,
):

    latent_outs = generate_latents(
        ddim_generations=samples,
        batch_size=BATCH_SIZE,
        diffusion_pipeline=diffusion_pipeline,
        ddim_model=ddpm_model,
        device=DEVICE,
        from_each_t=internal,
    )

    if internal:
        latents = latent_outs["latents"]
        all_t_latents = latent_outs["all_t_latents"]
    else:
        latents = latent_outs
        all_t_latents = None
    return latents, all_t_latents


def main(model_name: str, T: int, samples_path: str, save_path_dir: str, internal: bool = False):
    samples = torch.load(samples_path)
    if model_name in ["cifar_pixel_32", "imagenet_pixel_64", "imagenet_pixel_256"]:
        latents, _ = obtain_latents_pixel(samples=samples, T=T, internal=internal)
    else:
        raise ValueError(f"Unknown dataset name: {model_name}")

    save_results(latents=latents, T=T, model_name=model_name, save_path_dir=save_path_dir, internal=internal)


if __name__ == "__main__":
    from datetime import datetime

    import submitit

    MODELS = ["imagenet_pixel_64"]
    TS = [10, 100, 1000]
    DS_NAMES = ["imagenet", "celeba"]
    INTERNAL = False
    LOG_FOLDER = f"slurm_out/ood_invert/{datetime.now().strftime('%Y%m%d')}/%j"

    executor = submitit.AutoExecutor(folder=LOG_FOLDER)
    executor.update_parameters(
        timeout_min=720,  # 4:30:00 converted to minutes (4*60 + 30 = 270)
        gpus_per_node=1,  # --gres gpu:1
        cpus_per_task=16,  # --cpus-per-task=16
        mem_gb=100,  # --mem 100G
        slurm_partition="<SLURM_PARTITION_NAME>",
        slurm_account="<SLURM_ACCOUNT_NAME>",
        nodes=1,  # --nodes 1
        tasks_per_node=1,  # --ntasks 1
    )
    with executor.batch():
        for model_name in MODELS:
            for ds_name in DS_NAMES:
                for T in TS:
                    executor.submit(
                        main,
                        model_name,
                        T,
                        f"experiments/ood_denorm/{ds_name}_samples.pt",
                        f"experiments/ood_denorm/{ds_name}",
                        INTERNAL,
                    )
