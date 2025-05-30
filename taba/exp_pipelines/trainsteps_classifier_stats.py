import os

import torch
from tqdm import tqdm

from taba.models.adm.models import get_ls_cifar10
from taba.models.adm.ddim import generate_noises, generate_samples
from taba.metrics.annotator import (
    GenerationsDataset,
    annotate,
    get_vit_cifar10_annotator,
)

NUMBER_OF_SAMPLES = 1024
BATCH_SIZE = 256

annotator = get_vit_cifar10_annotator()

dir_path = f"experiments/trainsteps/openai_cifar10_otherseed"
os.makedirs(dir_path, exist_ok=True)

for tr_step in tqdm(range(0, 570_000, 5_000)):
    model, diffusion, args = get_ls_cifar10(
        steps=100,
        model_path=f"res/openai_cifar10_learnsigmafalse_otherseed_checkpointed/model{tr_step:06}.pt",
        learn_sigma=False,
    )
    noises = generate_noises(NUMBER_OF_SAMPLES, args)
    samples = generate_samples(
        random_noises=noises,
        number_of_samples=NUMBER_OF_SAMPLES,
        batch_size=BATCH_SIZE,
        diffusion_pipeline=diffusion,
        ddim_model=model,
        diffusion_args=args,
    )
    dataset = GenerationsDataset(x=samples)
    labels = annotate(dataset=dataset, n_samples=NUMBER_OF_SAMPLES, batch_size=BATCH_SIZE, annotator=annotator)
    samples_final = samples.detach().cpu()
    labels_final = torch.stack(labels).detach().cpu()
    torch.save(labels_final, f"{dir_path}/labels_{tr_step:06}.pt")
    torch.save(samples_final, f"{dir_path}/samples_{tr_step:06}.pt")
