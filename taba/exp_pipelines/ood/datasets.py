from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision import transforms as T
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from tqdm import tqdm

from datasets.celeba import CelebA
from datasets.tiny_imagenet import TinyImageNetDataset


class UnNormalize(Normalize):
    def __init__(self, mean, std, *args, **kwargs):
        new_mean = [-m / s for m, s in zip(mean, std)]
        new_std = [1 / s for s in std]
        super().__init__(new_mean, new_std, *args, **kwargs)


TRANSFORMS_TEST_NONORMALIZE = Compose([Resize((64, 64)), ToTensor()])
TRANSFORMS_CELEBA_DENORMALIZE = Compose(
    [
        Resize((64, 64)),
        ToTensor(),
        Normalize(mean=[0.4998, 0.4189, 0.3766], std=[0.3054, 0.2838, 0.2829]),
        UnNormalize(mean=[0.4714, 0.4411, 0.3918], std=[0.2753, 0.2679, 0.2791]),
    ]
)

TRANSFORMS_CELEBA64_TEST = Compose(
    [Resize((64, 64)), ToTensor(), Normalize(mean=[0.4998, 0.4189, 0.3766], std=[0.3054, 0.2838, 0.2829])]
)
TRANSFORMS_TINY_IMAGENET64_TEST = Compose(
    [Resize((64, 64)), ToTensor(), Normalize(mean=[0.4714, 0.4411, 0.3918], std=[0.2753, 0.2679, 0.2791])]
)

TRANSFORM = T.Compose([T.Resize((32, 32)), T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


class Cifar100Dataset(Dataset):
    def __init__(self, transform):
        self.cifar_training_dataset = datasets.CIFAR100("./data_src", train=True, download=True, transform=transform)

    def __len__(self):
        return len(self.cifar_training_dataset)

    def __getitem__(self, index):
        return self.cifar_training_dataset[index][0]


def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data in tqdm(dataloader, desc="Calculating mean and std"):
        if isinstance(data, list) or isinstance(data, tuple):
            data = data[0]
        elif isinstance(data, dict):
            data = data["image"]
        else:
            raise Exception("Unknown data type")
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean**2) ** 0.5

    return mean, std


def get_celeba_test_loader(batch_size: int = 128):
    transforms = Compose(
        [Resize((64, 64)), ToTensor(), Normalize(mean=[0.4998, 0.4189, 0.3766], std=[0.3054, 0.2838, 0.2829])]
    )
    ds = CelebA(root="data", split="test", transform=transforms)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def get_tiny_imagenet_test_loader(batch_size: int = 128):
    transforms = Compose(
        [Resize((64, 64)), ToTensor(), Normalize(mean=[0.4714, 0.4411, 0.3918], std=[0.2753, 0.2679, 0.2791])]
    )
    ds = TinyImageNetDataset(split="test", transform=transforms)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def get_ds_samples(n_max_samples: int = 10_000, dir: str = "experiments/ood"):
    Path(dir).mkdir(parents=True, exist_ok=True)

    ds_celeba = CelebA(root="data", split="test", transform=TRANSFORMS_CELEBA64_TEST)
    samples_celeba = torch.stack([ds_celeba[i][0] for i in range(min(len(ds_celeba), n_max_samples))])
    torch.save(samples_celeba, Path(dir) / "celeba_samples.pt")

    ds_imagenet = TinyImageNetDataset(split="test", transform=TRANSFORMS_TINY_IMAGENET64_TEST)
    samples_imagenet = torch.stack([ds_imagenet[i] for i in range(min(len(ds_imagenet), n_max_samples))])
    torch.save(samples_imagenet, Path(dir) / "imagenet_samples.pt")

    print(f"Saved samples to {dir}")

    return samples_celeba, samples_imagenet


def get_ds_samples_fid(n_max_samples: int = 10_000, dir: str = "experiments/fid"):
    Path(dir).mkdir(parents=True, exist_ok=True)

    ds_celeba = CelebA(root="data", split="test", transform=TRANSFORMS_TEST_NONORMALIZE)
    samples_celeba = torch.stack([ds_celeba[i][0] for i in range(min(len(ds_celeba), n_max_samples))])
    torch.save(samples_celeba, Path(dir) / "celeba_samples.pt")

    ds_imagenet = TinyImageNetDataset(split="test", transform=TRANSFORMS_TEST_NONORMALIZE)
    samples_imagenet = torch.stack([ds_imagenet[i] for i in range(min(len(ds_imagenet), n_max_samples))])
    torch.save(samples_imagenet, Path(dir) / "imagenet_samples.pt")

    print(f"Saved samples to {dir}")

    return samples_celeba, samples_imagenet

def get_ds_samples_ood(n_max_samples: int = 10_000, dir: str = "experiments/ood_denorm"):
    Path(dir).mkdir(parents=True, exist_ok=True)

    ds_celeba = CelebA(root="data", split="test", transform=TRANSFORMS_CELEBA_DENORMALIZE)
    samples_celeba = torch.stack([ds_celeba[i][0] for i in range(min(len(ds_celeba), n_max_samples))])
    torch.save(samples_celeba, Path(dir) / "celeba_samples.pt")

    ds_imagenet = TinyImageNetDataset(split="test", transform=TRANSFORMS_TEST_NONORMALIZE)
    samples_imagenet = torch.stack([ds_imagenet[i] for i in range(min(len(ds_imagenet), n_max_samples))])
    torch.save(samples_imagenet, Path(dir) / "imagenet_samples.pt")

    print(f"Saved samples to {dir}")

    return samples_celeba, samples_imagenet
