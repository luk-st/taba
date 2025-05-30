import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split
from torchvision import transforms as T

TRANSFORM = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


class NoiseDataset(Dataset):
    def __init__(self, noises, labels, tf=TRANSFORM):
        super().__init__()
        self.noises = noises
        self.labels = labels
        self.tf = tf

    def __len__(self):
        return self.noises.shape[0]

    def __getitem__(self, idx):
        x = self.noises[idx]
        x -= x.min(1, keepdim=True)[0]
        x /= x.max(1, keepdim=True)[0]
        x = T.ToPILImage()(x)
        return self.tf(x), self.labels[idx]


def _get_equal_classes_sampler(labels_dist: torch.Tensor) -> WeightedRandomSampler:
    class_weights = 1.0 / torch.bincount(labels_dist)
    sample_weights = class_weights[labels_dist]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


def get_dataloaders(noises, labels, split=(0.85, 0.15), batch_size=32, equal_classes: bool = False):
    dataset = NoiseDataset(noises=noises, labels=labels)
    generator = torch.Generator().manual_seed(42)

    train_ds, test_ds = random_split(dataset, split, generator=generator)
    train_sampler, train_shuffle = (
        (_get_equal_classes_sampler(labels_dist=labels[train_ds.indices]), False) if equal_classes else (None, True)
    )
    training_dataloader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=train_shuffle, drop_last=True, num_workers=7, sampler=train_sampler
    )
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=7)
    return training_dataloader, test_dataloader
