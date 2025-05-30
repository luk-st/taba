from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.utils import make_grid
from tqdm import tqdm
from transformers import PreTrainedModel, ViTForImageClassification, ViTImageProcessor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
@dataclass
class Annotator:
    processor: Any
    model: PreTrainedModel


class GenerationsDataset(Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return T.ToPILImage()(make_grid(self.x[idx], nrow=1, normalize=True))


def get_vit_cifar10_annotator(device=DEVICE):
    processor = ViTImageProcessor.from_pretrained("nateraw/vit-base-patch16-224-cifar10")
    model = ViTForImageClassification.from_pretrained("nateraw/vit-base-patch16-224-cifar10").to(device)
    return Annotator(processor=processor, model=model)


def get_vit_imagenet_annotator(device=DEVICE):
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224").to(device)
    return Annotator(processor=processor, model=model)


def annotate(dataset, n_samples, batch_size, annotator: Annotator):
    labels = []
    with torch.no_grad():
        for idx_start in tqdm(range(0, n_samples, batch_size)):
            idx_end = idx_start + batch_size
            dat_in = [dataset[idx] for idx in range(idx_start, idx_end)]
            inputs = annotator.processor(images=dat_in, return_tensors="pt")
            outputs = annotator.model(**inputs)
            logits = outputs.logits
            softmax_logits = F.softmax(logits, dim=1)
            _, max_index = torch.max(softmax_logits, dim=1)
            labels.append(max_index)
    return labels


def annotate_dl(dataloader, annotator: Annotator):
    labels = []
    with torch.no_grad():
        for samples in tqdm(dataloader):
            inputs = annotator.processor(images=samples, return_tensors="pt")
            outputs = annotator.model(**inputs)
            logits = outputs.logits
            softmax_logits = F.softmax(logits, dim=1)
            _, max_index = torch.max(softmax_logits, dim=1)
            labels.append(max_index)
    return labels
