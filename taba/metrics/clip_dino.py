from typing import List

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModel, AutoProcessor

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
DINO_MODEL_NAME = "facebook/dino-vits16"
BS = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_clip(device=DEVICE):
    clip_processor = AutoProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip_model = AutoModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    return clip_processor, clip_model


def get_dino(device=DEVICE):
    dino_model = AutoModel.from_pretrained(DINO_MODEL_NAME, add_pooling_layer=False).to(device)
    return dino_model


def get_clip_features(imgs: List[Image.Image], clip_processor, clip_model, device=DEVICE):
    outs = []
    for batch_ids in range(0, len(imgs), BS):
        batch = imgs[batch_ids : batch_ids + BS]
        clip_batch_in = clip_processor(images=batch, return_tensors="pt").pixel_values.to(device)
        feats = clip_model.get_image_features(clip_batch_in)
        outs.append(feats.detach().cpu())
    return torch.cat(outs)


def get_dino_features(imgs: List[Image.Image], dino_model, device=DEVICE):
    T = transforms.Compose(
        [
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    outs = []
    for batch_ids in range(0, len(imgs), BS):
        batch = imgs[batch_ids : batch_ids + BS]
        pred_imgs_processed: torch.Tensor = torch.stack([T(img) for img in batch])
        pred_imgs_processed = pred_imgs_processed.to(device)
        pred_features = dino_model(pred_imgs_processed).last_hidden_state[:, 0, :]
        outs.append(pred_features.detach().cpu())
    return torch.cat(outs)


def get_mean_cosine_sim(vec1, vec2):
    vec1 = vec1.view(vec1.shape[0], -1)
    vec2 = vec2.view(vec2.shape[0], -1)
    vec1 = torch.nn.functional.normalize(vec1, dim=1)
    vec2 = torch.nn.functional.normalize(vec2, dim=1)
    return torch.sum(vec1 * vec2, dim=1).mean()
