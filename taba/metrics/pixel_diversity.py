import io

import PIL.Image as Image
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import cv2
import zlib



def png_bpp(imgs, mask):
    if imgs.min() < 0:
        imgs = (imgs + 1) * 0.5
    imgs = imgs.mul(255).clamp(0, 255).byte()
    plain_bpp, non_bpp = 0.0, 0.0
    for x, m in tqdm(zip(imgs, mask.bool()), total=imgs.shape[0]):
        p = x[:, m].permute(1, 0)
        n = x[:, ~m].permute(1, 0)
        for s, arr in ((p, "plain"), (n, "non")):
            if s.numel():
                im = Image.fromarray(s.reshape(1, -1, 3).cpu().numpy())
                buf = io.BytesIO()
                im.save(buf, format="PNG", compress_level=9)
                bpp = len(buf.getvalue()) * 8 / s.shape[0]
                if arr == "plain":
                    plain_bpp += bpp
                else:
                    non_bpp += bpp
    k = imgs.shape[0]
    return plain_bpp / k, non_bpp / k


def patch_var(imgs, mask):
    imgs = imgs.float()
    N, _, H, W = imgs.shape
    u = F.unfold(imgs, 3, padding=1)
    u = u.view(N, 3, 9, H * W)
    v = (u - u.mean(2, keepdim=True)).pow(2).mean(2).mean(1).view(N, H, W)
    m = mask.bool()
    plains = v[m]
    non_plains = v[~m]
    return {
        "plains": plains,
        "non_plains": non_plains,
        "all": v,
        "plain_mean": plains.mean().item(),
        "non_plain_mean": non_plains.mean().item(),
        "all_mean": v.mean().item(),
    }


def shannon_entropy(image):
    hist = np.histogram(image.ravel(), bins=256, range=[0, 256])[0]
    hist = hist / np.sum(hist)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))


def kolmogorov_complexity(image):
    _, buffer = cv2.imencode(".png", image)
    compressed = zlib.compress(buffer.tobytes())
    return len(compressed)
