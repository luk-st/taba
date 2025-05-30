import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from sklearn.cross_decomposition import CCA


def calculate_psnr(img1, img2):
    return cv2.PSNR(img1, img2)


def calculate_ssim(img1, img2):
    ssim_values = []
    for i in range(3):  # For each color channel
        ssim_value = ssim(img1[:, :, i], img2[:, :, i], data_range=1)
        ssim_values.append(ssim_value)
    return np.mean(ssim_values)


def hsic_unbiased(K, L):
    m = K.shape[0]
    K_tilde = K.clone().fill_diagonal_(0)
    L_tilde = L.clone().fill_diagonal_(0)

    HSIC_value = (
        (torch.sum(K_tilde * L_tilde.T))
        + (torch.sum(K_tilde) * torch.sum(L_tilde) / ((m - 1) * (m - 2)))
        - (2 * torch.sum(torch.mm(K_tilde, L_tilde)) / (m - 2))
    )

    HSIC_value /= m * (m - 3)
    return HSIC_value


def hsic_biased(K, L):
    """Compute the biased HSIC (the original CKA)"""
    H = torch.eye(K.shape[0], dtype=K.dtype, device=K.device) - 1 / K.shape[0]
    return torch.trace(K @ H @ L @ H)


def cka(feats_A, feats_B, kernel_metric="ip", rbf_sigma=1.0, unbiased=False):
    if kernel_metric == "ip":
        K = torch.mm(feats_A, feats_A.T)
        L = torch.mm(feats_B, feats_B.T)
    elif kernel_metric == "rbf":
        K = torch.exp(-torch.cdist(feats_A, feats_A) ** 2 / (2 * rbf_sigma**2))
        L = torch.exp(-torch.cdist(feats_B, feats_B) ** 2 / (2 * rbf_sigma**2))
    else:
        raise ValueError(f"Invalid kernel metric {kernel_metric}")

    hsic_fn = hsic_unbiased if unbiased else hsic_biased
    hsic_kk = hsic_fn(K, K)
    hsic_ll = hsic_fn(L, L)
    hsic_kl = hsic_fn(K, L)

    cka_value = hsic_kl / (torch.sqrt(hsic_kk * hsic_ll) + 1e-6)
    return cka_value


def svcca(feats_A: torch.Tensor, feats_B: torch.Tensor, cca_dim: int = 10):
    def preprocess_activations(act: torch.Tensor):
        act = act - act.mean(dim=0)
        act = act / (act.std(dim=0) + 1e-8)
        return act

    feats_A = preprocess_activations(feats_A)
    feats_B = preprocess_activations(feats_B)

    U1, _, _ = torch.svd_lowrank(feats_A, q=cca_dim)
    U2, _, _ = torch.svd_lowrank(feats_B, q=cca_dim)
    U1 = U1.cpu().detach().numpy()
    U2 = U2.cpu().detach().numpy()

    cca = CCA(n_components=cca_dim)
    cca.fit(U1, U2)
    U1_c, U2_c = cca.transform(U1, U2)

    U1_c += 1e-10 * np.random.randn(*U1_c.shape)
    U2_c += 1e-10 * np.random.randn(*U2_c.shape)

    svcca_similarity = np.mean([np.corrcoef(U1_c[:, i], U2_c[:, i])[0, 1] for i in range(cca_dim)])
    return svcca_similarity
