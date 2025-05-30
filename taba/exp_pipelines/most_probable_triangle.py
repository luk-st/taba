import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms


def tensors_to_pils(tensors):
    tensors = tensors - tensors.min()
    tensors = tensors / tensors.max()
    return tensors


def calculate_angle(v1, v2):
    cos_theta = torch.dot(v1, v2) / (v1.norm() * v2.norm())

    # Clip the cos_theta value to avoid numerical issues with arccos
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    # Compute the angle in radians
    angle = torch.rad2deg(torch.acos(cos_theta))
    return angle


def find_max_prob_triangle_angles(hist_A, hist_B, hist_C, angle_bins):
    """
    Finds the triangle angles that maximize the joint probability, given histograms.

    hist_A, hist_B, hist_C: Tensors of probabilities for each angle.
    angle_bins: Tensor of angle values corresponding to histogram bins.
    """
    max_prob = 0.0
    best_angles = None

    # Iterate over all combinations of angles A, B, C that satisfy A + B + C = 180
    for i, A in enumerate(angle_bins):
        for j, B in enumerate(angle_bins):
            C = 180 - A - B  # C must satisfy the triangle constraint
            if C in angle_bins:  # Ensure C is a valid bin value
                k = (angle_bins == C).nonzero(as_tuple=True)[0].item()  # Find index of C in bins

                # Get the probabilities from the histograms
                prob_A = hist_A[i]
                prob_B = hist_B[j]
                prob_C = hist_C[k]

                # Compute joint probability
                joint_prob = prob_A * prob_B * prob_C

                # Track the maximum probability and corresponding angles
                if joint_prob > max_prob:
                    max_prob = joint_prob
                    best_angles = (A.item(), B.item(), C.item())

    return best_angles, max_prob


def main(noise_path, samples_path, latents_path):
    assert os.path.exists(noise_path)
    assert os.path.exists(samples_path)
    assert os.path.exists(latents_path)
    noise = torch.load(noise_path, weights_only=False)
    samples = torch.load(samples_path, weights_only=False)
    latents = torch.load(latents_path, weights_only=False)
    angles_s = []
    angles_n = []
    angles_l = []
    for i in range(noise.shape[0]):
        ns = noise[i].flatten() - samples[i].flatten()
        ls = latents[i].flatten() - samples[i].flatten()
        nl = noise[i].flatten() - latents[i].flatten()
        angles_s.append(calculate_angle(ns, ls))
        angles_l.append(calculate_angle(nl, -ls))
        angles_n.append(calculate_angle(-nl, -ns))

    hist_l, bins = np.histogram(angles_l, bins=range(1, 180))
    hist_s, bins = np.histogram(angles_s, bins=range(1, 180))
    hist_n, bins = np.histogram(angles_n, bins=range(1, 180))
    (angle_l, angle_s, angle_n), _ = find_max_prob_triangle_angles(
        hist_l / hist_l.sum(), hist_s / hist_s.sum(), hist_n / hist_n.sum(), torch.from_numpy(bins)
    )

    print(angle_l, angle_s, angle_n)


if __name__ == "__main__":
    main(
        noise_path="<path_to_noise>",
        samples_path="<path_to_samples>",
        latents_path="<path_to_latents>",
    )
