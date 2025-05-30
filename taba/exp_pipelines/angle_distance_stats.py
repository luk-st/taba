import argparse
import os

import torch

from taba.metrics.angles_distances import calc_angles, calc_distances

def run_angles_distances(noise, samples, latents, T):
    angles = calc_angles(noise, samples, latents)
    dists = calc_distances(noise, samples, latents)
    str_out = f"${T}$"
    str_out += f" & ${angles['mean_cossim_img']:.2f}$"
    str_out += f" & ${angles['mean_cossim_noise']:.2f}$"
    str_out += f" & ${angles['mean_cossim_latent']:.2f}$"
    str_out += f" & ${dists['mean_l2_noise_lat']:.2f}_" + "{\pm" + f"{dists['std_l2_noise_lat']:.2f}" + "}$"
    str_out += f" & ${dists['mean_l2_img_lat']:.2f}_" + "{\pm" + f"{dists['std_l2_img_lat']:.2f}" + "}$"
    str_out += f" & ${dists['mean_l2_img_noise']:.2f}_" + "{\pm" + f"{dists['std_l2_img_noise']:.2f}" + "}$"
    str_out += " \\\\"
    print(str_out)

def run_metrics(path):
    latents = torch.load(os.path.join(path, "latents.pt")).cuda()
    noise = torch.load(os.path.join(path, "noise.pt")).cuda()
    samples = torch.load(os.path.join(path, "samples.pt")).cuda()

    T = path.split("T_")[-1]

    run_angles_distances(noise, samples, latents, T)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the experiment directory, example: experiments/outputs_per_T/ldm_outs/T_1000",
    )
    args = parser.parse_args()
    run_metrics(args.path)
