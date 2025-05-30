from pathlib import Path

import torch

from taba.noise_classification.distance_classification import get_noise_sample_by_distance_classification


def img_noise_cls_divs(dir: Path):
    divs = list(dir.glob("class_idx_*"))
    n_divs = len(divs)
    divs_noise_to_img = []
    divs_img_to_noise = []

    for div in range(n_divs):
        noise = torch.stack(torch.load((dir / f"class_idx_{div}" / "latents.pt").resolve(), map_location="cpu"))
        samples = torch.stack(torch.load((dir / f"class_idx_{div}" / "samples.pt").resolve(), map_location="cpu"))

        outs = get_noise_sample_by_distance_classification(
            noises=noise.clone(), samples=samples.clone(), with_plotting=False, examples_limit=4, top_n_fits=3
        )
        divs_img_to_noise.append(outs["closest_noise_to_sample_acc"])
        divs_noise_to_img.append(outs["closest_sample_to_noise_acc"])

    divs_img_to_noise = torch.tensor(divs_img_to_noise)
    divs_noise_to_img = torch.tensor(divs_noise_to_img)

    str_out = f"T={T}\n"
    str_out += f"Reconstruction->latent: ${divs_img_to_noise.mean()*100:.1f}_" + r"{\pm" + f"{ divs_img_to_noise.std()*100:.1f}" + "}$" + "\n"
    str_out += f"Latent->reconstruction: ${divs_noise_to_img.mean()*100:.1f}_" + r"{\pm" + f"{divs_noise_to_img.std()*100:.1f}" + "}$"
    return str_out


if __name__ == "__main__":
    print("Noise to image:")
    for T in [10, 50, 100, 500, 1000]:
        dir = Path(f"experiments/outputs_per_T/imagenet_dit_256/T_{T}")
        print(img_noise_cls_divs(dir))
