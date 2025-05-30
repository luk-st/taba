import torch

from taba.exp_pipelines.noise_classification.distance_classification import (
    get_noise_sample_by_distance_classification,
)

# from taba.utils import plot_diffusion


def convert(tensor: torch.Tensor) -> torch.Tensor:
    tensor2 = tensor - tensor.min()
    tensor2 = tensor2 / tensor2.max()
    return tensor2


def img_noise_cls_divs(noise, samples, divs: int = 8, with_plotting: bool = False):
    divs_noise_to_img = []
    divs_img_to_noise = []

    for div in range(divs):
        noise_div = noise[div * (len(noise) // divs) : (div + 1) * (len(noise) // divs)].cuda()
        samples_div = samples[div * (len(samples) // divs) : (div + 1) * (len(samples) // divs)].cuda()
        outs = get_noise_sample_by_distance_classification(
            noises=noise_div.clone(),
            samples=samples_div.clone(),
            with_plotting=with_plotting,
            examples_limit=4,
            top_n_fits=3,
        )
        divs_img_to_noise.append(outs["closest_noise_to_sample_acc"])
        divs_noise_to_img.append(outs["closest_sample_to_noise_acc"])

    divs_img_to_noise = torch.tensor(divs_img_to_noise)
    divs_noise_to_img = torch.tensor(divs_noise_to_img)

    str_out = ""
    str_out += f"${divs_img_to_noise.mean()*100:.1f}_" + "{\pm" + f"{ divs_img_to_noise.std()*100:.1f}" + "}$"
    str_out += f" & ${divs_noise_to_img.mean()*100:.1f}_" + "{\pm" + f"{divs_noise_to_img.std()*100:.1f}" + "}$"
    return str_out


if __name__ == "__main__":
    samples = torch.load("experiments/angles/openai_imagenet_samples.pth")
    vars_samples = [(sample.var(), sample) for sample in samples.cpu()]
    vars_samples2 = sorted(vars_samples, key=lambda x: x[0])
    samples_3 = [convert(x[1]) for x in vars_samples2[:10]]
    # plot_diffusion(samples_3)

    samples = torch.load("experiments/angles/openai_cifar10_samples.pth")
    vars_samples = [(sample.var(), sample) for sample in samples.cpu()]
    vars_samples2 = sorted(vars_samples, key=lambda x: x[0])
    samples_3 = [convert(x[1]) for x in vars_samples2[:10]]
    # plot_diffusion(samples_3)

    noise = torch.load("experiments/angles/openai_cifar10_noise.pth")
    samples = torch.load("experiments/angles/openai_cifar10_samples.pth")
    get_noise_sample_by_distance_classification(
        noises=noise.clone(), samples=samples.clone(), examples_limit=10, top_n_fits=4
    )
