from typing import List, Tuple

import matplotlib.pyplot as plt
import torch


def _find_closest_x_for_y_with_distances(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    num_y = y.shape[0]
    closest_indices = torch.empty(num_y, dtype=torch.long)
    all_distances = torch.empty((num_y, len(x)), dtype=torch.float32)
    # for i in tqdm(range(num_y)):
    for i in range(num_y):
        distances = torch.mean((x - y[i]) ** 2, dim=[1, 2, 3])
        closest_indices[i] = torch.argmin(distances)
        all_distances[i] = distances
    return closest_indices, all_distances


def normalize_image(img: torch.Tensor) -> torch.Tensor:
    # normalizes image to values [0,1] for vis
    img = img - img.min()
    img = img / img.max()
    return img


def plot_images(images: List[torch.Tensor], titles: List[str] = None, top_n_fits: int = None):
    num_cols = 2 + top_n_fits if top_n_fits is not None else len(images)
    _, axes = plt.subplots(1, num_cols, figsize=(20, 5))
    for idx, ax in enumerate(axes):
        img = normalize_image(images[idx])
        ax.imshow(img.permute(1, 2, 0))
        if titles is not None:
            ax.title.set_text(titles[idx])
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def get_image_tensor_str_stats(img_tensor: torch.Tensor) -> str:
    assert len(img_tensor.shape) == 3 and img_tensor.shape[0] == 3
    mean, ch1_mean, ch2_mean, ch3_mean = (
        img_tensor.mean().item(),
        img_tensor[0].mean().item(),
        img_tensor[1].mean().item(),
        img_tensor[2].mean().item(),
    )
    var, ch1_var, ch2_var, ch3_var = (
        img_tensor.var().item(),
        img_tensor[0].var().item(),
        img_tensor[1].var().item(),
        img_tensor[2].var().item(),
    )

    return f"""μ: {mean:.2f} ( {ch1_mean:.2f} | {ch2_mean:.2f} | {ch3_mean:.2f} )
σ^2: {var:.2f} ( {ch1_var:.2f} | {ch2_var:.2f} | {ch3_var:.2f} )
"""


def get_closest_and_plot(
    noises,
    samples,
    closest_indices,
    all_distances,
    examples_limit: int,
    top_n_fits: int,
    is_sample_to_noise: bool = False,
):
    (src_name, target_name) = ("sample", "noise") if is_sample_to_noise else ("noise", "sample")
    exp_mode = f"{src_name}->{target_name}"
    source, target = (samples, noises) if is_sample_to_noise else (noises, samples)

    examples = 0
    for idx in range(len(closest_indices)):
        if examples >= examples_limit:
            # plotting reached limit
            break
        elif closest_indices[idx] == idx:
            # correct prediction, no need to plot
            continue
        else:
            # mistake
            examples += 1

        sorted_indices = torch.argsort(all_distances[idx])
        top_matches = sorted_indices[:top_n_fits]
        best_fitting_samples = samples[top_matches]
        best_fitting_noises = noises[top_matches]

        images = [source[idx], target[idx]] + list(best_fitting_samples)
        tensors_calc_valmeans = (
            images if not is_sample_to_noise else [source[idx], target[idx]] + list(best_fitting_noises)
        )

        img_valmeans = list(map(get_image_tensor_str_stats, tensors_calc_valmeans))
        titles = [f"Source {src_name}", f"Target {target_name}"] + [
            f"#{j+1} Best Fit {target_name}\n" for j in range(top_n_fits)
        ]
        titles = [f"{title}\n{valmean}" for title, valmean in zip(titles, img_valmeans)]

        print(f"Mistaken #{examples} {exp_mode}")
        plot_images(images, titles, top_n_fits)


def plot_most_attractive(samples: torch.Tensor, distances: torch.Tensor, plot_limit: int):
    closest_sample_indices = torch.argmin(distances, dim=1)
    unique_indices, counts = torch.unique(closest_sample_indices, return_counts=True)

    _, sorted_indices = torch.sort(counts, descending=True)
    counts = counts[sorted_indices]
    most_attractive_indices = unique_indices[sorted_indices]

    images, titles = [], []

    for idx in range(min(plot_limit, len(most_attractive_indices))):
        sample_idx = most_attractive_indices[idx]
        images.append(samples[sample_idx])
        titles.append(f"CNT: {counts[idx]}")

    _, axes = plt.subplots(1, len(images), figsize=(20, 5))
    for idx, ax in enumerate(axes):
        img = normalize_image(images[idx])
        ax.imshow(img.permute(1, 2, 0))
        ax.title.set_text(titles[idx])
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def get_noise_sample_by_distance_classification(
    noises: torch.Tensor,
    samples: torch.Tensor,
    examples_limit: int = 10,
    top_n_fits: int = 3,
    with_plotting: bool = True,
    top_k: int = 1,
):
    closest_noise_idx_for_samples, noises_distances_for_samples = _find_closest_x_for_y_with_distances(
        x=noises.to(samples.device), y=samples
    )
    closest_sample_idx_for_noises, samples_distances_for_noises = _find_closest_x_for_y_with_distances(
        x=samples, y=noises.to(samples.device)
    )

    top_k_closest_noise_idx_for_samples = torch.topk(
        noises_distances_for_samples, k=top_k, dim=1, largest=False
    ).indices
    top_k_closest_sample_idx_for_noises = torch.topk(
        samples_distances_for_noises, k=top_k, dim=1, largest=False
    ).indices

    closest_noise_to_sample_acc = sum(
        [idx in top_k_closest_noise_idx_for_samples[idx] for idx in torch.arange(len(closest_noise_idx_for_samples))]
    ) / len(closest_noise_idx_for_samples)
    closest_sample_to_noise_acc = sum(
        [idx in top_k_closest_sample_idx_for_noises[idx] for idx in torch.arange(len(closest_sample_idx_for_noises))]
    ) / len(closest_sample_idx_for_noises)

    if with_plotting:
        print(f"Closest noise to sample accuracy: {closest_noise_to_sample_acc}")
        print(f"Closest sample to noise accuracy: {closest_sample_to_noise_acc}")
        get_closest_and_plot(
            noises=noises.cpu(),
            samples=samples.cpu(),
            closest_indices=closest_noise_idx_for_samples,
            all_distances=noises_distances_for_samples,
            is_sample_to_noise=True,
            examples_limit=examples_limit,
            top_n_fits=top_n_fits,
        )
        get_closest_and_plot(
            noises=noises.cpu(),
            samples=samples.cpu(),
            closest_indices=closest_sample_idx_for_noises,
            all_distances=samples_distances_for_noises,
            is_sample_to_noise=False,
            examples_limit=examples_limit,
            top_n_fits=top_n_fits,
        )
        print("Most attracting noises")
        plot_most_attractive(samples=samples.cpu(), distances=noises_distances_for_samples, plot_limit=8)
        print("Most attracting samples")
        plot_most_attractive(samples=samples.cpu(), distances=samples_distances_for_noises, plot_limit=8)

    return {
        "closest_noise_to_sample_acc": closest_noise_to_sample_acc,
        "closest_sample_to_noise_acc": closest_sample_to_noise_acc,
    }
