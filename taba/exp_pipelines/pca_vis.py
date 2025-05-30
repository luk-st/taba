import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA


def plot_pca_steps(all_t_samples, all_t_latents, T, num_components=2, example_idx=0, path=None):
    num_runs = all_t_samples.shape[1]
    all_samples_flat = []
    all_latents_flat = []
    all_full_paths_flat = []

    # Flatten all the samples and latents
    for s_idx in range(num_runs):
        noise = all_t_samples[:1, s_idx, :, :, :]
        sample = all_t_samples[T:, s_idx, :, :, :]
        latent = all_t_latents[T:, s_idx, :, :, :]
        t_samples = all_t_samples[1:T, s_idx, :, :, :]
        t_latents = all_t_latents[1:T, s_idx, :, :, :]
        full_path = torch.cat([noise, t_samples, sample, t_latents, latent])

        t_samples_flat = t_samples.view(t_samples.shape[0], -1).cpu().numpy()
        t_latents_flat = t_latents.view(t_latents.shape[0], -1).cpu().numpy()
        full_path_flat = full_path.view(full_path.shape[0], -1).cpu().numpy()

        all_samples_flat.append(t_samples_flat)
        all_latents_flat.append(t_latents_flat)
        all_full_paths_flat.append(full_path_flat)

    all_samples_flat = np.array(all_samples_flat)
    all_latents_flat = np.array(all_latents_flat)
    all_full_paths_flat = np.array(all_full_paths_flat)

    # Fit PCA on all examples for each step
    num_steps = all_full_paths_flat.shape[1]
    pca_models = []
    transformed_full_paths = []

    for step_idx in range(num_steps):
        # Collect all examples at this step across all paths
        step_data = all_full_paths_flat[:, step_idx, :]

        # Fit PCA on all paths for this step
        pca = PCA(n_components=num_components)
        pca.fit(step_data)
        pca_models.append(pca)

        # Transform the step data for the example path
        example_step_data = all_full_paths_flat[example_idx, step_idx].reshape(1, -1)
        transformed_step = pca.transform(example_step_data)
        transformed_full_paths.append(transformed_step)

    transformed_full_paths = np.array(transformed_full_paths).squeeze()

    # Visualization of the selected example path after PCA transformation
    fig, ax = plt.subplots(figsize=(10, 6))

    # Visualize the trajectory of the example path
    ax.plot(transformed_full_paths[:, 0], transformed_full_paths[:, 1], marker="o", label="Example Path")

    ax.set_title("PCA Transformed Example Diffusion Path (Fitted on All Examples)")
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    plt.legend()
    if path is None:
        plt.show()
    else:
        plt.savefig(path)

    # mean_samples_flat = np.mean(all_samples_flat, axis=0)
    # mean_latents_flat = np.mean(all_latents_flat, axis=0)
    # mean_full_path_flat = np.mean(all_full_paths_flat, axis=0)

    # pca = PCA(n_components=2)
    # pca.fit(mean_full_path_flat)
    # mean_samples_2d = pca.transform(mean_samples_flat)
    # mean_latents_2d = pca.transform(mean_latents_flat)

    # # Plotting the averaged forward path
    # plt.figure(figsize=(12, 8))
    # plt.plot(mean_samples_2d[:, 0], mean_samples_2d[:, 1], color="blue", label="Averaged Forward Path", alpha=0.7)
    # plt.plot(mean_latents_2d[:, 0], mean_latents_2d[:, 1], color="red", label="Averaged Reverse Path", alpha=0.7)

    # # Adding arrows for the average paths
    # for i in range(mean_samples_2d.shape[0] - 1):
    #     plt.arrow(
    #         mean_samples_2d[i, 0],
    #         mean_samples_2d[i, 1],
    #         mean_samples_2d[i + 1, 0] - mean_samples_2d[i, 0],
    #         mean_samples_2d[i + 1, 1] - mean_samples_2d[i, 1],
    #         head_width=0.02,
    #         head_length=0.02,
    #         fc="blue",
    #         ec="blue",
    #         alpha=0.7,
    #     )

    # for i in range(mean_latents_2d.shape[0] - 1):
    #     plt.arrow(
    #         mean_latents_2d[i, 0],
    #         mean_latents_2d[i, 1],
    #         mean_latents_2d[i + 1, 0] - mean_latents_2d[i, 0],
    #         mean_latents_2d[i + 1, 1] - mean_latents_2d[i, 1],
    #         head_width=0.02,
    #         head_length=0.02,
    #         fc="red",
    #         ec="red",
    #         alpha=0.7,
    #     )
    # plt.scatter(mean_samples_2d[:1, 0], mean_samples_2d[:1, 1], color="darkblue", label="Noise", s=200)
    # plt.scatter(mean_samples_2d[-1:, 0], mean_samples_2d[-1:, 1], color="darkviolet", label="Sample", s=200)
    # plt.scatter(mean_latents_2d[-1:, 0], mean_latents_2d[-1:, 1], color="darkred", label="Latent", s=200)
    # plt.title("Averaged Forward and Reverse DDIM Path Visualization", fontsize=25)
    # plt.xlabel("PCA-1", fontsize=20)
    # plt.ylabel("PCA-2", fontsize=20)
    # plt.legend(fontsize=20)
    # plt.grid(True)
    # if path is None:
    #     plt.show()
    # else:
    #     plt.savefig(path)


if __name__ == "__main__":
    # ddpm cifar10
    cifar_t_samples = torch.load(
        "experiments/interpolate_diffusion/cifar/all_t_samples_ddpm-cifar10-32.pt", weights_only=False
    )
    cifar_t_latents = torch.load(
        "experiments/interpolate_diffusion/cifar/all_t_latents_ddpm-cifar10-32.pt", weights_only=False
    )
    plot_pca_steps(all_t_samples=cifar_t_samples, all_t_latents=cifar_t_latents, T=100, path="cifar_pca.png")

    # # ddpm imgnet
    # imgnet_t_latents = torch.load(
    #     "experiments/interpolate_diffusion/imgnet/all_t_latents_ddpm-imagenet-64.pt", weights_only=False
    # )
    # imgnet_t_samples = torch.load(
    #     "experiments/interpolate_diffusion/imgnet/all_t_samples_ddpm-imagenet-64.pt", weights_only=False
    # )
    # plot_pca_steps(all_t_samples=imgnet_t_samples, all_t_latents=imgnet_t_latents, T=100, path="imgnet_pca.png")

    # # ldm celeba
    # ldm_t_latents = torch.load(
    #     "experiments/interpolate_diffusion/celeba/all_t_latents_ldm-celeba-64.pt", weights_only=False
    # )
    # ldm_t_samples = torch.load(
    #     "experiments/interpolate_diffusion/celeba/all_t_samples_ldm-celeba-64.pt", weights_only=False
    # )
    # plot_pca_steps(all_t_samples=ldm_t_samples, all_t_latents=ldm_t_latents, T=100, path="celeba_pca.png")
