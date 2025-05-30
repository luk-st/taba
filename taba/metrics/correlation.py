import torch


def get_top_k_corr(tensor: torch.Tensor, top_k: int = 10) -> dict:
    assert tensor.ndim == 4, "Tensor must have 4 dimensions: [batch_size, channels, height, width]"
    corr_matrix = torch.corrcoef(tensor.flatten(1, 3).T)
    corr_matrix = corr_matrix.triu(diagonal=1)
    top_k_coeffs = torch.topk(corr_matrix.abs().flatten(), top_k)
    top_k_values = top_k_coeffs[0]

    return {"mean": top_k_values.mean().item(), "std": top_k_values.std().item()}


def get_top_k_corr_in_patches(tensor: torch.Tensor, patch_size: int = 8, top_k: int = 10) -> dict:
    assert tensor.ndim == 4, "Tensor must have 4 dimensions: [batch_size, channels, height, width]"
    num_examples, channels, height, width = tensor.shape

    avg_topk_values = []
    patch_counter = 0

    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            patches = tensor[:, :, i : i + patch_size, j : j + patch_size].reshape(num_examples, -1)

            if patches.size(1) > 1:  # Ensure there are at least two elements
                mean = patches.mean(dim=0, keepdim=True)
                std = patches.std(dim=0, keepdim=True)
                normalized_patches = (patches - mean) / (std + 1e-5)

                corr_matrix = torch.corrcoef(normalized_patches.T)

                # Extract the upper triangle of the correlation matrix, excluding the diagonal
                triu_indices = torch.triu_indices(patches.size(1), patches.size(1), offset=1)
                upper_triangle_values = corr_matrix[triu_indices[0], triu_indices[1]]

                # Get the top-k absolute correlation coefficients for this patch
                top_k_values, _ = torch.topk(upper_triangle_values.abs(), min(top_k, upper_triangle_values.numel()))
                avg_topk_values.append(top_k_values)
            else:
                patch_counter += 1

    if avg_topk_values:
        avg_topk_values = torch.cat(avg_topk_values, dim=0)
        mean_top_k = avg_topk_values.mean().item()
        std_top_k = avg_topk_values.std().item()
    else:
        mean_top_k = 0
        std_top_k = 0

    if patch_counter > 0:
        print(f"Warning: {patch_counter} patches were empty")

    return {"mean": mean_top_k, "std": std_top_k}


if __name__ == "__main__":
    for idx, model in enumerate(["cifar_pixel_32", "imagenet_pixel_64", "imagenet_pixel_256", "celeba_ldm_256", "imagenet_dit_256"]):
        print("Model name: ", model)
        noisess = torch.load(f"experiments/outputs_correlation/{model}/T_100/noise.pt", weights_only=False).to(torch.float16)
        latents = torch.load(f"experiments/outputs_correlation/{model}/T_100/latents.pt", weights_only=False).to(torch.float16)
        samples = torch.load(f"experiments/outputs_correlation/{model}/T_100/samples.pt", weights_only=False).to(torch.float16)
        print(f"Dtypes: {noisess.dtype=}, {latents.dtype=}, {samples.dtype=}")
        print(f"Shapes: {noisess.shape=}, {latents.shape=}, {samples.shape=}")
        print("Noisess: ", get_top_k_corr_in_patches(noisess, top_k=20, patch_size=8))
        print("Latents: ", get_top_k_corr_in_patches(latents, top_k=20, patch_size=8))
        print("Samples: ", get_top_k_corr_in_patches(samples, top_k=20, patch_size=8))
