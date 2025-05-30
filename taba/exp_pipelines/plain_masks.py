import torch


def find_plain_areas(images, threshold=0.025):
    """
    Identifies plain areas in images where pixel differences are below a threshold.

    Args:
        images (torch.Tensor): Input tensor of shape (N, C, H, W).
        threshold (float): Threshold for detecting plain areas.

    Returns:
        torch.Tensor: Masks indicating plain areas (1 for plain, 0 otherwise).
    """
    # Compute differences along height and width dimensions
    diff_h = torch.abs(images[:, :, 1:, :] - images[:, :, :-1, :])
    diff_w = torch.abs(images[:, :, :, 1:] - images[:, :, :, :-1])

    # Pad differences to match original image dimensions
    diff_h = torch.nn.functional.pad(diff_h, (0, 0, 0, 1))  # Pad bottom row
    diff_w = torch.nn.functional.pad(diff_w, (0, 1, 0, 0))  # Pad right column

    # Combine differences and threshold
    diff_combined = (diff_h + diff_w) / 2
    plain_mask = (diff_combined < threshold).all(dim=1)
    return plain_mask
