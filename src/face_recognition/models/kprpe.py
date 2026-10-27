import math

import torch
import torch.nn as nn


def scale_landmarks(
    landmarks: torch.Tensor,
    original_size: int = 250,
    target_size: int = 112,
) -> torch.Tensor:
    """
    Scale facial landmark coordinates to the ViT input size.

    Args:
        landmarks: Tensor of shape [B, 5, 2].
        original_size: Original square image size.
        target_size: Resized square image size.

    Returns:
        Scaled landmarks of shape [B, 5, 2].
    """
    if landmarks.ndim != 3 or landmarks.shape[1:] != (5, 2):
        raise ValueError("Expected landmarks shape [B, 5, 2]")

    if not torch.is_floating_point(landmarks):
        raise TypeError("Landmarks must be floating-point tensors")

    if original_size <= 0 or target_size <= 0:
        raise ValueError("Image sizes must be positive")

    return landmarks * (target_size / original_size)

def normalize_landmarks(
    landmarks: torch.Tensor,
    image_size: int = 112,
) -> torch.Tensor:
    """
    Normalize landmark coordinates into [0, 1].

    Args:
        landmarks: Tensor [B, 5, 2] in image pixel coordinates.
        image_size: Image width and height.

    Returns:
        Normalized landmarks [B, 5, 2].
    """
    if landmarks.ndim != 3 or landmarks.shape[1:] != (5, 2):
        raise ValueError("Expected landmarks shape [B, 5, 2]")

    if image_size <= 0:
        raise ValueError("image_size must be positive")

    if not torch.is_floating_point(landmarks):
        raise TypeError("Landmarks must be floating-point tensors")

    if not torch.isfinite(landmarks).all():
        raise ValueError("Landmarks contain NaN or Inf")

    normalized = landmarks / image_size

    if ((normalized < 0) | (normalized > 1)).any():
        raise ValueError("Landmarks outside image bounds")

    return normalized

def get_patch_centers(
    image_size: int = 112,
    patch_size: int = 8,
    device=None,
) -> torch.Tensor:
    """
    Return normalized patch-center coordinates.

    Returns:
        Tensor [N, 2], where N = (image_size // patch_size) ** 2.
        Coordinate order: (x, y).
    """
    if image_size <= 0 or patch_size <= 0:
        raise ValueError("Image and patch sizes must be positive")

    if image_size % patch_size != 0:
        raise ValueError("image_size must be divisible by patch_size")

    grid_size = image_size // patch_size

    coords = (
        torch.arange(grid_size, device=device, dtype=torch.float32)
        + 0.5
    ) / grid_size

    grid_y, grid_x = torch.meshgrid(
        coords, coords, indexing="ij"
    )

    centers = torch.stack(
        [grid_x, grid_y], dim=-1
    )

    return centers.reshape(-1, 2)

def compute_relative_positions(
    landmarks: torch.Tensor,
    patch_centers: torch.Tensor,
) -> torch.Tensor:
    """
    Compute patch-to-landmark relative positions.

    Args:
        landmarks: Normalized coordinates [B, 5, 2].
        patch_centers: Normalized coordinates [N, 2].

    Returns:
        Relative positions [B, N, 5, 2].
    """
    if landmarks.ndim != 3 or landmarks.shape[1:] != (5, 2):
        raise ValueError("Expected landmarks shape [B, 5, 2]")

    if patch_centers.ndim != 2 or patch_centers.shape[1] != 2:
        raise ValueError("Expected patch_centers shape [N, 2]")

    if landmarks.device != patch_centers.device:
        raise ValueError("Landmarks and patches must use the same device")

    if not torch.isfinite(landmarks).all():
        raise ValueError("Invalid landmark coordinates")

    if not torch.isfinite(patch_centers).all():
        raise ValueError("Invalid patch center coordinates")

    relative_positions = (
        patch_centers[None, :, None, :]
        - landmarks[:, None, :, :]
    )

    return relative_positions

def flatten_relative_positions(
    relative_positions: torch.Tensor,
) -> torch.Tensor:
    """
    Flatten patch-to-landmark relative positions.

    Args:
        relative_positions: Tensor [B, N, 5, 2].

    Returns:
        Tensor [B, N, 10].
    """
    if relative_positions.ndim != 4:
        raise ValueError("Expected a 4D tensor")

    if relative_positions.shape[2:] != (5, 2):
        raise ValueError("Expected shape [B, N, 5, 2]")

    return relative_positions.flatten(start_dim=2)

class KeypointProjection(nn.Module):
    def __init__(self, num_keypoints=5, num_buckets=49):
        super().__init__()

        self.linear = nn.Linear(
            2 * num_keypoints,
            num_buckets,
        )

        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, relative_positions):
        # Input: [B, N, 10]
        # Output: [B, 1, N, num_buckets]
        return self.linear(relative_positions).unsqueeze(1)

@torch.no_grad()
def piecewise_index(
    relative_position: torch.Tensor,
    alpha: float = 1.9,
    beta: float = 3.8,
    gamma: float = 15.2,
) -> torch.Tensor:
    """
    Map relative patch distances to discrete indices.

    Adapted from the KP-RPE reference implementation.

    Returns:
        Integer indices in [-int(beta), int(beta)].
    """
    if not (0 < alpha < beta < gamma):
        raise ValueError("Expected 0 < alpha < beta < gamma")

    distance = relative_position.float()
    absolute = distance.abs()

    result = distance.round()

    mask = absolute > alpha

    if mask.any():
        outside = distance[mask]
        magnitude = absolute[mask]

        mapped = (
            alpha
            + torch.log(magnitude / alpha)
            / math.log(gamma / alpha)
            * (beta - alpha)
        )

        result[mask] = (
            torch.sign(outside) * mapped
        ).round()

    # Keep indices within the declared bucket range.
    beta_int = int(beta)

    return result.clamp(
        min=-beta_int,
        max=beta_int,
    ).long()

@torch.no_grad()
def get_relative_position_buckets(
    image_size: int = 112,
    patch_size: int = 8,
    alpha: float = 1.9,
    beta: float = 3.8,
    gamma: float = 15.2,
    device=None,
) -> torch.Tensor:
    """
    Build 2D relative-position bucket indices.

    Returns:
        Tensor [N, N] with integer bucket indices,
        where N = (image_size // patch_size) ** 2.
    """
    if image_size <= 0 or patch_size <= 0:
        raise ValueError("Image and patch sizes must be positive")

    if image_size % patch_size != 0:
        raise ValueError("image_size must be divisible by patch_size")

    grid_size = image_size // patch_size

    rows, cols = torch.meshgrid(
        torch.arange(grid_size, device=device),
        torch.arange(grid_size, device=device),
        indexing="ij",
    )

    # Patch order matches PatchEmbedding flatten order.
    positions = torch.stack(
        [rows.flatten(), cols.flatten()],
        dim=-1,
    )

    # Relative offsets between query and key patches.
    diff = positions[:, None, :] - positions[None, :, :]

    beta_int = int(beta)
    side = 2 * beta_int + 1

    row_index = piecewise_index(
        diff[..., 0],
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    ) + beta_int

    col_index = piecewise_index(
        diff[..., 1],
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    ) + beta_int

    buckets = row_index * side + col_index

    return buckets.long()

def build_attention_bias(
    projected_keypoints: torch.Tensor,
    bucket_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Gather keypoint-relative positional values into attention bias.

    Args:
        projected_keypoints: [B, 1, N, K]
        bucket_indices: [N, N], containing indices in [0, K-1]

    Returns:
        Attention bias: [B, 1, N, N]
    """
    if projected_keypoints.ndim != 4:
        raise ValueError("Expected projected_keypoints [B, 1, N, K]")

    B, H, N, K = projected_keypoints.shape

    if H != 1:
        raise ValueError("Expected shared positional bias with H=1")

    if bucket_indices.shape != (N, N):
        raise ValueError("Expected bucket_indices shape [N, N]")

    if bucket_indices.device != projected_keypoints.device:
        raise ValueError("Tensors must be on the same device")

    if bucket_indices.dtype != torch.long:
        raise TypeError("Bucket indices must be torch.long")

    if bucket_indices.min() < 0 or bucket_indices.max() >= K:
        raise ValueError("Bucket index outside valid range")

    indices = bucket_indices[None, None, :, :].expand(
        B, 1, N, N
    )

    return torch.gather(
        projected_keypoints,
        dim=-1,
        index=indices,
    )

class KPRPE(nn.Module):
    def __init__(
        self,
        image_size=112,
        patch_size=8,
        num_keypoints=5,
        num_buckets=49,
    ):
        super().__init__()

        self.image_size = image_size
        self.num_patches = (image_size // patch_size) ** 2

        self.projection = KeypointProjection(
            num_keypoints=num_keypoints,
            num_buckets=num_buckets,
        )

        self.register_buffer(
            "patch_centers",
            get_patch_centers(image_size, patch_size),
            persistent=False,
        )

        self.register_buffer(
            "bucket_indices",
            get_relative_position_buckets(
                image_size, patch_size
            ),
            persistent=False,
        )

    def forward(self, landmarks):
        """
        Args:
            landmarks: [B, 5, 2], normalized to [0, 1].

        Returns:
            Attention bias: [B, 1, N, N].
        """
        relative = compute_relative_positions(
            landmarks,
            self.patch_centers,
        )

        flattened = flatten_relative_positions(relative)

        projected = self.projection(flattened)

        bias = build_attention_bias(
            projected,
            self.bucket_indices,
        )

        return bias