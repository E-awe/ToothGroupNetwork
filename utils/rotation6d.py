"""PyTorch3D-compatible 6D rotations, implemented using only PyTorch.

The convention is the FIRST TWO ROWS of a rotation matrix, not its columns.
Leading batch dimensions, dtype, device and autograd are preserved.
"""
import torch
import torch.nn.functional as F


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """Convert (..., 3, 3) matrices to (..., 6), as in PyTorch3D."""
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("Expected rotation matrices with shape (..., 3, 3)")
    return matrix[..., :2, :].clone().reshape(*matrix.shape[:-2], 6)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Convert (..., 6) to (..., 3, 3) using PyTorch3D's Gram-Schmidt rule."""
    if d6.shape[-1:] != (6,):
        raise ValueError("Expected 6D rotations with shape (..., 6)")
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)
