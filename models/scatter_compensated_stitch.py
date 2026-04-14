"""Seam stitching: forward scatter_mean, backward gradient compensation (方案 A).

Shared subdivision points receive 1/N of the usual gradient from scatter_mean.
This module keeps the forward identical but scales each contributor's grad by N
so boundary/vertex predictions learn at a rate comparable to interior points.
"""

import torch
from torch_scatter import scatter_mean, scatter_sum


class scatter_mean_gather_grad_compensated(torch.autograd.Function):
    """out[j] = mean_{k: index[k]==index[j]} src[k]; backward multiplies per-row grad by group size."""

    @staticmethod
    def forward(ctx, src: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        averaged = scatter_mean(src, index, dim=0)
        ctx.dim_size = int(averaged.size(0))
        ctx.save_for_backward(index)
        return averaged[index]

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (index,) = ctx.saved_tensors
        dim_size = ctx.dim_size
        grad_sum = scatter_sum(grad_out, index, dim=0, dim_size=dim_size)
        grad_src = grad_sum[index]
        return grad_src, None


def stitch_displacements_compensated(
    disp_flat: torch.Tensor, merge_idx: torch.Tensor
) -> torch.Tensor:
    """Same as scatter_mean(disp, idx)[idx] forward; compensated backward.

    Args:
        disp_flat: (N, C) per-face-subdivision displacements
        merge_idx: (N,) long, maps each row to a unique stitch id

    Returns:
        (N, C) stitched displacements (identical to naive scatter_mean gather)
    """
    return scatter_mean_gather_grad_compensated.apply(disp_flat, merge_idx)
