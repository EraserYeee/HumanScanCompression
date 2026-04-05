"""
Stage-2 training helpers: geometric losses, config I/O, debug mesh/image export.
Kept out of train_stage2.py for readability.
"""
from __future__ import annotations

import os
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image


# --- Laplacian (NGF-style uniform, L1) ---


@torch.no_grad()
def _build_uniform_adjacency(faces, num_verts, device):
    """Build unique directed-edge lists and per-vertex degree from triangle faces."""
    idx = faces.long()
    rows = torch.cat([idx[:, 0], idx[:, 1], idx[:, 1], idx[:, 2], idx[:, 0], idx[:, 2]])
    cols = torch.cat([idx[:, 1], idx[:, 0], idx[:, 2], idx[:, 1], idx[:, 2], idx[:, 0]])
    adj = torch.sparse_coo_tensor(
        torch.stack([rows, cols]),
        torch.ones(rows.shape[0], device=device),
        size=(num_verts, num_verts),
    ).coalesce()
    src = adj.indices()[0]
    dst = adj.indices()[1]
    degree = torch.zeros(num_verts, device=device)
    degree.index_add_(0, src, torch.ones(src.shape[0], device=device))
    degree.clamp_(min=1.0)
    return src, dst, degree


def compute_uniform_laplacian_l1(verts, faces):
    """
    NGF-style uniform Laplacian regularization (L1).
    L = mean |v_i - mean(v_j for j in N(i))|
    """
    V = verts.shape[0]
    src, dst, degree = _build_uniform_adjacency(faces, V, verts.device)
    neighbor_sum = torch.zeros(V, 3, device=verts.device)
    neighbor_sum.scatter_add_(0, src.unsqueeze(1).expand(-1, 3), verts[dst])
    smoothed = neighbor_sum / degree.unsqueeze(1)
    return (verts - smoothed).abs().mean()


# --- Geo normal loss + spatial gradients ---


def compute_normal_spatial_gradients(pred_normal, gt_normal):
    """
    Same finite differences as in the geo normal loss (on L2-normalized normals).
    pred_dx / pred_dy: ∂N/∂x, ∂N/∂y with shapes (B, H, W-1, 3) and (B, H-1, W, 3).
    """
    pred_n = F.normalize(pred_normal, dim=-1, p=2)
    gt_n = F.normalize(gt_normal, dim=-1, p=2)
    pred_dx = pred_n[:, :, 1:, :] - pred_n[:, :, :-1, :]
    gt_dx = gt_n[:, :, 1:, :] - gt_n[:, :, :-1, :]
    pred_dy = pred_n[:, 1:, :, :] - pred_n[:, :-1, :, :]
    gt_dy = gt_n[:, 1:, :, :] - gt_n[:, :-1, :, :]
    return pred_n, gt_n, pred_dx, pred_dy, gt_dx, gt_dy


def compute_normal_loss_geo(pred_normal, gt_normal, mask):
    """
    L = (1 - N_hat·N) + ||N_hat - N||^2 + ||∇N_hat - ∇N||^2 (masked).
    """
    mask_f = mask.float()
    num_valid = mask_f.sum().clamp(min=1.0)

    pred_n, gt_n, pred_dx, pred_dy, gt_dx, gt_dy = compute_normal_spatial_gradients(pred_normal, gt_normal)

    cos_dist = (1.0 - (pred_n * gt_n).sum(dim=-1, keepdim=True)) * mask_f
    loss_cos = cos_dist.sum() / num_valid

    l2_dist = ((pred_n - gt_n) ** 2).sum(dim=-1, keepdim=True) * mask_f
    loss_l2 = l2_dist.sum() / num_valid

    mask_dx = mask_f[:, :, 1:, :] * mask_f[:, :, :-1, :]
    mask_dy = mask_f[:, 1:, :, :] * mask_f[:, :-1, :, :]

    grad_loss_x = (((pred_dx - gt_dx) ** 2).sum(dim=-1, keepdim=True) * mask_dx).sum() / mask_dx.sum().clamp(min=1.0)
    grad_loss_y = (((pred_dy - gt_dy) ** 2).sum(dim=-1, keepdim=True) * mask_dy).sum() / mask_dy.sum().clamp(min=1.0)
    loss_grad = grad_loss_x + grad_loss_y

    return loss_cos + loss_l2 + loss_grad


def _pad_normal_dx_to_hw(dx):
    """(B, H, W-1, 3) -> (B, H, W, 3), pad missing column with 0."""
    return F.pad(dx, (0, 0, 0, 1, 0, 0, 0, 0))


def _pad_normal_dy_to_hw(dy):
    """(B, H-1, W, 3) -> (B, H, W, 3), pad missing row with 0."""
    return F.pad(dy, (0, 0, 0, 0, 0, 1, 0, 0))


def _numpy_signed_vec_to_rgb(arr):
    """arr (H, W, 3): robust [-1,1] RGB viz (per-image max-abs scale)."""
    m = float(np.abs(arr).max())
    if m < 1e-8:
        return np.zeros((*arr.shape[:2], 3), dtype=np.uint8)
    scaled = np.clip(arr / m, -1.0, 1.0)
    return ((scaled + 1.0) * 0.5 * 255.0).astype(np.uint8)


def _numpy_grayscale_from_mag(dx_hw, dy_hw):
    mag = np.sqrt((dx_hw ** 2).sum(-1) + (dy_hw ** 2).sum(-1))
    m = float(mag.max()) + 1e-8
    return np.clip(mag / m * 255.0, 0, 255).astype(np.uint8)


def _numpy_err_heatmap(err_vec):
    e = np.sqrt((err_vec ** 2).sum(-1))
    m = float(e.max()) + 1e-8
    return np.clip(e / m * 255.0, 0, 255).astype(np.uint8)


# --- Config ---


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config, save_path):
    with open(save_path, 'w') as f:
        yaml.dump(config, f)


# --- Debug export ---


def debug_export_meshes(base_v, base_f, fine_v, fine_f, gt_v, gt_f, step, batch_idx, tag=""):
    save_dir = f"/mnt/lab/data/yeruisi/data/compression/debug_train_meshes/step_{step:04d}_b{batch_idx}"
    os.makedirs(save_dir, exist_ok=True)

    if base_v is not None:
        trimesh.Trimesh(
            vertices=base_v.detach().cpu().numpy(),
            faces=base_f.detach().cpu().numpy(),
        ).export(os.path.join(save_dir, f"base_{tag}.obj"))

    if fine_v is not None:
        trimesh.Trimesh(
            vertices=fine_v.detach().cpu().numpy(),
            faces=fine_f.detach().cpu().numpy(),
        ).export(os.path.join(save_dir, f"fine_{tag}.obj"))

    if gt_v is not None:
        trimesh.Trimesh(
            vertices=gt_v.detach().cpu().numpy(),
            faces=gt_f.detach().cpu().numpy(),
        ).export(os.path.join(save_dir, f"gt_{tag}.obj"))

    print(f"[Debug] Exported meshes to {save_dir}")


def debug_export_images(pred_img, gt_img, step, batch_idx, tag="", include_geo_grad=False):
    """
    pred_img, gt_img: (K, H, W, 3) or (B*K, H, W, 3)
    include_geo_grad: export ∂N/∂x, ∂N/∂y visualizations matching geo loss.
    """
    save_dir = f"/mnt/lab/data/yeruisi/data/compression/debug_train_images/step_{step:04d}_b{batch_idx}"
    os.makedirs(save_dir, exist_ok=True)

    count = min(4, pred_img.shape[0])

    for i in range(count):
        p = pred_img[i].detach().cpu().numpy()
        p = (p + 1.0) * 0.5 * 255.0
        p = np.clip(p, 0, 255).astype(np.uint8)
        Image.fromarray(p).save(os.path.join(save_dir, f"pred_view_{i}_{tag}.png"))

        if gt_img is not None:
            g = gt_img[i].detach().cpu().numpy()
            g = (g + 1.0) * 0.5 * 255.0
            g = np.clip(g, 0, 255).astype(np.uint8)
            Image.fromarray(g).save(os.path.join(save_dir, f"gt_view_{i}_{tag}.png"))

        if include_geo_grad and gt_img is not None:
            with torch.no_grad():
                pi = pred_img[i : i + 1]
                gi = gt_img[i : i + 1]
                _, _, pdx, pdy, gdx, gdy = compute_normal_spatial_gradients(pi, gi)
                pdx_hw = _pad_normal_dx_to_hw(pdx)[0].detach().cpu().numpy()
                pdy_hw = _pad_normal_dy_to_hw(pdy)[0].detach().cpu().numpy()
                gdx_hw = _pad_normal_dx_to_hw(gdx)[0].detach().cpu().numpy()
                gdy_hw = _pad_normal_dy_to_hw(gdy)[0].detach().cpu().numpy()

            Image.fromarray(_numpy_signed_vec_to_rgb(pdx_hw)).save(
                os.path.join(save_dir, f"pred_dndx_view_{i}_{tag}.png")
            )
            Image.fromarray(_numpy_signed_vec_to_rgb(pdy_hw)).save(
                os.path.join(save_dir, f"pred_dndy_view_{i}_{tag}.png")
            )
            Image.fromarray(_numpy_signed_vec_to_rgb(gdx_hw)).save(
                os.path.join(save_dir, f"gt_dndx_view_{i}_{tag}.png")
            )
            Image.fromarray(_numpy_signed_vec_to_rgb(gdy_hw)).save(
                os.path.join(save_dir, f"gt_dndy_view_{i}_{tag}.png")
            )
            Image.fromarray(_numpy_grayscale_from_mag(pdx_hw, pdy_hw)).save(
                os.path.join(save_dir, f"pred_gradmag_view_{i}_{tag}.png")
            )
            Image.fromarray(_numpy_grayscale_from_mag(gdx_hw, gdy_hw)).save(
                os.path.join(save_dir, f"gt_gradmag_view_{i}_{tag}.png")
            )
            err_x = pdx_hw - gdx_hw
            err_y = pdy_hw - gdy_hw
            Image.fromarray(_numpy_err_heatmap(err_x)).save(
                os.path.join(save_dir, f"err_grad_dx_view_{i}_{tag}.png")
            )
            Image.fromarray(_numpy_err_heatmap(err_y)).save(
                os.path.join(save_dir, f"err_grad_dy_view_{i}_{tag}.png")
            )

    print(f"[Debug] Exported rendered images to {save_dir}")
