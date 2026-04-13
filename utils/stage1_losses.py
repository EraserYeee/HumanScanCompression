"""Stage 1 loss functions for mesh generation training.

Adapted from:
  - LightweightMR/models/losses.py: curvature estimation, curvature-weighted Chamfer
  - LightweightMR/models/modules/netutils.py: Gaussian kernel
"""

import torch
import torch.nn.functional as F
from pytorch3d.ops import knn_points


def gaussian_kernel(points, queries):
    """Gaussian kernel weights for KNN neighbors.

    Adapted from LightweightMR/models/modules/netutils.py:41-46.

    Args:
        points:  (N, K, 3) neighbor positions
        queries: (N, 3) query positions

    Returns:
        weights: (N, K) normalized Gaussian weights
    """
    pts_dist = torch.linalg.norm(points - queries.unsqueeze(1), ord=2, dim=-1)  # (N, K)
    h = pts_dist.mean(dim=-1, keepdim=True).clamp(min=1e-8)  # (N, 1) bandwidth
    dist_exp = torch.exp(-pts_dist ** 2 / h ** 2)  # (N, K)
    weights = dist_exp / dist_exp.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # (N, K)
    return weights


@torch.no_grad()
def compute_curvature_from_normals(points, normals, knn_k=30):
    """Estimate surface curvature from point cloud normals.

    Adapted from LightweightMR/models/losses.py:78-89 (cal_curvature_with_normal).

    For each point, measures normal variation among KNN neighbors,
    smoothed with Gaussian kernel, then sigmoid-normalized to [0, 1].

    Args:
        points:  (P, 3) point positions
        normals: (P, 3) unit normals
        knn_k:   number of neighbors (excluding self)

    Returns:
        curvature: (P, 1) normalized curvature in [0, 1]
    """
    # KNN (k+1 to exclude self)
    knn_result = knn_points(
        points.unsqueeze(0), points.unsqueeze(0), K=knn_k + 1
    )
    neigh_idx = knn_result.idx[0, :, 1:]  # (P, K) exclude self

    neigh_pts = points[neigh_idx]       # (P, K, 3)
    neigh_normals = normals[neigh_idx]  # (P, K, 3)

    # Curvature proxy: normal variation
    neigh_curvature = 1.0 - F.cosine_similarity(
        normals.unsqueeze(1), neigh_normals, dim=-1
    )  # (P, K)

    # Gaussian-weighted average
    g_weight = gaussian_kernel(neigh_pts, points)  # (P, K)
    curvature_ave = (neigh_curvature * g_weight).sum(dim=-1, keepdim=True)  # (P, 1)

    # Sigmoid normalization (LightweightMR pattern)
    curvature_ave = torch.sigmoid(curvature_ave - curvature_ave.mean())
    cur_min = curvature_ave.min()
    cur_max = curvature_ave.max()
    curvature_ave = (curvature_ave - cur_min + 1e-6) / (cur_max - cur_min + 1e-6)

    return curvature_ave  # (P, 1)


def curvature_weighted_chamfer(
    scan_points,
    scan_curvature,
    seed_positions,
    w_curv=3e3,
    w_uniform=1e2,
    w_backward=1e3,
    w_repulsion=1e2,
    repulsion_clamp=3.0,
):
    """Curvature-weighted bidirectional Chamfer distance with self-repulsion.

    Adapted from LightweightMR/models/losses.py:143-167 (cal_chamfer_loss).

    Forward (scan → seed): weighted by curvature to encourage more seeds
    near high-curvature regions.
    Backward (seed → scan): uniform, ensures all seeds lie on surface.
    Self-repulsion: negative nearest-neighbor distance among seeds to prevent collapse.

    Args:
        scan_points:    (P, 3) GT scan points
        scan_curvature: (P, 1) precomputed curvature weights in [0, 1]
        seed_positions: (N, 3) predicted seed positions (differentiable)
        w_curv:         weight for curvature-weighted forward term
        w_uniform:      weight for uniform forward term
        w_backward:     weight for backward term
        w_repulsion:    weight for self-repulsion term
        repulsion_clamp: clamp factor for self-repulsion (max = clamp * mean)

    Returns:
        loss: scalar loss value
        loss_dict: dict with individual loss components for logging
    """
    # Forward: scan → nearest seed (LightweightMR lines 146-149)
    fwd = knn_points(scan_points.unsqueeze(0), seed_positions.unsqueeze(0), K=1)
    dist_fwd = fwd.dists[0, :, 0]  # (P,) squared distances

    loss_fwd_curv = w_curv * (dist_fwd * scan_curvature.squeeze(-1)).mean()
    loss_fwd_uniform = w_uniform * dist_fwd.mean()

    # Backward: seed → nearest scan (LightweightMR lines 151-155)
    bwd = knn_points(seed_positions.unsqueeze(0), scan_points.unsqueeze(0), K=1)
    dist_bwd = bwd.dists[0, :, 0]  # (N,)

    loss_backward = w_backward * dist_bwd.mean()

    # Self-repulsion: seed → nearest other seed (LightweightMR lines 157-163)
    self_knn = knn_points(seed_positions.unsqueeze(0), seed_positions.unsqueeze(0), K=2)
    dist_self = self_knn.dists[0, :, 1]  # (N,) exclude self

    clamp_max = repulsion_clamp * dist_self.mean().detach()
    loss_repulsion = -w_repulsion * torch.clamp(dist_self, max=clamp_max).mean()

    loss = loss_fwd_curv + loss_fwd_uniform + loss_backward + loss_repulsion

    loss_dict = {
        'chamfer_fwd_curv': loss_fwd_curv.item(),
        'chamfer_fwd_uniform': loss_fwd_uniform.item(),
        'chamfer_backward': loss_backward.item(),
        'chamfer_repulsion': loss_repulsion.item(),
    }
    return loss, loss_dict


def normal_consistency_loss(seed_positions, seed_normals, scan_points, scan_normals):
    """Normal consistency between predicted seed normals and nearest GT normals.

    Adapted from LightweightMR/models/losses.py:137-140 (cal_nc_loss).

    Args:
        seed_positions: (N, 3) predicted seed positions
        seed_normals:   (N, 3) predicted unit normals
        scan_points:    (P, 3) GT scan points
        scan_normals:   (P, 3) GT unit normals

    Returns:
        loss: scalar (1 - mean cosine similarity)
    """
    # Find nearest scan point for each seed
    knn_result = knn_points(
        seed_positions.unsqueeze(0), scan_points.unsqueeze(0), K=1
    )
    nearest_idx = knn_result.idx[0, :, 0]  # (N,)
    nearest_normals = scan_normals[nearest_idx]  # (N, 3)

    loss = (1.0 - F.cosine_similarity(seed_normals, nearest_normals, dim=-1)).mean()
    return loss


def mesh_regularity_loss(positions, faces):
    """Penalize degenerate triangles: edge length variance + zero-area faces.

    Args:
        positions: (V, 3) vertex positions
        faces:     (F, 3) int64 face indices

    Returns:
        loss: scalar
    """
    v0 = positions[faces[:, 0]]
    v1 = positions[faces[:, 1]]
    v2 = positions[faces[:, 2]]

    # Edge lengths
    e01 = torch.linalg.norm(v1 - v0, dim=-1)
    e12 = torch.linalg.norm(v2 - v1, dim=-1)
    e20 = torch.linalg.norm(v0 - v2, dim=-1)
    all_edges = torch.cat([e01, e12, e20])

    mean_len = all_edges.mean().clamp(min=1e-8)
    edge_var_loss = all_edges.var() / (mean_len ** 2)

    # Face area penalty (avoid degenerate faces)
    cross = torch.cross(v1 - v0, v2 - v0, dim=-1)
    areas = 0.5 * torch.linalg.norm(cross, dim=-1)
    area_penalty = torch.relu(1e-6 - areas).mean() * 100.0

    return edge_var_loss + area_penalty
