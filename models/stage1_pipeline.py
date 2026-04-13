"""Stage 1: Feed-forward mesh generator from point clouds.

Generates a coarse base mesh (1000-3000 faces) from a dense point cloud,
compatible with Stage 2's face-anchored encoding pipeline.

Architecture:
    Stage1Encoder  → per-point features + global feature
    SeedPredictor  → FPS + learned offsets/normals
    DelaunayMeshExtractor → triangle mesh via tet classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_points, knn_gather

from .mesh_extraction import DelaunayMeshExtractor


def _farthest_point_sample(xyz, npoint):
    """Farthest point sampling.

    Adapted from LightweightMR/models/sampling.py:106-121.

    Args:
        xyz: (B, N, C) point positions
        npoint: number of points to sample

    Returns:
        centroids: (B, npoint) indices
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.linalg.norm(xyz - centroid, ord=2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.argmax(distance, -1)
    return centroids


class Stage1Encoder(nn.Module):
    """Shared MLP backbone for per-point feature extraction.

    Same architectural pattern as LocalFeatureEncoder in models/encoder.py.
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dims: list = None,
        global_feat_dim: int = 512,
        point_feat_dim: int = 128,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512]

        # Shared MLP backbone (per-point, no inter-point communication)
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
            ])
            in_dim = h_dim
        self.backbone = nn.Sequential(*layers)

        # Per-point feature projection
        self.point_proj = nn.Sequential(
            nn.Linear(hidden_dims[-1], point_feat_dim),
            nn.ReLU(inplace=True),
        )

        # Global feature projection (from max-pooled backbone)
        self.global_proj = nn.Sequential(
            nn.Linear(hidden_dims[-1], global_feat_dim),
            nn.ReLU(inplace=True),
        )

        self._backbone_out_dim = hidden_dims[-1]
        self.point_feat_dim = point_feat_dim
        self.global_feat_dim = global_feat_dim

    def forward(self, scan_points, scan_normals):
        """
        Args:
            scan_points:  (B, P, 3)
            scan_normals: (B, P, 3)

        Returns:
            global_feat:    (B, global_feat_dim)
            per_point_feat: (B, P, point_feat_dim)
        """
        B, P, _ = scan_points.shape
        x = torch.cat([scan_points, scan_normals], dim=-1)  # (B, P, 6)

        # Backbone with BatchNorm: reshape to (B*P, D) for BN, then back
        x_flat = x.reshape(B * P, -1)
        backbone_out = self.backbone(x_flat)  # (B*P, hidden_dims[-1])

        per_point_feat = self.point_proj(backbone_out).view(B, P, -1)  # (B, P, point_feat_dim)

        # Global feature via max pooling
        backbone_3d = backbone_out.view(B, P, -1)  # (B, P, hidden_dims[-1])
        global_pool = backbone_3d.max(dim=1).values  # (B, hidden_dims[-1])
        global_feat = self.global_proj(global_pool)  # (B, global_feat_dim)

        return global_feat, per_point_feat


class SeedPredictor(nn.Module):
    """Predict seed positions and normals from FPS-selected points.

    Uses FPS for initial selection, KNN local aggregation for context,
    and MLPs for learned position offsets and normal prediction.
    """

    def __init__(
        self,
        point_feat_dim: int = 128,
        global_feat_dim: int = 512,
        hidden_dim: int = 256,
        local_k: int = 32,
        fps_subsample: int = 50000,
    ):
        super().__init__()
        self.local_k = local_k
        self.fps_subsample = fps_subsample

        # Input: local_max_feat + seed_point_feat + global_feat + xyz + normal
        mlp_input_dim = point_feat_dim * 2 + global_feat_dim + 6

        # Offset prediction MLP
        self.offset_mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )

        # Normal prediction MLP
        self.normal_mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )

        # Near-zero init for offset (initial output = FPS positions)
        nn.init.uniform_(self.offset_mlp[-1].weight, -1e-4, 1e-4)
        nn.init.constant_(self.offset_mlp[-1].bias, 0.0)

    def forward(
        self,
        scan_points,
        scan_normals,
        per_point_feat,
        global_feat,
        num_seeds,
    ):
        """
        Args:
            scan_points:    (B, P, 3)
            scan_normals:   (B, P, 3)
            per_point_feat: (B, P, D_point)
            global_feat:    (B, D_global)
            num_seeds:      int, number of seed points to generate

        Returns:
            seed_positions: (B, N, 3) differentiable
            seed_normals:   (B, N, 3) differentiable, unit vectors
            fps_indices:    (B, N) indices into scan_points
        """
        B, P, _ = scan_points.shape
        device = scan_points.device

        # --- FPS with subsampling for speed ---
        if P > self.fps_subsample:
            # Random subsample, then FPS on subset
            sub_idx = torch.randperm(P, device=device)[:self.fps_subsample]
            sub_idx = sub_idx.unsqueeze(0).expand(B, -1)  # (B, S)
            sub_points = torch.gather(
                scan_points, 1, sub_idx.unsqueeze(-1).expand(-1, -1, 3)
            )  # (B, S, 3)
            fps_sub_idx = _farthest_point_sample(sub_points, num_seeds)  # (B, N) into [0, S)

            # Map back to original indices
            fps_indices = torch.gather(sub_idx, 1, fps_sub_idx)  # (B, N) into [0, P)
        else:
            fps_indices = _farthest_point_sample(scan_points, num_seeds)  # (B, N)

        # Gather point data at FPS positions
        bi = torch.arange(B, device=device).view(-1, 1).expand(-1, num_seeds)
        fps_points = scan_points[bi, fps_indices]       # (B, N, 3)
        fps_normals = scan_normals[bi, fps_indices]      # (B, N, 3)
        fps_feat = per_point_feat[bi, fps_indices]       # (B, N, D_point)

        # --- KNN local aggregation around each seed ---
        knn_result = knn_points(fps_points, scan_points, K=self.local_k)
        knn_idx = knn_result.idx  # (B, N, K)
        knn_feat = knn_gather(per_point_feat, knn_idx)  # (B, N, K, D_point)
        local_max_feat = knn_feat.max(dim=2).values  # (B, N, D_point)

        # --- Assemble MLP input ---
        global_exp = global_feat.unsqueeze(1).expand(-1, num_seeds, -1)  # (B, N, D_global)
        mlp_input = torch.cat([
            local_max_feat, fps_feat, global_exp, fps_points, fps_normals
        ], dim=-1)  # (B, N, D_point*2 + D_global + 6)

        # --- Predict offsets and normals ---
        offsets = self.offset_mlp(mlp_input)  # (B, N, 3)
        raw_normals = self.normal_mlp(mlp_input)  # (B, N, 3)

        seed_positions = fps_points + offsets  # differentiable via offsets
        seed_normals = F.normalize(raw_normals, dim=-1)

        return seed_positions, seed_normals, fps_indices


class Stage1Pipeline(nn.Module):
    """Complete Stage 1 pipeline: point cloud → base mesh.

    Wraps encoder, seed predictor, and mesh extractor into a single module
    whose output is directly compatible with Stage2Pipeline.
    """

    def __init__(self, config: dict):
        super().__init__()

        point_feat_dim = config.get('point_feat_dim', 128)
        global_feat_dim = config.get('global_feat_dim', 512)
        hidden_dims = config.get('encoder_hidden_dims', [64, 128, 256, 512])

        self.encoder = Stage1Encoder(
            input_dim=6,
            hidden_dims=hidden_dims,
            global_feat_dim=global_feat_dim,
            point_feat_dim=point_feat_dim,
        )

        self.seed_predictor = SeedPredictor(
            point_feat_dim=point_feat_dim,
            global_feat_dim=global_feat_dim,
            hidden_dim=config.get('seed_hidden_dim', 256),
            local_k=config.get('seed_local_k', 32),
            fps_subsample=config.get('fps_subsample', 50000),
        )

        self.mesh_extractor = DelaunayMeshExtractor(
            bbox_padding=config.get('bbox_padding', 0.15),
            vote_threshold=config.get('vote_threshold', 3),
        )

        self.num_seeds_min = config.get('num_seeds_min', 600)
        self.num_seeds_max = config.get('num_seeds_max', 1500)

    def forward(self, scan_points, scan_normals, num_seeds=None):
        """
        Args:
            scan_points:  (B, P, 3)
            scan_normals: (B, P, 3)
            num_seeds:    int or None (random from range if None)

        Returns:
            base_verts:   (1, V, 3) differentiable
            base_faces:   (1, F, 3) int64
            base_normals: (1, V, 3) differentiable unit normals
            aux: dict with seed_positions, seed_normals, fps_indices, vert_indices
        """
        B = scan_points.shape[0]
        assert B == 1, "Stage 1 currently supports batch_size=1 only"

        if num_seeds is None:
            num_seeds = torch.randint(
                self.num_seeds_min, self.num_seeds_max + 1, (1,)
            ).item()

        # 1. Feature extraction
        global_feat, per_point_feat = self.encoder(scan_points, scan_normals)

        # 2. Seed prediction
        seed_positions, seed_normals, fps_indices = self.seed_predictor(
            scan_points, scan_normals, per_point_feat, global_feat, num_seeds
        )

        # 3. Mesh extraction (non-differentiable topology, preserves position grad)
        faces, vert_indices = self.mesh_extractor.extract(
            seed_positions[0], seed_normals[0]
        )

        # Index back into differentiable tensors
        base_verts = seed_positions[0][vert_indices]  # (V, 3) — preserves grad!
        base_normals = seed_normals[0][vert_indices]  # (V, 3)

        # Compute vertex normals from face geometry (more accurate for Stage 2)
        base_normals_geom = self._compute_vertex_normals(base_verts, faces)
        # Use predicted normals where geometric normals are degenerate
        valid_mask = base_normals_geom.norm(dim=-1) > 1e-6
        final_normals = torch.where(
            valid_mask.unsqueeze(-1), base_normals_geom, base_normals
        )
        final_normals = F.normalize(final_normals, dim=-1)

        return (
            base_verts.unsqueeze(0),   # (1, V, 3)
            faces.unsqueeze(0),        # (1, F, 3)
            final_normals.unsqueeze(0),  # (1, V, 3)
            {
                'seed_positions': seed_positions,
                'seed_normals': seed_normals,
                'fps_indices': fps_indices,
                'vert_indices': vert_indices,
                'num_seeds': num_seeds,
            },
        )

    @staticmethod
    def _compute_vertex_normals(verts, faces):
        """Compute per-vertex normals from face geometry (area-weighted).

        Args:
            verts: (V, 3) vertex positions
            faces: (F, 3) int64 face indices

        Returns:
            normals: (V, 3) unit normals
        """
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)  # (F, 3) area-weighted

        V = verts.shape[0]
        vertex_normals = torch.zeros(V, 3, device=verts.device, dtype=verts.dtype)
        for i in range(3):
            vertex_normals.scatter_add_(0, faces[:, i:i+1].expand(-1, 3), face_normals)

        return F.normalize(vertex_normals, dim=-1)

    def forward_seeds_only(self, scan_points, scan_normals, num_seeds=None):
        """Forward pass that returns only seed positions/normals (no mesh extraction).

        Used in Phase 1 training where only Chamfer + normal losses are needed.
        Much faster since it skips Delaunay tetrahedralization.
        """
        B = scan_points.shape[0]
        assert B == 1

        if num_seeds is None:
            num_seeds = torch.randint(
                self.num_seeds_min, self.num_seeds_max + 1, (1,)
            ).item()

        global_feat, per_point_feat = self.encoder(scan_points, scan_normals)
        seed_positions, seed_normals, fps_indices = self.seed_predictor(
            scan_points, scan_normals, per_point_feat, global_feat, num_seeds
        )

        return seed_positions, seed_normals, fps_indices
