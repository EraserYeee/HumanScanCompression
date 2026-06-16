"""Base mesh vertex displacement head (2.6 lightweight two-stage tier).

把 base mesh 顶点先"掰正"到更贴合 scan 的位置, 之后 FaceTriangleDecoder 在
corrected base 上做细分位移。详见 plan_rvq_and_base_displacement.md Part 2。

设计要点:
- 与 fine decoder **权重独立**: base 位移是 per-vertex 全局搬运(大位移), fine 是
  per-face 高频细节(小残差), 共享权重会让 base 欠位移。
- per-face 特征经 scatter_mean 聚合到每个 base 顶点, 再 MLP 预测 3-DoF 世界坐标位移。
- 末层 near-zero 初始化 => step 0 时 corrected_base ≈ base(恒等起步, 稳定);
  位移幅度由 fine 渲染 loss 端到端学出(2.6 轻量档: 不 re-encode)。
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_scatter import scatter_mean
from .grouper import VertexKNNGrouper


class BaseVertexDisplacementHead(nn.Module):
    """Predict a per-base-vertex world-space displacement from per-face features.

    Args:
        feature_dim: per-face feature dim D (post-RVQ feature fed to decoder)
        hidden_dim: int or list[int], MLP 隐藏层
        init_scale: 末层权重 uniform(-init_scale, init_scale) 初始化幅度
        use_normal: 是否把 base vertex normal 拼到 MLP 输入(提供朝向上下文)
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim=(128, 64),
        init_scale: float = 1e-4,
        use_normal: bool = True,
    ):
        super().__init__()
        self.use_normal = use_normal
        in_dim = feature_dim + (3 if use_normal else 0)
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim]
        dims = [in_dim] + list(hidden_dim) + [3]

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LeakyReLU())
        self.mlp = nn.Sequential(*layers)

        nn.init.uniform_(self.mlp[-1].weight, -init_scale, init_scale)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        base_verts: torch.Tensor,
        base_faces: torch.Tensor,
        face_features: torch.Tensor,
        base_normals: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            base_verts:    (B, V, 3)
            base_faces:    (B, F, 3) LongTensor, 顶点索引 ∈ [0, V)
            face_features: (B, F, D) per-face feature (post-RVQ)
            base_normals:  (B, V, 3) or None

        Returns:
            base_disp: (B, V, 3) world-space per-vertex displacement
        """
        B, V, _ = base_verts.shape
        _, F_num, _ = base_faces.shape
        D = face_features.shape[-1]

        # 每个面把自己的特征广播给 3 个顶点, 再 scatter_mean 聚合到顶点。
        feat_per_corner = face_features.unsqueeze(2).expand(B, F_num, 3, D)   # (B,F,3,D)
        batch_offset = (torch.arange(B, device=base_verts.device) * V).view(B, 1, 1)
        flat_idx = (base_faces + batch_offset).reshape(-1)                    # (B*F*3,)
        flat_feat = feat_per_corner.reshape(-1, D)                            # (B*F*3, D)
        vert_feat = scatter_mean(flat_feat, flat_idx, dim=0, dim_size=B * V)  # (B*V, D)
        vert_feat = vert_feat.reshape(B, V, D)

        if self.use_normal and base_normals is not None:
            mlp_in = torch.cat([vert_feat, base_normals], dim=-1)
        else:
            mlp_in = vert_feat

        base_disp = self.mlp(mlp_in.reshape(-1, mlp_in.shape[-1])).reshape(B, V, 3)
        return base_disp


class BaseDisplacePredictor(nn.Module):
    """#1: 廉价、几何驱动的 base 顶点位移预测头(独立于重型 face encoder)。

    每个 base 顶点对 scan 做 *小 K* 的 vertex-KNN, 经 PointNet(共享 MLP + maxpool)
    得到逐顶点特征, 再 MLP 预测 3-DoF 世界位移。整个模块 **无 self-attention**, 廉价。
    pipeline 用它先把 base 移好, 重型 face encoder 之后只在 corrected base 上跑一次,
    从而避免 re-encode 的"重型 encoder 跑两遍"(计算/显存翻倍)。

    与 BaseVertexDisplacementHead(聚合 face 特征)的区别: 这里 **直接看 scan 局部
    几何**(相对坐标 + 法线), 信号更强更具体, 治 #2 的"base 几乎不动只被平滑"。

    Args:
        knn_k: 每个 base 顶点取的 scan 邻居数(小, 如 16~64)
        pointnet_hidden: 逐点共享 MLP 隐藏维
        head_hidden: pooled 特征 -> 位移 的 MLP 隐藏维
        use_normal: 是否用 scan 法线(逐点输入)与 base 法线(pooled 后拼接)
        init_scale: 末层 near-zero 初始化幅度(corrected_base≈base 起步, 稳)
        knn_chunk_size: vertex-KNN 分块大小
    """

    def __init__(
        self,
        knn_k: int = 32,
        pointnet_hidden=(64, 128),
        head_hidden=(64,),
        use_normal: bool = True,
        init_scale: float = 1e-4,
        knn_chunk_size: int = 512,
    ):
        super().__init__()
        self.use_normal = use_normal
        self.grouper = VertexKNNGrouper(k=knn_k, knn_chunk_size=knn_chunk_size)

        in_dim = 6 if use_normal else 3  # rel_xyz (+ scan_normal)
        if isinstance(pointnet_hidden, int):
            pointnet_hidden = [pointnet_hidden]
        pn_dims = [in_dim] + list(pointnet_hidden)
        pn_layers = []
        for i in range(len(pn_dims) - 1):
            pn_layers += [nn.Linear(pn_dims[i], pn_dims[i + 1]), nn.LeakyReLU()]
        self.pointnet = nn.Sequential(*pn_layers)
        feat_dim = pn_dims[-1]

        if isinstance(head_hidden, int):
            head_hidden = [head_hidden]
        h_in = feat_dim + (3 if use_normal else 0)
        h_dims = [h_in] + list(head_hidden) + [3]
        h_layers = []
        for i in range(len(h_dims) - 1):
            h_layers.append(nn.Linear(h_dims[i], h_dims[i + 1]))
            if i < len(h_dims) - 2:
                h_layers.append(nn.LeakyReLU())
        self.head = nn.Sequential(*h_layers)

        nn.init.uniform_(self.head[-1].weight, -init_scale, init_scale)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        base_verts: torch.Tensor,
        base_normals: torch.Tensor | None,
        scan_points: torch.Tensor,
        scan_normals: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            base_verts:   (B, V, 3)
            base_normals: (B, V, 3) or None
            scan_points:  (B, P, 3)
            scan_normals: (B, P, 3) or None

        Returns:
            base_disp: (B, V, 3) world-space per-vertex displacement
        """
        # 逐顶点 gather K 个 scan 邻居(局部坐标: 相对位置[, scan 法线])
        grouped, _ = self.grouper(
            base_verts, base_normals, scan_points,
            scan_normals=scan_normals if self.use_normal else None,
        )                                                # (B, V, K, Din)
        B, Vv, K, Din = grouped.shape

        x = self.pointnet(grouped.reshape(B * Vv * K, Din))
        x = x.reshape(B, Vv, K, -1)
        pooled = x.max(dim=2).values                     # (B, V, feat) max-pool over K

        if self.use_normal and base_normals is not None:
            pooled = torch.cat([pooled, base_normals], dim=-1)

        base_disp = self.head(pooled.reshape(B * Vv, -1)).reshape(B, Vv, 3)
        return base_disp
