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
