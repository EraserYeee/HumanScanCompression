import torch
import torch.nn as nn
from torch_scatter import scatter_max

class LocalFeatureEncoder(nn.Module):
    """
    负责从分组后的局部点云中提取 Base Mesh 顶点的特征向量。

    职责:
    1. 对每个局部点应用 PointMLP (Point-wise features)。
    2. 使用 Scatter Max 将属于同一顶点的点特征聚合。
    3. 输出 Base Mesh 顶点的特征 Embedding。
    """
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        
        # Point-wise MLP (Shared MLP)
        # [Input -> 64 -> 64 -> Output]
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, local_points: torch.Tensor, cluster_idx: torch.Tensor, num_verts: int):
        """
        Args:
            local_points: (B, P, 3) 局部坐标点
            cluster_idx: (B, P) 点归属索引, 值域 [0, V-1]
            num_verts: int 最大顶点数 V (用于 scatter 的 dim_size)

        Returns:
            vertex_features: (B, V, D) 每个 Base Mesh 顶点的特征
        """
        B, P, _ = local_points.shape
        
        # 1. Point-wise Feature Extraction
        # (B, P, 3) -> (B, P, D)
        point_feats = self.mlp(local_points)
        
        # 2. Scatter Aggregation (Max Pooling)
        # scatter_max 不支持 Batch 维度直接操作 (它在 dim=0 上操作)，
        # 所以我们需要将 Batch 维度展平，并调整 cluster_idx 加上 batch offset。
        
        # Flatten features: (B*P, D)
        flat_point_feats = point_feats.view(-1, self.output_dim)
        
        # Create batch offset for indices
        # offset: [0, V, 2V, ...] shape (B, 1) -> expand to (B, P)
        batch_offset = (torch.arange(B, device=local_points.device) * num_verts).view(-1, 1)
        
        # global_idx: (B, P) -> (B*P)
        # global_idx[i] = batch_idx * V + vertex_idx
        global_cluster_idx = (cluster_idx + batch_offset).view(-1)
        
        # Max Pooling Aggregation
        # out: (B*V, D)
        # dim_size = B * V (Total number of vertices in the batch)
        # 注意: scatter_max 返回 (values, indices)，我们要 values
        aggregated_feats, _ = scatter_max(
            flat_point_feats, 
            global_cluster_idx, 
            dim=0, 
            dim_size=B * num_verts
        )
        
        # 3. Reshape back to Batch
        # (B*V, D) -> (B, V, D)
        vertex_features = aggregated_feats.view(B, num_verts, self.output_dim)
        
        # 注意: 如果某个顶点没有分配到任何点，scatter_max 默认返回 0 (对于 ReLU 激活后的特征是合理的)
        # 如果使用的是 scatter_mean，可能需要处理分母为 0 的情况
        
        return vertex_features
