import torch
import torch.nn as nn
from torch_scatter import scatter_max
from .tnet import ScatterSTNkd

class LocalFeatureEncoder(nn.Module):
    """
    负责从分组后的局部点云中提取 Base Mesh 顶点的特征向量。
    
    Updated: Integrated ScatterSTNkd (Feature Transform)
    """
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=128, use_feature_transform=True):
        super().__init__()
        self.output_dim = output_dim
        self.use_feature_transform = use_feature_transform
        
        # 1. First Layer (Input -> 64)
        # PointNet: 64 dim before T-Net
        self.conv1 = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # 2. Feature Transform (STNkd)
        if self.use_feature_transform:
            self.fstn = ScatterSTNkd(k=64)
            
        # 3. Subsequent Layers
        # 64 -> hidden(usually 128) -> output(usually 1024 in PointNet, but here we keep it smaller e.g. 128)
        self.conv2 = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        self.conv3 = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim)
            # No ReLU here usually if it's the final feature before pooling? 
            # Original PointNet uses ReLU before MaxPool. Let's add it.
        )
        self.relu = nn.ReLU()

    def forward(self, local_points: torch.Tensor, cluster_idx: torch.Tensor, num_verts: int):
        """
        Args:
            local_points: (B, P, 3) 局部坐标点
            cluster_idx: (B, P) 点归属索引, 值域 [0, V-1]
            num_verts: int 最大顶点数 V (用于 scatter 的 dim_size)

        Returns:
            vertex_features: (B, V, D) 每个 Base Mesh 顶点的特征
            trans_feat: (B*V, K, K) or None
        """
        B, P, D = local_points.shape
        
        # Flatten for Scatter operations
        # local_points: (B*P, 3)
        flat_points = local_points.view(-1, D)
        
        # Calculate Global Cluster Indices (handling batch offset)
        # offset: [0, V, 2V, ...] shape (B, 1) -> expand to (B, P)
        batch_offset = (torch.arange(B, device=local_points.device) * num_verts).view(-1, 1)
        global_cluster_idx = (cluster_idx + batch_offset).view(-1)
        
        # Total clusters (vertices) in the batch
        total_clusters = B * num_verts
        
        # --- PointNet Pipeline ---
        
        # 1. Layer 1
        x = self.conv1(flat_points) # (B*P, 64)
        
        # 2. Feature Transform
        trans_feat = None
        if self.use_feature_transform:
            x, trans_feat = self.fstn(x, global_cluster_idx, total_clusters)
            
        # 3. Layer 2 & 3
        x = self.conv2(x) # (B*P, hidden)
        point_feats = self.conv3(x) # (B*P, output)
        
        # 4. Scatter Max Pooling
        # aggregated_feats: (B*V, output)
        aggregated_feats, _ = scatter_max(
            point_feats, 
            global_cluster_idx, 
            dim=0, 
            dim_size=total_clusters
        )
        
        # Apply ReLU after pooling? Or before?
        # PointNet: conv(relu) -> maxpool. 
        # My conv3 has BN but no ReLU. Let's apply ReLU to point_feats before pooling?
        # Or apply after? Standard PointNet:
        # x = F.relu(bn(conv(x)))
        # x = max_pool(x)
        
        # So I should add ReLU to conv3 or apply it here.
        # Let's apply it here to point_feats before max pooling.
        # Actually scatter_max on ReLU-ed features is safe (min is 0).
        point_feats = self.relu(point_feats)
        aggregated_feats, _ = scatter_max(
            point_feats, 
            global_cluster_idx, 
            dim=0, 
            dim_size=total_clusters
        )
        
        # 5. Reshape back to Batch
        # (B*V, D) -> (B, V, D)
        vertex_features = aggregated_feats.view(B, num_verts, self.output_dim)
        
        return vertex_features, trans_feat
