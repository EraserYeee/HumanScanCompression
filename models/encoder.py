import torch
import torch.nn as nn
from torch_scatter import scatter_max, scatter_softmax
from .tnet import ScatterSTNkd, ScatterSTN3d

class LocalFeatureEncoder(nn.Module):
    """
    负责从分组后的局部点云中提取 Base Mesh 顶点的特征向量。
    
    Updated: Integrated ScatterSTNkd (Feature Transform) and ScatterSTN3d (Input Transform)
    
    hidden_dim: int 或 list[int]。int 等价于 [int]，list 表示多层隐藏层宽度。
    """
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=128, use_feature_transform=True):
        super().__init__()
        self.output_dim = output_dim
        self.use_feature_transform = use_feature_transform
        
        hidden_dims = [hidden_dim] if isinstance(hidden_dim, int) else list(hidden_dim)
        
        # 0. Input Transform (STN3d)
        if self.use_feature_transform:
            self.istn = ScatterSTN3d()
        
        # 1. First Layer (Input -> 64)
        self.conv1 = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # 2. Feature Transform (STNkd)
        if self.use_feature_transform:
            self.fstn = ScatterSTNkd(k=64)
            
        # 3. Subsequent Layers: 64 -> hidden_dims[0] -> ... -> hidden_dims[-1] -> output_dim
        self.conv2 = nn.Sequential(
            nn.Linear(64, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU()
        )
        
        self.conv_extra = nn.ModuleList()
        for i in range(1, len(hidden_dims)):
            self.conv_extra.append(nn.Sequential(
                nn.Linear(hidden_dims[i - 1], hidden_dims[i]),
                nn.BatchNorm1d(hidden_dims[i]),
                nn.ReLU()
            ))
        
        self.conv3 = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim),
            nn.BatchNorm1d(output_dim)
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
        x = flat_points
        
        # 0. Input Transform
        if self.use_feature_transform:
            # We don't regularize input transform usually
            x, _ = self.istn(x, global_cluster_idx, total_clusters)
        
        # 1. Layer 1
        x = self.conv1(x) # (B*P, 64)
        
        # 2. Feature Transform
        trans_feat = None
        if self.use_feature_transform:
            x, trans_feat = self.fstn(x, global_cluster_idx, total_clusters)
            
        # 3. Layer 2 & extra & 3
        x = self.conv2(x)
        for extra_layer in self.conv_extra:
            x = extra_layer(x)
        point_feats = self.conv3(x)
        
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


class AttentiveLocalFeatureEncoder(nn.Module):
    """
    Multi-Head Attention Pooling 版本的局部特征编码器。
    使用 scatter_softmax 实现变长 cluster 的注意力加权聚合，
    替代原始 LocalFeatureEncoder 中的 Max Pooling。
    
    可选: 同时保留 Max Pooling 作为互补信号 (use_max_pool_residual=True)。
    
    hidden_dim: int 或 list[int]。int 等价于 [int]，list 表示多层隐藏层宽度。
    """
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=128, 
                 num_attention_heads=4, use_max_pool_residual=True, 
                 attention_temperature=1.0, score_init_scale=0.1,
                 score_clip_value=5.0):
        super().__init__()
        self.output_dim = output_dim
        self.num_heads = num_attention_heads
        self.use_max_pool_residual = use_max_pool_residual
        self.attention_temperature = attention_temperature
        self.score_clip_value = score_clip_value
        assert output_dim % num_attention_heads == 0, \
            f"output_dim ({output_dim}) must be divisible by num_attention_heads ({num_attention_heads})"
        
        hidden_dims = [hidden_dim] if isinstance(hidden_dim, int) else list(hidden_dim)
        
        # === Backbone (same structure as LocalFeatureEncoder, no T-Net) ===
        self.conv1 = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Linear(64, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU()
        )
        self.conv_extra = nn.ModuleList()
        for i in range(1, len(hidden_dims)):
            self.conv_extra.append(nn.Sequential(
                nn.Linear(hidden_dims[i - 1], hidden_dims[i]),
                nn.BatchNorm1d(hidden_dims[i]),
                nn.ReLU()
            ))
        self.conv3 = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim),
            nn.BatchNorm1d(output_dim)
        )
        self.relu = nn.ReLU()
        
        # === Multi-Head Attention Pooling ===
        # Score network: per-point -> H attention logits
        self.score_linear = nn.Linear(output_dim, num_attention_heads)
        # Value projection: per-point -> output_dim (split into H heads internally)
        self.value_linear = nn.Linear(output_dim, output_dim)
        
        # Initialize score_linear with smaller weights to prevent extreme logits
        # This helps avoid softmax saturation in early training
        with torch.no_grad():
            # Scale down the default initialization
            self.score_linear.weight.data *= score_init_scale
            # Initialize bias to small positive values to avoid all-negative logits
            # This helps create a more balanced initial attention distribution
            # Using a larger bias multiplier to push initial logits closer to zero
            self.score_linear.bias.data.zero_()
            self.score_linear.bias.data += 0.3 * score_init_scale  # Positive bias to balance logits
        
        # Projection for combining Attention + Max pooling
        if use_max_pool_residual:
            self.projection = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.ReLU()
            )

    def forward(self, local_points: torch.Tensor, cluster_idx: torch.Tensor, num_verts: int, 
                return_attention=False, return_diagnostics=False):
        """
        Args:
            local_points: (B, P, D) 局部坐标点 (可能含法线拼接, D=3 or 6)
            cluster_idx: (B, P) 点归属索引, 值域 [0, V-1]
            num_verts: int 最大顶点数 V
            return_attention: bool, 是否返回注意力分数用于可视化
            return_diagnostics: bool, 是否返回诊断信息（用于调试注意力机制）

        Returns:
            vertex_features: (B, V, output_dim) 每个 Base Mesh 顶点的特征
            trans_feat: None (保持 API 兼容)
            attention_scores: (B*P, H) or None, 每个点在每个头的注意力权重
            global_cluster_idx: (B*P,) or None, 全局 cluster 索引
            diagnostics: dict or None, 诊断信息
        """
        B, P, D = local_points.shape
        flat_points = local_points.view(-1, D)
        
        # Global cluster indices with batch offset
        batch_offset = (torch.arange(B, device=local_points.device) * num_verts).view(-1, 1)
        global_cluster_idx = (cluster_idx + batch_offset).view(-1)  # (B*P,)
        total_clusters = B * num_verts
        
        # === Backbone ===
        x = self.conv1(flat_points)    # (B*P, 64)
        x = self.conv2(x)             # (B*P, hidden_dims[0])
        for extra_layer in self.conv_extra:
            x = extra_layer(x)
        point_feats = self.conv3(x)   # (B*P, output_dim)
        point_feats = self.relu(point_feats)
        
        # === Multi-Head Attention Pooling ===
        # 1. Compute attention logits
        scores = self.score_linear(point_feats)  # (B*P, H)
        
        # 2. Clip logits to prevent extreme values (prevents weight explosion)
        # This helps maintain stable training and prevents softmax saturation
        if self.score_clip_value > 0:
            scores = torch.clamp(scores, min=-self.score_clip_value, max=self.score_clip_value)
        
        # 3. Apply temperature scaling to reduce softmax saturation
        # T > 1 makes distribution smoother, T < 1 makes it sharper
        scores_scaled = scores / self.attention_temperature
        
        # 4. Per-cluster softmax (scatter_softmax handles variable-size groups)
        alpha = scatter_softmax(scores_scaled, global_cluster_idx, dim=0)  # (B*P, H)
        
        # 3. Value projection + multi-head reshape
        values = self.value_linear(point_feats)  # (B*P, output_dim)
        H = self.num_heads
        D_head = self.output_dim // H
        values_mh = values.view(-1, H, D_head)  # (B*P, H, D/H)
        
        # 4. Weighted aggregation per head
        alpha_exp = alpha.unsqueeze(-1)          # (B*P, H, 1)
        weighted = (alpha_exp * values_mh).view(-1, self.output_dim)  # (B*P, output_dim)
        # Keep scatter_add operands in the same dtype under AMP/bf16.
        weighted = weighted.to(point_feats.dtype)
        
        # Scatter sum (using PyTorch built-in for reliability)
        attn_feats = torch.zeros(total_clusters, self.output_dim,
                                 device=flat_points.device, dtype=weighted.dtype)
        attn_feats.scatter_add_(0, 
            global_cluster_idx.unsqueeze(-1).expand(-1, self.output_dim), 
            weighted)
        
        # === Optional: combine with Max Pooling ===
        if self.use_max_pool_residual:
            max_feats, _ = scatter_max(point_feats, global_cluster_idx, 
                                       dim=0, dim_size=total_clusters)
            max_feats = max_feats.to(attn_feats.dtype)
            combined = torch.cat([attn_feats, max_feats], dim=-1)  # (B*V, 2*output_dim)
            aggregated_feats = self.projection(combined)           # (B*V, output_dim)
        else:
            aggregated_feats = attn_feats
        
        # Reshape to batch
        vertex_features = aggregated_feats.view(B, num_verts, self.output_dim)
        
        # Collect diagnostics if requested
        diagnostics = None
        if return_diagnostics:
            # Compute statistics for each head
            diagnostics = {
                'point_feats': {
                    'mean': point_feats.mean().item(),
                    'std': point_feats.std().item(),
                    'min': point_feats.min().item(),
                    'max': point_feats.max().item(),
                },
                'scores': {  # Before softmax (logits, original scale)
                    'mean': scores.mean().item(),
                    'std': scores.std().item(),
                    'min': scores.min().item(),
                    'max': scores.max().item(),
                },
                'scores_scaled': {  # After temperature scaling (before softmax)
                    'mean': scores_scaled.mean().item(),
                    'std': scores_scaled.std().item(),
                    'min': scores_scaled.min().item(),
                    'max': scores_scaled.max().item(),
                },
                'attention_scores': {  # After softmax (alpha)
                    'mean': alpha.mean().item(),
                    'std': alpha.std().item(),
                    'min': alpha.min().item(),
                    'max': alpha.max().item(),
                },
                'score_linear_grad_norm': None,  # Will be filled in training loop
                'num_points': point_feats.shape[0],
                'num_clusters': total_clusters,
            }
            # Per-head statistics for attention scores
            for h in range(self.num_heads):
                alpha_h = alpha[:, h]
                diagnostics[f'head_{h}'] = {
                    'mean': alpha_h.mean().item(),
                    'std': alpha_h.std().item(),
                    'min': alpha_h.min().item(),
                    'max': alpha_h.max().item(),
                }
        
        if return_attention:
            if return_diagnostics:
                return vertex_features, None, alpha, global_cluster_idx, diagnostics
            else:
                return vertex_features, None, alpha, global_cluster_idx
        else:
            if return_diagnostics:
                return vertex_features, None, diagnostics
            else:
                return vertex_features, None  # None for trans_feat (API compatibility)


class VAEHead(nn.Module):
    """
    变分自编码器 (VAE) 瓶颈头。
    
    解耦设计: 可附加到任意编码器输出之后，将确定性特征映射为
    参数化的高斯分布 (μ, σ)，通过重参数化采样得到隐变量 z。
    
    训练时从 q(z|x) = N(μ, σ²I) 采样；推理时直接使用 μ。
    """
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.mu_linear = nn.Linear(input_dim, latent_dim)
        self.logvar_linear = nn.Linear(input_dim, latent_dim)
    
    def forward(self, x):
        """
        Args:
            x: (B, V, D) 编码器输出的顶点特征
            
        Returns:
            z: (B, V, latent_dim) 采样/均值 隐变量
            kl_loss: scalar, KL 散度损失 D_KL(q(z|x) || N(0,I))
        """
        mu = self.mu_linear(x)           # (B, V, latent_dim)
        log_var = self.logvar_linear(x)  # (B, V, latent_dim)
        
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu
        
        # KL divergence: -0.5 * E[1 + log(σ²) - μ² - σ²]
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        
        return z, kl_loss
