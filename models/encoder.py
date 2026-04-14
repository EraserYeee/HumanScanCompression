import torch
import torch.nn as nn
from torch_scatter import scatter_max, scatter_softmax
from .tnet import ScatterSTNkd, ScatterSTN3d
from .timing_hooks import prof_start, prof_split


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.ReLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

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
        do_log = getattr(self, "_profile_do_log", False)
        t0 = prof_start() if do_log else None
        
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
        if do_log:
            t0 = prof_split(do_log, t0, "local_backbone", "enc")
        
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
        if do_log:
            prof_split(do_log, t0, "local_scatter_pool", "enc")
        
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
        do_log = getattr(self, "_profile_do_log", False)
        t0 = prof_start() if do_log else None
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
        if do_log:
            t0 = prof_split(do_log, t0, "attn_backbone", "enc")
        
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
        if do_log:
            prof_split(do_log, t0, "attn_pool_proj", "enc")
        
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


class CrossAttentionFeatureEncoder(nn.Module):
    """
    MaxPool-as-Query Cross-Attention encoder.
    
    Two-pass design:
    1. PointNet backbone → per-point features → MaxPool per cluster → coarse vertex features
    2. Coarse features as Q, per-point features as K/V → cluster-restricted cross-attention
    3. Residual: output = coarse (MaxPool) + attention refinement
    
    Q and K/V are all in the same PointNet feature space, so Q·K is semantically meaningful:
    "how relevant is this point's feature to the cluster's dominant feature?"
    """
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=128,
                 num_heads=4, use_ffn=True, **kwargs):
        super().__init__()
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.use_ffn = use_ffn
        assert output_dim % num_heads == 0, \
            f"output_dim ({output_dim}) must be divisible by num_heads ({num_heads})"
        self.dim_head = output_dim // num_heads
        self.scale = self.dim_head ** -0.5

        hidden_dims = [hidden_dim] if isinstance(hidden_dim, int) else list(hidden_dim)

        # === PointNet Backbone ===
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
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )

        # === Cross-Attention projections ===
        self.norm_q = nn.LayerNorm(output_dim)
        self.norm_kv = nn.LayerNorm(output_dim)
        self.to_q = nn.Linear(output_dim, output_dim, bias=False)
        self.to_k = nn.Linear(output_dim, output_dim, bias=False)
        self.to_v = nn.Linear(output_dim, output_dim, bias=False)
        self.to_out = nn.Linear(output_dim, output_dim)

        # === Optional FeedForward ===
        if use_ffn:
            self.norm_ffn = nn.LayerNorm(output_dim)
            self.ffn = FeedForward(output_dim)

    def forward(self, local_points, cluster_idx, num_verts, return_diagnostics=False, **kwargs):
        """
        Args:
            local_points: (B, P, D) per-point local coords (possibly with normals)
            cluster_idx:  (B, P) vertex assignment per point, values in [0, V-1]
            num_verts:    int, number of base mesh vertices V
            return_diagnostics: if True, return a small dict (ca_residual mean vs coarse)

        Returns:
            vertex_features: (B, V, output_dim)
            trans_feat: None (API compatibility)
            diagnostics: optional dict when return_diagnostics=True
        """
        B, P, D = local_points.shape
        V = num_verts

        # Global cluster indices with batch offset
        batch_offset = (torch.arange(B, device=local_points.device) * V).view(-1, 1)
        global_cluster_idx = (cluster_idx + batch_offset).view(-1)  # (B*P,)
        total_clusters = B * V

        # === Pass 1: PointNet Backbone + MaxPool ===
        x = self.conv1(local_points.view(-1, D))
        x = self.conv2(x)
        for extra_layer in self.conv_extra:
            x = extra_layer(x)
        point_feats = self.conv3(x)  # (B*P, output_dim)

        vertex_coarse, _ = scatter_max(
            point_feats, global_cluster_idx, dim=0, dim_size=total_clusters
        )  # (B*V, output_dim)

        # === Pass 2: Cross-Attention (Q=MaxPool, K/V=point_feats) ===
        H, D_h = self.num_heads, self.dim_head

        q = self.to_q(self.norm_q(vertex_coarse)).view(-1, H, D_h)   # (B*V, H, D_h)

        kv_normed = self.norm_kv(point_feats)
        k = self.to_k(kv_normed).view(-1, H, D_h)                    # (B*P, H, D_h)
        v = self.to_v(kv_normed).view(-1, H, D_h)                    # (B*P, H, D_h)

        # Gather each point's assigned vertex query
        q_per_point = q[global_cluster_idx]  # (B*P, H, D_h)

        # Q·K: "how relevant is this point to the cluster's dominant feature?"
        scores = (q_per_point * k).sum(dim=-1) * self.scale  # (B*P, H)

        # Per-cluster softmax
        alpha = scatter_softmax(scores, global_cluster_idx, dim=0)  # (B*P, H)

        # Weighted aggregation
        weighted = (alpha.unsqueeze(-1) * v).view(-1, self.output_dim)  # (B*P, output_dim)
        weighted = weighted.to(point_feats.dtype)

        attn_out = torch.zeros(total_clusters, self.output_dim,
                               device=point_feats.device, dtype=weighted.dtype)
        attn_out.scatter_add_(
            0, global_cluster_idx.unsqueeze(-1).expand(-1, self.output_dim), weighted
        )

        # Residual: MaxPool base + attention refinement (to_out branch only)
        refinement = self.to_out(attn_out)
        vertex_feats = vertex_coarse + refinement

        diagnostics = None
        if return_diagnostics:
            eps = 1e-8
            r_mean = refinement.abs().mean()
            c_mean = vertex_coarse.abs().mean()
            diagnostics = {
                'ca_residual_mean_abs': r_mean.item(),
                'ca_coarse_mean_abs': c_mean.item(),
                'ca_residual_to_coarse_ratio': (r_mean / (c_mean + eps)).item(),
            }

        # === Optional FeedForward + residual ===
        if self.use_ffn:
            vertex_feats = vertex_feats + self.ffn(self.norm_ffn(vertex_feats))

        vertex_features = vertex_feats.view(B, V, self.output_dim)
        if return_diagnostics:
            return vertex_features, None, diagnostics
        return vertex_features, None


class SinusoidalPositionEncoding(nn.Module):
    """Multi-frequency sinusoidal encoding for 3-D coordinates."""
    def __init__(self, num_frequencies: int = 4):
        super().__init__()
        self.num_frequencies = num_frequencies
        # output dim per coordinate = 2*num_frequencies  (sin + cos)
        # total output dim = 3 * 2 * num_frequencies
        freqs = 2.0 ** torch.arange(num_frequencies).float() * torch.pi
        self.register_buffer("freqs", freqs)  # (L,)

    @property
    def out_dim(self) -> int:
        return 3 * 2 * self.num_frequencies

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (..., 3)
        Returns:
            pe: (..., 3*2*L)
        """
        # coords: (..., 3) -> (..., 3, 1) * (L,) -> (..., 3, L)
        x = coords.unsqueeze(-1) * self.freqs  # (..., 3, L)
        pe = torch.cat([x.sin(), x.cos()], dim=-1)  # (..., 3, 2L)
        return pe.flatten(-2)  # (..., 3*2L)


class TransformerSABlock(nn.Module):
    """Pre-norm Transformer self-attention block with residual."""
    def __init__(self, dim: int, num_heads: int = 4, use_ffn: bool = True, ffn_mult: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

        self.use_ffn = use_ffn
        if use_ffn:
            self.norm2 = nn.LayerNorm(dim)
            self.ffn = nn.Sequential(
                nn.Linear(dim, dim * ffn_mult),
                nn.GELU(),
                nn.Linear(dim * ffn_mult, dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, S, D)  –  N sequences of length S
        Returns:
            x: (N, S, D)
        """
        H, D_h = self.num_heads, self.head_dim
        residual = x
        x_norm = self.norm1(x)
        N, S, D = x_norm.shape

        qkv = self.qkv(x_norm).reshape(N, S, 3, H, D_h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each (N, H, S, D_h)

        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)  # Flash-eligible
        x = x.transpose(1, 2).reshape(N, S, D)
        x = self.out_proj(x)
        x = residual + x

        if self.use_ffn:
            x = x + self.ffn(self.norm2(x))
        return x


def _farthest_point_sample(coords: torch.Tensor, n_sample: int) -> torch.Tensor:
    """Pure-PyTorch farthest-point sampling (no extra deps).

    Args:
        coords: (N, S, 3)
        n_sample: number of points to select

    Returns:
        idx: (N, n_sample) long
    """
    N, S, _ = coords.shape
    device = coords.device
    idx = torch.zeros(N, n_sample, dtype=torch.long, device=device)
    dists = torch.full((N, S), 1e10, device=device)
    farthest = torch.randint(0, S, (N,), device=device)
    batch_idx = torch.arange(N, device=device)
    for i in range(n_sample):
        idx[:, i] = farthest
        centroid = coords[batch_idx, farthest].unsqueeze(1)       # (N, 1, 3)
        d = ((coords - centroid) ** 2).sum(-1)                    # (N, S)
        dists = torch.min(dists, d)
        farthest = dists.argmax(-1)                               # (N,)
    return idx


def _knn_idx(src: torch.Tensor, query: torch.Tensor, k: int) -> torch.Tensor:
    """Brute-force KNN: for each query point find k nearest in src.

    Args:
        src:   (N, S, 3)
        query: (N, M, 3)
        k: int

    Returns:
        idx: (N, M, k) long – indices into src
    """
    d = torch.cdist(query, src)                                   # (N, M, S)
    return d.topk(k, dim=-1, largest=False).indices               # (N, M, k)


class VectorAttentionBlock(nn.Module):
    """Point Transformer v1 vector-attention with relative position MLP."""

    def __init__(self, dim: int):
        super().__init__()
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.pos_mlp = nn.Sequential(nn.Linear(3, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.attn_mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.out_proj = nn.Linear(dim, dim)

    def forward(
        self,
        query: torch.Tensor,
        neighbors: torch.Tensor,
        delta_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query:     (N, M, D)     representative points
            neighbors: (N, M, k, D)  neighbour features
            delta_pos: (N, M, k, 3)  relative coordinates (nbr - query)

        Returns:
            out: (N, M, D)
        """
        pe = self.pos_mlp(delta_pos)                              # (N, M, k, D)

        q = self.to_q(query).unsqueeze(2)                         # (N, M, 1, D)
        k = self.to_k(neighbors)                                  # (N, M, k, D)
        v = self.to_v(neighbors)                                  # (N, M, k, D)

        w = self.attn_mlp(q - k + pe)                             # (N, M, k, D)
        w = torch.softmax(w, dim=2)                               # softmax over k
        out = (w * (v + pe)).sum(dim=2)                           # (N, M, D)
        return self.out_proj(out) + query                         # residual


class TransitionDown(nn.Module):
    """FPS down-sample + KNN + VectorAttention."""

    def __init__(self, dim: int, n_sample: int, k: int = 16):
        super().__init__()
        self.n_sample = n_sample
        self.k = k
        self.attn = VectorAttentionBlock(dim)

    def forward(
        self, x: torch.Tensor, coords: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:      (N, S, D)  point features
            coords: (N, S, 3)  point coordinates

        Returns:
            x_down:      (N, n_sample, D)
            coords_down: (N, n_sample, 3)
        """
        with torch.no_grad():
            fps_idx = _farthest_point_sample(coords, self.n_sample)  # (N, M)
        N = x.shape[0]
        bi = torch.arange(N, device=x.device).unsqueeze(1)

        coords_down = coords[bi, fps_idx]                         # (N, M, 3)
        x_down = x[bi, fps_idx]                                   # (N, M, D)

        with torch.no_grad():
            knn_idx = _knn_idx(coords, coords_down, self.k)       # (N, M, k)
        nbr_feat = x[bi.unsqueeze(2), knn_idx]                    # (N, M, k, D)
        nbr_pos = coords[bi.unsqueeze(2), knn_idx]                # (N, M, k, 3)
        delta_pos = nbr_pos - coords_down.unsqueeze(2)            # (N, M, k, 3)

        x_down = self.attn(x_down, nbr_feat, delta_pos)           # (N, M, D)
        return x_down, coords_down


class PTSAEncoder(nn.Module):
    """Hierarchical Point Transformer encoder (PT v1 style).

    Three-stage FPS down-sampling with local vector-attention:
        512 pts → 128 → 32 → 1  (face feature)

    Internal dimension stays at sa_dim (default 128) throughout;
    a final linear projects to feature_dim (default 512).
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dims: list | None = None,
        sa_dim: int = 128,
        num_heads: int = 4,
        num_sa_layers: int = 1,
        feature_dim: int = 512,
        pooling_type: str = "attentive",
        use_ffn: bool = True,
        pe_num_frequencies: int = 4,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64]
        self.sa_dim = sa_dim
        self.feature_dim = feature_dim

        # --- Per-point MLP backbone ---
        layers: list[nn.Module] = []
        in_d = input_dim
        for h_d in hidden_dims:
            layers.extend([nn.Linear(in_d, h_d), nn.BatchNorm1d(h_d), nn.ReLU()])
            in_d = h_d
        layers.extend([nn.Linear(in_d, sa_dim), nn.BatchNorm1d(sa_dim), nn.ReLU()])
        self.backbone = nn.Sequential(*layers)

        # --- Hierarchical TransitionDown stages ---
        self.td1 = TransitionDown(sa_dim, n_sample=128, k=16)
        self.td2 = TransitionDown(sa_dim, n_sample=32, k=16)
        self.final_attn = VectorAttentionBlock(sa_dim)

        # --- Output projection ---
        self.out_proj = nn.Linear(sa_dim, feature_dim)

    def _hierarchy_chunk(
        self, x: torch.Tensor, coords: torch.Tensor
    ) -> torch.Tensor:
        """Run td1 → td2 → final_attn on a small chunk of face groups.

        Args:
            x:      (C, K, sa_dim)
            coords: (C, K, 3)
        Returns:
            pooled: (C, sa_dim)
        """
        x, coords = self.td1(x, coords)
        x, coords = self.td2(x, coords)
        center = coords.mean(dim=1, keepdim=True)
        query = x.mean(dim=1, keepdim=True)
        delta = coords.unsqueeze(1) - center.unsqueeze(2)
        nbr = x.unsqueeze(1)
        return self.final_attn(query, nbr, delta).squeeze(1)

    def forward(
        self,
        grouped_features: torch.Tensor,
        local_coords: torch.Tensor,
    ):
        """
        Args:
            grouped_features: (B, V, K, D_in)
            local_coords:     (B, V, K, 3)

        Returns:
            vertex_features: (B, V, feature_dim)
            trans_feat:      None  (API compat)
        """
        B, V, K, D_in = grouped_features.shape
        N = B * V

        # --- Per-point MLP (memory-safe, benefits from large BN batch) ---
        x = self.backbone(grouped_features.reshape(N * K, D_in))  # (N*K, sa_dim)
        x = x.reshape(N, K, self.sa_dim)
        coords = local_coords.reshape(N, K, 3)

        # --- Chunked hierarchical attention ---
        # VectorAttention creates (chunk, 128, 16, D) intermediates;
        # chunk_size=256 keeps peak VRAM ≈ 1.2 GB instead of 17+ GB.
        chunk_size = 256
        if N <= chunk_size:
            pooled = self._hierarchy_chunk(x, coords)
        else:
            parts = []
            for i in range(0, N, chunk_size):
                j = min(i + chunk_size, N)
                parts.append(self._hierarchy_chunk(x[i:j], coords[i:j]))
            pooled = torch.cat(parts, dim=0)                       # (N, sa_dim)

        # --- Project ---
        out = self.out_proj(pooled).reshape(B, V, self.feature_dim)
        return out, None


class FlashCrossAttnDown(nn.Module):
    """FPS down-sample + flash cross-attention (query=selected, key/value=ALL source).

    Unlike VectorAttention-based TransitionDown which explicitly gathers
    (N, M, k, D) neighbor tensors, this uses standard multi-head cross-attention
    via F.scaled_dot_product_attention (flash-eligible), keeping VRAM O(N·S·H)
    instead of O(N·M·k·D).
    """

    def __init__(self, dim: int, n_sample: int, num_heads: int = 4,
                 pe_num_frequencies: int = 4):
        super().__init__()
        self.n_sample = n_sample
        self.num_heads = num_heads
        assert dim % num_heads == 0
        self.head_dim = dim // num_heads

        self.pe = SinusoidalPositionEncoding(pe_num_frequencies)
        self.pe_proj = nn.Linear(self.pe.out_dim, dim)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(),
                                 nn.Linear(dim * 4, dim))

    def forward(
        self, x: torch.Tensor, coords: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        N, S, D = x.shape
        H, d = self.num_heads, self.head_dim
        M = self.n_sample

        with torch.no_grad():
            fps_idx = _farthest_point_sample(coords, M)
        bi = torch.arange(N, device=x.device).unsqueeze(1)

        coords_down = coords[bi, fps_idx]
        x_down = x[bi, fps_idx]

        pe_q = self.pe_proj(self.pe(coords_down))
        pe_kv = self.pe_proj(self.pe(coords))

        x_normed = self.norm_kv(x)
        q = self.to_q(self.norm_q(x_down) + pe_q)
        k = self.to_k(x_normed + pe_kv)
        v = self.to_v(x_normed)

        q = q.view(N, M, H, d).transpose(1, 2)
        k = k.view(N, S, H, d).transpose(1, 2)
        v = v.view(N, S, H, d).transpose(1, 2)

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(N, M, D)
        x_down = self.out_proj(out) + x_down

        x_down = self.ffn(self.norm_ffn(x_down)) + x_down
        return x_down, coords_down


class FlashPoolAttention(nn.Module):
    """Cross-attend from a mean-query to all points → single pooled feature."""

    def __init__(self, dim: int, num_heads: int = 4, pe_num_frequencies: int = 4):
        super().__init__()
        self.num_heads = num_heads
        assert dim % num_heads == 0
        self.head_dim = dim // num_heads

        self.pe = SinusoidalPositionEncoding(pe_num_frequencies)
        self.pe_proj = nn.Linear(self.pe.out_dim, dim)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        N, S, D = x.shape
        H, d = self.num_heads, self.head_dim

        query = x.mean(dim=1, keepdim=True)
        query_pos = coords.mean(dim=1, keepdim=True)

        pe_q = self.pe_proj(self.pe(query_pos))
        pe_kv = self.pe_proj(self.pe(coords))

        x_normed = self.norm_kv(x)
        q = self.to_q(self.norm_q(query) + pe_q).view(N, 1, H, d).transpose(1, 2)
        k = self.to_k(x_normed + pe_kv).view(N, S, H, d).transpose(1, 2)
        v = self.to_v(x_normed).view(N, S, H, d).transpose(1, 2)

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(N, D)
        return self.out_proj(out) + query.squeeze(1)


class PTFlashHierarchicalEncoder(nn.Module):
    """Hierarchical encoder: FPS downsampling + flash dot-product cross-attention.

    Same hierarchical structure as PTSAEncoder (512→128→32→1) but replaces
    VectorAttention with multi-head cross-attention via flash attention:
      - No explicit (N, M, k, D) neighbor tensors → ~20x less VRAM
      - Attends to ALL source points per stage (not just k=16) → wider receptive field
      - No chunking needed
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dims: list | None = None,
        sa_dim: int = 128,
        num_heads: int = 4,
        num_sa_layers: int = 1,
        feature_dim: int = 512,
        pooling_type: str = "attentive",
        use_ffn: bool = True,
        pe_num_frequencies: int = 4,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64]
        self.sa_dim = sa_dim
        self.feature_dim = feature_dim

        layers: list[nn.Module] = []
        in_d = input_dim
        for h_d in hidden_dims:
            layers.extend([nn.Linear(in_d, h_d), nn.BatchNorm1d(h_d), nn.ReLU()])
            in_d = h_d
        layers.extend([nn.Linear(in_d, sa_dim), nn.BatchNorm1d(sa_dim), nn.ReLU()])
        self.backbone = nn.Sequential(*layers)

        self.td1 = FlashCrossAttnDown(sa_dim, n_sample=128, num_heads=num_heads,
                                      pe_num_frequencies=pe_num_frequencies)
        self.td2 = FlashCrossAttnDown(sa_dim, n_sample=32, num_heads=num_heads,
                                      pe_num_frequencies=pe_num_frequencies)
        self.pool = FlashPoolAttention(sa_dim, num_heads=num_heads,
                                       pe_num_frequencies=pe_num_frequencies)

        self.out_proj = nn.Linear(sa_dim, feature_dim)

    def forward(
        self,
        grouped_features: torch.Tensor,
        local_coords: torch.Tensor,
    ):
        B, V, K, D_in = grouped_features.shape
        N = B * V

        x = self.backbone(grouped_features.reshape(N * K, D_in))
        x = x.reshape(N, K, self.sa_dim)
        coords = local_coords.reshape(N, K, 3)

        x, coords = self.td1(x, coords)
        x, coords = self.td2(x, coords)
        pooled = self.pool(x, coords)

        out = self.out_proj(pooled).reshape(B, V, self.feature_dim)
        return out, None


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
