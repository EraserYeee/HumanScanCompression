# Technical Report: Two-Stage Neural Mesh Compression for Human Scans

## 1. Pipeline Overview

本系统将高分辨率人体扫描（~819k 点+法线）压缩为一个粗糙 base mesh（~2000 面）加上每面编码特征（per-face latent），经细分解码后重建出百万量级顶点的 fine mesh。总体数据流如下：

```
┌─────────────────────── Stage 1 (Offline) ──────────────────────┐
│  GT Scan Mesh                                                  │
│      │                                                         │
│      ▼                                                         │
│  Curvature-Adaptive QEM Simplification                         │
│      │  compute absolute curvature |k1|+|k2|                   │
│      │  smooth curvature → vertex quality                      │
│      │  QEM decimation with qualityweight=True                 │
│      ▼                                                         │
│  base_verts (V, 3)    base_faces (F, 3)   [F ≈ 2000–3500]     │
└────────────────────────────────────────────────────────────────┘

┌─────────────────────── Stage 2 (Neural) ──────────────────────┐
│                                                                │
│  scan_points (1, P=819200, 6)   base_mesh (V, F)              │
│      │                              │                          │
│      ▼                              ▼                          │
│  ┌──────────────────────────────────────┐                      │
│  │  FaceKNNGrouper                      │                      │
│  │    centroid_f = v0 + (e1+e2)/3       │                      │
│  │    for each face: KNN(centroid, P)   │                      │
│  │    → (u, v, d) triangle parametric   │                      │
│  │  Output: (1, F, K=512, 6)           │                      │
│  └──────────────┬───────────────────────┘                      │
│                 │                                               │
│                 ▼                                               │
│  ┌──────────────────────────────────────┐                      │
│  │  PTSAEncoder                         │                      │
│  │    MLP: 6→64→128 (BN+ReLU)          │                      │
│  │    + SinPE(3→24) → concat → 128     │                      │
│  │    TransformerSA (4-head, FFN)       │                      │
│  │    Attentive Pooling → 128           │                      │
│  │    Linear → 512                      │                      │
│  │  Output: face_features (1, F, 512)   │                      │
│  └──────────────┬───────────────────────┘                      │
│                 │                                               │
│                 ▼                                               │
│  ┌──────────────────────────────────────┐                      │
│  │  FaceTriangleDecoder (ortho_frame)   │                      │
│  │    Subdivide: rate=14 → K=105/face   │                      │
│  │    MLP([feat, PE(u,v)]) → (d1,d2,d3) │                      │
│  │    disp = d1·ê1 + d2·ê2⊥ + d3·n     │                      │
│  │    SeamStitch: scatter_mean          │                      │
│  │  Output: fine_verts (1, ~210k, 3)    │                      │
│  └──────────────────────────────────────┘                      │
│                                                                │
│  Training Loss: differentiable normal-map rendering            │
│    L = w_n1·L1(N) + w_ns·SSIM(N) + w_ng·∇N + w_d·depth       │
│        + w_m·mask + w_lap·Laplacian                            │
└────────────────────────────────────────────────────────────────┘
```

当前配置 (`configs/config_ptsa.yaml`)：

| 组件 | 选择 | 关键参数 |
|------|------|----------|
| Base Mesh | Curvature-Adaptive QEM | 2000–3500 faces |
| Grouper | FaceKNNGrouper (vertex_knn mode) | K=512, chunk=512 |
| Encoder | PTSAEncoder (pt_sa_attentive) | sa_dim=128, heads=4, layers=1 |
| Decoder | FaceTriangleDecoder | ortho_frame, rate=14, PE levels=8 |
| Feature dim | 512 | — |
| VAE | Disabled | — |
| Rendering | 6 views, 512×512 | chunk=3 |

---

## 2. Stage 1: Curvature-Adaptive QEM Simplification

### 2.1 动机

Stage 2 的性能高度依赖 base mesh 质量。面锚编码 (face-anchored encoding) 的工作原理是将 scan 点投影到三角面的参数空间 `(u, v, d)`，其中 `d` 是法线方向的偏移。如果某区域（如手指、面部）的 base mesh 过于稀疏，该区域的所有 scan 点被"挤"进少数几个大三角形的参数空间，编码分辨率不足，导致重建质量下降。

**核心目标**：让高曲率区域（手指、面部、褶皱）拥有更多、更小的三角面，低曲率区域（躯干、四肢平坦处）拥有更少、更大的三角面。

### 2.2 算法

```python
def simplify_mesh_adaptive(V, F, target_faces):
    """
    Input:  V ∈ R^{N×3}, F ∈ Z^{M×3}, target_faces ∈ Z
    Output: V' ∈ R^{N'×3}, F' ∈ Z^{M'×3}  where M' ≈ target_faces
    """
    # Step 1: Compute discrete absolute curvature per vertex
    #   κ_abs(v) = |κ_1(v)| + |κ_2(v)|
    #   where κ_1, κ_2 are principal curvatures estimated from
    #   the discrete Laplace-Beltrami operator (angle defect)
    q = discrete_absolute_curvature(V, F)      # q ∈ R^N, q >= 0

    # Step 2: Laplacian smoothing of curvature field (10 iterations)
    #   Reduces noise from scan artifacts while preserving curvature structure
    for _ in range(10):
        q = laplacian_smooth_scalar(q, V, F)   # q_new(v) = mean(q(N(v)))

    # Step 3: Normalize quality distribution
    #   - Clamp outliers at 95th percentile
    #   - Normalize to [0, 1]
    #   - sqrt spread + floor at 0.1
    p95 = percentile(q, 95)
    q = clamp(q, 0, p95) / p95
    q = 0.1 + 0.9 * sqrt(q)                   # q ∈ [0.1, 1.0]

    # Step 4: Quality-weighted QEM decimation
    #   Edge collapse cost = Q_geometric(e) × (1 + w_quality × q(v))
    #   High quality (high curvature) → high cost → preserved
    V', F' = QEM_decimate(V, F, target_faces,
                          vertex_quality=q,
                          quality_weight=True,
                          preserve_topology=True)
    return V', F'
```

与标准 QEM 的区别仅在 Step 4：standard QEM 的边折叠代价完全由几何误差二次型决定，而 quality-weighted QEM 将代价乘以顶点曲率权重。效果是：低曲率区域的边更"便宜"，优先被折叠；高曲率区域的边更"昂贵"，倾向被保留。

---

## 3. Stage 2: Face-Anchored Neural Subdivision

### 3.1 FaceKNNGrouper — 面锚局部编组

对于 base mesh 的每个三角面 `f`，寻找 scan 中距面心最近的 `K=512` 个点，并将它们转换到该面的三角参数坐标系 `(u, v, d)` 中。

```python
def FaceKNNGrouper_forward(base_verts, base_faces, scan_points, scan_normals):
    """
    Input:
      base_verts:  (B, V, 3)        base mesh 顶点
      base_faces:  (B, F, 3)        base mesh 面索引
      scan_points: (B, P=819200, 3) 稠密 scan 点云
      scan_normals:(B, P, 3)        法线 (optional)

    Output:
      grouped_features: (B, F, K, D)  D=3 (u,v,d) or 6 (+normals)
      local_coords:     (B, F, K, 3)  same as (u,v,d) part
    """
    # 1. Compute face basis
    v0, v1, v2 = base_verts[base_faces[:,:,0]], ...[:,:,1], ...[:,:,2]
    e1 = v1 - v0                          # (B, F, 3) edge vector
    e2 = v2 - v0                          # (B, F, 3) edge vector
    n  = normalize(e1 × e2)               # (B, F, 3) unit face normal

    # Gram matrix for parametric inversion
    g11 = dot(e1, e1)                     # (B, F, 1)
    g12 = dot(e1, e2)
    g22 = dot(e2, e2)
    det = g11 * g22 - g12²               # clamped ≥ 1e-8

    # 2. Face centroids
    centroid = v0 + (e1 + e2) / 3         # (B, F, 3)

    # 3. Chunked KNN: each face centroid → K nearest scan points
    #    Uses chunked cdist + topk to avoid O(F×P) full distance matrix
    idx = chunked_topk_knn(centroid, scan_points, K=512, chunk=512)
    #    idx: (B, F, K)

    # 4. Gather and convert to parametric coordinates
    gathered = scan_points[idx]            # (B, F, K, 3)
    delta = gathered - v0.unsqueeze(2)     # offset from v0

    # Solve for (u, v) via Gram matrix inverse:
    #   delta = u·e1 + v·e2 + d·n
    d1 = dot(delta, e1)                    # projection onto e1
    d2 = dot(delta, e2)                    # projection onto e2
    d_val = dot(delta, n)                  # projection onto normal

    u = (g22 * d1 - g12 * d2) / det       # parametric u
    v = (g11 * d2 - g12 * d1) / det       # parametric v
    # d_val is the signed normal distance

    local_coords = stack([u, v, d_val])    # (B, F, K, 3)

    if scan_normals is not None:
        gathered_n = scan_normals[idx]     # (B, F, K, 3)
        delta_n = gathered_n - n.unsqueeze(2)
        grouped_features = cat([local_coords, delta_n], dim=-1)  # (B, F, K, 6)
    else:
        grouped_features = local_coords    # (B, F, K, 3)

    return grouped_features, local_coords
```

**关键设计**："vertex_knn" 模式表示 **锚点侧** KNN——即从锚点（面心）出发搜索最近邻，而非从 scan 点出发。这确保每个面获得恰好 `K` 个邻居，产生固定大小的张量 `(B, F, K, D)`，可直接送入 batched self-attention，无需 scatter 操作或 padding。

### 3.2 PTSAEncoder — Point Transformer Self-Attention

```python
def PTSAEncoder_forward(grouped_features, local_coords):
    """
    Input:
      grouped_features: (B, F, K=512, D_in=6)  triangle parametric + normals
      local_coords:     (B, F, K, 3)           (u, v, d) for position encoding

    Output:
      face_features: (B, F, feature_dim=512)
    """
    B, F, K, D_in = grouped_features.shape

    # 1. Per-point MLP backbone (shared across all points)
    x = grouped_features.reshape(B*F*K, D_in)
    x = Linear(D_in, 64) → BN1d → ReLU            # (B*F*K, 64)
    x = Linear(64, sa_dim=128) → BN1d → ReLU       # (B*F*K, 128)
    x = x.reshape(B*F, K, 128)

    # 2. Sinusoidal Positional Encoding
    #    coords (u,v,d) → sin/cos at 4 frequency bands
    #    PE(c) = [sin(π·c), cos(π·c), sin(2π·c), cos(2π·c),
    #             sin(4π·c), cos(4π·c), sin(8π·c), cos(8π·c)]
    pe = SinPE(local_coords)                        # (B*F, K, 3×2×4=24)
    x = Linear(128 + 24, 128)(cat([x, pe]))         # (B*F, K, 128)

    # 3. Transformer Self-Attention (1 layer, 4 heads)
    #    Pre-norm architecture with residual
    residual = x
    x = LayerNorm(x)
    Q, K_, V_ = Linear(128, 384)(x).split(128, dim=-1)  # 3 × (B*F, K, 128)
    Q, K_, V_ = reshape_to_heads(Q, K_, V_, num_heads=4)
    #    Each head: dim = 128/4 = 32
    x = scaled_dot_product_attention(Q, K_, V_)     # Flash Attention eligible
    x = reshape_from_heads(x)                       # (B*F, K, 128)
    x = Linear(128, 128)(x) + residual

    # FFN block
    x = x + FFN(LayerNorm(x))
    #   FFN: Linear(128, 512) → GELU → Linear(512, 128)

    # 4. Attentive Pooling
    scores = Linear(128, 1)(x).squeeze(-1)          # (B*F, K)
    alpha = softmax(scores, dim=-1)                  # (B*F, K)
    pooled = sum(alpha.unsqueeze(-1) * x, dim=1)    # (B*F, 128)

    # 5. Output projection
    face_features = Linear(128, 512)(pooled)         # (B*F, 512)
    face_features = face_features.reshape(B, F, 512)

    return face_features
```

### 3.3 FaceTriangleDecoder — 面局部坐标位移预测

```python
def FaceTriangleDecoder_forward(base_verts, base_faces, face_features, ortho_frame):
    """
    Input:
      base_verts:    (B, V, 3)
      base_faces:    (B, F, 3)
      face_features: (B, F, 512)     per-face latent from encoder
      ortho_frame:   bool             orthonormal vs parametric basis

    Output:
      fine_verts: (B, V_fine, 3)
      fine_faces: (F_fine, 3)
      displacements: (B, V_fine, 3)
    """
    # 1. Compute face basis vectors
    v0, e1, e2, n = compute_face_basis(base_verts, base_faces)

    # 2. Choose displacement basis
    if ortho_frame:
        b1 = normalize(e1)                   # ê₁: unit vector along edge 1
        b2 = cross(n, b1)                    # ê₂⊥: perpendicular to n and ê₁
        # {b1, b2, n} forms an orthonormal frame per face
    else:
        b1 = e1                              # raw parametric direction
        b2 = e2

    # 3. Barycentric subdivision
    #    rate=14 → K = 14×15/2 = 105 points per face
    #    Generate uniform samples in barycentric coords:
    #      (A, B) where A + B ≤ 1, C = 1 - A - B
    #    Parametric coords: u = B, v = C = 1 - A - B
    uv_A, uv_B = sample_uniform_bary(rate=14, F)
    K = 105  # points per face
    u_param = uv_B.reshape(F, K)             # (F, K)
    v_param = (1 - uv_A - uv_B).reshape(F, K)

    # 4. Base subdivision positions (in world space, uses original e1, e2)
    #    p_base = v0 + u·e1 + v·e2
    lp = v0[:,:,None,:] + u[:,:,:,None]*e1[:,:,None,:] + v[:,:,:,None]*e2[:,:,None,:]
    #    lp: (B, F, K, 3)

    # 5. Fourier positional encoding of parametric coords
    uv = stack([u_param, v_param], dim=-1)   # (F, K, 2)
    pe = PE_2D(uv, levels=8)                 # (F, K, 2×2×8=32)
    #   PE_2D(x) = [sin(π·x), cos(π·x), sin(2π·x), cos(2π·x), ..., sin(128π·x), cos(128π·x)]

    # 6. MLP prediction per subdivision point
    feat = face_features[:,:,None,:].expand(-1,-1,K,-1)  # (B, F, K, 512)
    mlp_in = cat([feat, pe], dim=-1)                      # (B, F, K, 544)

    out = MLP(mlp_in)                        # 544 → 256 → 128 → 64 → 3
    # MLP uses LeakyReLU activations
    # Last layer initialized near-zero: weight ~ U(-1e-5, 1e-5), bias = 0
    d1, d2, d3 = out[..., 0], out[..., 1], out[..., 2]

    # 7. Displacement in world coordinates
    disp_world = d1[:,:,:,None]*b1[:,:,None,:] \
               + d2[:,:,:,None]*b2[:,:,None,:] \
               + d3[:,:,:,None]*n[:,:,None,:]  # (B, F, K, 3)

    # 8. Seam stitching
    #    Subdivision points on shared edges/vertices are duplicated across
    #    adjacent faces. scatter_mean averages their displacements for C0 continuity.
    disp_flat = disp_world.reshape(B, F*K, 3)
    merge_idx = compute_merge_indices(base_faces, rate=14)
    #   merge_idx: (B, F*K) → unique point indices [0, V_fine-1]
    disp_stitched = scatter_mean(disp_flat, merge_idx)

    # 9. Final vertex positions
    lp_flat = lp.reshape(B, F*K, 3)
    fine_verts = lp_flat + disp_stitched[merge_idx]

    # 10. Build fine mesh topology
    #     Each base face subdivided into (rate-1)² ≈ 169 sub-faces
    fine_faces = build_subdivided_faces(rate=14, F)

    return fine_verts, fine_faces, disp_stitched
```

### 3.4 Seam Stitching 详解

Barycentric 细分将每个面独立细分为 `K=105` 个点，但相邻面在共享边上有重复的细分点。Seam stitching 通过唯一标识合并这些重复点：

```
      v0 ────────── v1          对于 edge v0-v1:
      │ ╲  face A  ╱ │          rate=14 → 边上有 15 个点 (含端点)
      │  ╲        ╱  │          face A 和 face B 各独立生成了这 15 个点
      │   ╲      ╱   │          它们具有相同的 raw_key = f(v0, v1, edge_position)
      │    ╲    ╱    │          scatter_mean 对相同 key 的位移取平均
      │     ╲  ╱     │
      │ face B╲╱     │          效果: 共享边上的位移被两个面的预测平均
      │       v2     │          → 自动保证 C0 连续性, 无需额外约束
```

标识规则：
- **顶点点**: key = vertex_index（全局唯一）
- **边上点**: key = `(min(v_i, v_j) × V_max + max(v_i, v_j)) × (rate+1) + edge_sub_idx`
  - 通过 min/max 排序确保两个面对同一条边使用相同的 key
- **面内点**: key = 面内唯一偏移（不与其他面共享）

### 3.5 输出规模

| 参数 | 值 | 说明 |
|------|------|------|
| Base faces `F` | 2000–3500 | 从 config 的 min/max 范围 |
| Subdivision rate | 14 | 每边 14 段 |
| Points per face `K` | 105 | `(14+1)×(14+2)/2 = 120`... 实际 `rate=14 → K = (rate×(rate+1))/2 = 105` |
| Sub-faces per face | ~169 | `(rate-1)² ≈ 169` |
| Total fine verts | ~150k–260k | `F × K` 减去共享点 |
| Total fine faces | ~338k–592k | `F × 169` |

---

## 4. Training Losses

### 4.1 可微分渲染监督

Stage 2 的主要监督信号来自可微分法线图渲染。每个训练 step 从随机视角渲染 predicted 和 GT mesh 的法线图 + 深度图 + mask，计算像素级损失。

```python
def compute_rendering_loss(fine_mesh, gt_mesh, num_views=6, image_size=512):
    """
    在 num_views 个随机相机视角下渲染并比较。
    view_chunk_size=3: 每次渲染 3 个视角, 累积梯度, 节省显存。
    """
    cameras = sample_random_cameras(num_views, dist_multiplier=0.4)

    L_total = 0
    for chunk in cameras.chunks(view_chunk_size=3):
        # Differentiable rasterization → normal map, depth, mask
        pred_normal, pred_depth, pred_mask = render(fine_mesh, chunk)
        gt_normal, gt_depth, gt_mask = render(gt_mesh, chunk)

        # AND-mask: only supervise where BOTH pred and GT have coverage
        valid = pred_mask & gt_mask

        L_total += (
            4.0 * L1(pred_normal, gt_normal, mask=valid)           # w_normal_l1
          + 0.5 * (1 - SSIM(pred_normal, gt_normal, mask=valid))   # w_normal_ssim
          + 4.0 * normal_gradient_loss(pred_normal, gt_normal, valid)  # w_normal_geo
          + 4.0 * L1(pred_depth, gt_depth, mask=valid)             # w_depth_l1
          + 1.0 * BCE(pred_mask, gt_mask)                          # w_mask
        )

    # Regularization
    L_total += 1.0 * laplacian_smoothness(fine_mesh)               # w_laplacian
    L_total += 0.001 * feature_transform_reg(trans_feat)           # w_mat (if T-Net)

    return L_total
```

**Normal Gradient Loss** (`w_normal_geo`): 计算法线图的空间梯度 `∂N/∂x`, `∂N/∂y` 的 L1 差异。这鼓励 fine mesh 在 GT 有 sharp feature 的地方也产生 sharp 的法线变化。**关键细节**：仅在 AND-mask 上计算，因为在 GT silhouette 外的区域，梯度 loss 没有"推回"的力，会导致 mesh 向外膨胀。

### 4.2 为什么不用 Chamfer Distance 作为主 Loss?

Config 中 `use_chamfer: false`。原因：
1. **计算开销**：对 ~210k fine verts × ~819k scan points 做 NN 搜索，每 step ~2s，而渲染 loss 仅 ~0.5s
2. **梯度质量**：Chamfer 的梯度方向是"最近点方向"，在密集区域多个点的梯度互相冲突；渲染 loss 的梯度通过像素空间传播，更全局一致
3. **缺乏细节监督**：Chamfer 只衡量距离，不感知法线/曲率；渲染 loss 直接监督视觉效果

---

## 5. Discussion

### 5.1 为什么 PTSA 使用 Vertex KNN？改回 Scan KNN 的可行性

**当前选择**: "vertex_knn" (实际为 FaceKNNGrouper) — 每个面心搜索 K=512 个最近 scan 点。

**对比 scan_knn**: (FaceLocalGrouper) — 每个 scan 点搜索 1 个最近面心，然后用 scatter 聚合到面。

选择 vertex_knn 的原因：

1. **固定大小张量**: FaceKNNGrouper 输出 `(B, F, K, D)` 是规则张量，可直接送入 batched self-attention（TransformerSABlock 的 `scaled_dot_product_attention` 要求固定序列长度）。而 scan_knn 每个面分到的点数不等（有的面几十个，有的面上万个），需要 padding 或 scatter 操作。

2. **Self-Attention 的二次复杂度**: Transformer 的复杂度是 `O(S²·d)`，其中 S 是序列长度。vertex_knn 中 S=K=512 是受控的；scan_knn 中 S 是不固定的（取决于面的大小和 scan 密度），最大可达数千，会爆显存。

3. **信息覆盖**: K=512 足以覆盖面周围的局部几何细节。scan_knn 模式中，小面可能只分到很少的点（<100），信息不足。

**能否兼顾 scan_knn 的优势？**

scan_knn 的优势在于：每个 scan 点恰好贡献一次，无重复无遗漏（vertex_knn 中同一个 scan 点可能被多个面重复选中）。如果想恢复 scan_knn，以下方案可兼顾显存/速度：

| 方案 | 思路 | 显存 | 速度 |
|------|------|------|------|
| Scatter + MaxPool (现有 baseline) | 每个 scan 点 → 最近面 → scatter_max 聚合 | 低 | 快 |
| Scatter + Attention | scan_knn 分配后，per-face attention pooling（需 padding 到 K_max 或用 packed sequence） | 中 | 中 |
| 两阶段混合 | 先 scan_knn 做粗聚合 → 再 vertex_knn 取 top-K 做 SA | 高 | 慢 |

**实际建议**：当前 vertex_knn + PTSA 的效果已经优于 scan_knn + scatter 方案。切换回 scan_knn 的主要动机是避免信息重复，但 PTSA 的 attention 机制已经能自适应地加权这些重复点（给已被多个面看到的点更低的 attention weight），因此收益有限。如果确实想尝试，推荐 "Scatter + Attention" 方案：先用 scan_knn 分配（零额外显存），然后对每个面的局部点集做 padding-free attention（使用 `torch.nn.functional.scaled_dot_product_attention` 的 `attn_mask` 参数处理变长序列）。

### 5.2 为什么 Face Coding 优于 Vertex Coding？"归一化"的意义

#### Face Coding vs Vertex Coding

**Vertex Coding** (LocalPatchGrouper + SumOfFeatureDecoder):
```
scan 点 → 最近 vertex → 局部坐标 (Δx, Δy, Δz)
→ scatter 聚合到 vertex feature
→ 3 个 vertex feature 重心插值 → 细分点 feature
→ MLP 预测标量位移 d → fine_vert = base_pos + d·normal
```

**Face Coding** (FaceKNNGrouper + FaceTriangleDecoder):
```
scan 点 → 最近 face → 三角参数坐标 (u, v, d)
→ attention 聚合到 face feature
→ 面内所有细分点共享同一个 face feature
→ MLP([feature, PE(u_sub, v_sub)]) → (d1, d2, d3)
→ fine_vert = base_pos + d1·b1 + d2·b2 + d3·n
```

Face coding 更优的原因：

**1. 编码空间的一致性**

Face coding 中，同一面内所有 scan 点的 `(u, v, d)` 坐标有明确的几何含义：`(u, v)` 是面内参数位置，`d` 是法线方向偏移。这构成了一个局部正则的参数空间。

Vertex coding 中，分配给同一 vertex 的 scan 点来自该顶点周围所有相邻面，它们的 `(Δx, Δy, Δz)` 没有一致的参考系——一个顶点可能是 5 个面的公共顶点，每个面的法线方向不同，导致同一方向的 Δz 在不同面上的语义不同。

**2. 解码时的特征失配问题**

在 vertex coding + SumOfFeatureDecoder 中，细分点的 feature 由 3 个顶点 feature 重心插值得到。**关键问题**: 位于不同面、相同重心坐标的细分点接收到完全相同的插值 feature，但它们需要预测不同的位移（因为所在面的大小、朝向不同）。MLP 必须仅靠 normal 信息区分它们，负担过重。

Face coding 中，每个面有独立的 face feature，不存在此问题。

**3. "归一化"的核心价值**

"归一化"（normalization）在此上下文中指将 scan 点从世界坐标系转换到面局部参数坐标系 `(u, v, d)` 的过程。这本质上是一种 **空间归一化 (spatial normalization)**：

```
世界坐标: p_world = v0 + u·e1 + v·e2 + d·n

归一化后: (u, v, d) 对于同一面内的点，
  u ∈ [0, 1] 表示沿 e1 方向的比例位置
  v ∈ [0, 1] 表示沿 e2 方向的比例位置  (u+v ≤ 1)
  d ∈ R     表示法线方向的偏移距离
```

**效果**: 无论面的绝对大小如何，`(u, v)` 总是在 `[0, 1]` 范围内。一个面积 1cm² 的面和面积 100cm² 的面，其内部点的 `(u, v)` 分布是相似的。这意味着：
- encoder 看到的输入分布更一致 → 学习更高效
- decoder 只需要学习在 `[0, 1]²` 参数域上预测位移 → 泛化更好

相比之下，vertex coding 的 `(Δx, Δy, Δz)` 范围取决于邻域大小，大面的 Δ 值大、小面的 Δ 值小，编码器需要同时处理多个尺度。

**更进一步——ortho frame 的附加归一化**:

当 `ortho_frame=True` 时，位移基底从 `{e1, e2, n}` (参数基底) 变为 `{ê₁, ê₂⊥, n}` (正交归一化基底)：
- `ê₁ = e1 / ||e1||` — 单位化 edge 1 方向
- `ê₂⊥ = n × ê₁` — 垂直于 ê₁ 和 n 的单位向量

这进一步解耦了位移的三个分量：`d1` 是沿边方向的位移（长度单位），`d2` 是面内垂直方向的位移，`d3` 是法线方向的位移。三个方向相互正交、尺度一致（都是绝对距离），MLP 更容易学习。

非 ortho 模式下，`d1·e1 + d2·e2` 中 e1 和 e2 通常不正交且长度不同（取决于三角形形状），MLP 需要学习补偿这种非正交性。

### 5.3 LightweightMR 曲率 Loss 对 Base Mesh 位移的启发

[High-Fidelity Lightweight Mesh Reconstruction from Point Clouds](https://arxiv.org/abs/2501.06285) (LightweightMR) 的核心思想是通过曲率加权的 Chamfer loss 驱动顶点自适应地向高曲率区域聚集。其 loss 设计：

```python
# LightweightMR: cal_chamfer_loss (losses.py:143-167)
def curvature_weighted_chamfer(sur_pts, sample_pts, curvature, w):
    """
    sur_pts:   GT surface points (N, 3)     — 从 scan 采样
    sample_pts: predicted vertices (M, 3)   — 网络输出的顶点
    curvature: per-GT-point curvature (N, 1) — 预计算的曲率, ∈ [0, 1]
    w: [w0, w1, w2, w3] loss weights
    """
    # Forward: GT → nearest predicted vertex (曲率加权)
    nn_idx_fwd = nearest_neighbor(sample_pts, sur_pts)  # for each GT point
    dist_fwd = ||sur_pts - sample_pts[nn_idx_fwd]||²
    L_fwd = w0 * mean(dist_fwd * curvature) + w1 * mean(dist_fwd)

    # Backward: predicted vertex → nearest GT point
    nn_idx_bwd = nearest_neighbor(sur_pts, sample_pts)
    dist_bwd = ||sample_pts - sur_pts[nn_idx_bwd]||²
    L_bwd = w2 * mean(dist_bwd)

    # Self-repulsion: predicted vertex → nearest other predicted vertex
    nn_idx_self = nearest_neighbor_exclude_self(sample_pts, sample_pts)
    dist_self = ||sample_pts - sample_pts[nn_idx_self]||²
    L_repul = -w3 * mean(clamp(dist_self, max=C * mean(dist_self)))

    return L_fwd + L_bwd + L_repul
```

**对 base mesh 位移的启发**：

如果我们在 Stage 2 细分之前，先对 base mesh 顶点做一步位移使其更贴合 GT，可以借鉴以下思路：

**方案：Pre-Subdivision Vertex Displacement**

```python
# 在 Stage 2 的 decoder 之前增加一步
def pre_subdivision_displacement(base_verts, base_faces, scan_points, scan_curvature):
    """
    对 base mesh 顶点做小幅位移，使其更贴合 GT surface。
    不改变 topology，仅移动顶点。
    """
    # 1. 每个 base vertex 找最近 scan 点
    nn_idx = knn(scan_points, base_verts, K=1)
    target = scan_points[nn_idx]           # (V, 3) 最近 GT 位置

    # 2. 位移向量
    delta = target - base_verts            # (V, 3)

    # 3. 曲率加权：高曲率顶点位移更多
    vertex_curv = scan_curvature[nn_idx]   # (V, 1) ∈ [0, 1]
    alpha = 0.3 + 0.7 * vertex_curv        # weight ∈ [0.3, 1.0]
    displaced = base_verts + alpha * delta

    return displaced
```

**更有价值的应用方向**: 曲率 loss 可以指导自适应 base mesh 简化（已在 Stage 1 中实现为 curvature-adaptive QEM），也可以作为 Phase 3 联合训练中 Stage 1 的正则化信号。具体地：如果未来恢复神经化 Stage 1，curvature-weighted Chamfer 可以作为 seed 点位置优化的主 loss，其中 forward chamfer 的曲率加权项 `w0 * mean(dist * curvature)` 提供强信号迫使网络在高曲率区域放置更多 seed。

### 5.4 为什么 Vertex Feature 插值与 GT Embedding 对比的 Loss 不可行

**设想的 loss**: 在 vertex coding 模式下，对每个细分点 `p_sub`：
1. 计算插值 feature: `f_interp = Σ_i w_i · f_{v_i}`（重心插值自 3 个顶点 feature）
2. 计算 GT embedding: 在 `p_sub` 位置重新运行 grouper + encoder，得到 `f_gt`
3. Loss: `L = ||f_interp - f_gt||²`

**开销分析**:

```
当前 pipeline 中需要编码的锚点数:
  F ≈ 2000 faces (or V ≈ 3000 vertices)
  每个锚点: KNN (K=512) + backbone + SA + pooling

如果对每个细分点计算 GT embedding:
  细分点数: F × K_sub ≈ 2000 × 105 = 210,000
  每个细分点: KNN (K=512) + backbone + SA + pooling

开销对比:
  Encoder forward passes:     2,000 → 210,000     (105×)
  KNN queries:                2,000×512 = 1M → 210,000×512 = 107M   (107×)
  Self-attention:             2,000 × 512² = 524M → 210,000 × 512² = 55B  (105×)
  显存 (粗略估计):
    当前: ~4GB (grouped features + SA intermediates)
    细分点: ~420GB  ← 完全不可能
```

即使采用分块计算（每次处理一部分细分点），**时间开销仍然增加 105 倍**，将每个 training step 从 ~4s 增加到 ~7 分钟，完全不可行。

**额外的理论问题**: 即使计算可行，这个 loss 也有根本缺陷——它假设"在细分点位置独立编码的 feature"就是正确的目标。但 vertex feature 的目的是在**整个三角面内**提供一致的表示，而非逐点精确匹配。强迫插值 feature 等于逐点编码结果，会迫使 vertex feature 包含极高频的空间变化信息，这与"vertex feature 是面级别的低频表示" 的设计初衷矛盾。

### 5.5 PoNQ-inspired Stage 1 的原理与失败分析

#### 5.5.1 原理

PoNQ (Points of Normal Quadrics, CVPR 2024) 的核心是从一组带法线的点生成三角 mesh。我们的 Stage 1 尝试将其改造为 feed-forward 网络：

```
┌─────────────── Stage 1 Pipeline ───────────────┐
│                                                 │
│  scan_points (1, 819200, 6)                     │
│       │                                         │
│       ▼                                         │
│  Stage1Encoder (Shared MLP)                     │
│    Linear(6→64) → BN → ReLU                    │
│    Linear(64→128) → BN → ReLU                  │
│    Linear(128→256) → BN → ReLU                 │
│    Linear(256→512) → BN → ReLU                 │
│       │                                         │
│       ├── per_point_feat (1, 819200, 128)       │
│       └── global_feat (1, 512) [max pool]       │
│       │                                         │
│       ▼                                         │
│  SeedPredictor                                  │
│    Step 1: Random subsample 819k → 50k          │
│    Step 2: FPS on 50k → N seeds (N ∈ [600,1500])│
│    Step 3: For each seed:                       │
│      KNN(seed, scan, K=32) → local features     │
│      MLP([local_feat, seed_feat, global_feat,   │
│           seed_xyz, seed_normal]) → offset, normal│
│      seed_pos = fps_pos + offset  [DIFF]        │
│      seed_normal = normalize(MLP)  [DIFF]       │
│       │                                         │
│       ▼                                         │
│  DelaunayMeshExtractor [torch.no_grad]          │
│    Step 1: Add 8 bbox corners                   │
│    Step 2: scipy.spatial.Delaunay (CPU)         │
│      → tetrahedralization                       │
│    Step 3: Circumcenter-normal voting           │
│      For each tet:                              │
│        circumcenter = tet_circumcenter(4 verts) │
│        for each vertex v_i of tet:              │
│          vote = dot(v_i - circumcenter, n_i) > 0│
│        tet_color = sum(votes) ∈ {0,1,2,3,4}    │
│        if any vertex is bbox corner: color = 0  │
│    Step 4: Barycenter correction                │
│      Fix ambiguous tets using signed distance   │
│    Step 5: Binary threshold (color ≥ 3 = inside)│
│    Step 6: Extract surface triangles at          │
│      inside/outside tet boundaries              │
│    Step 7: Remove corner vertices + degenerate  │
│       │                                         │
│       ▼                                         │
│  base_verts = seed_positions[vert_indices]      │
│  [GRAD PRESERVED via indexing]                  │
│                                                 │
│  base_faces, base_normals                       │
└─────────────────────────────────────────────────┘
```

**可微分性设计**: 虽然 Delaunay + tet classification + surface extraction 都是不可微的离散操作，但输出的 `vert_indices` 指向可微分的 `seed_positions` 张量。因此 `base_verts = seed_positions[vert_indices]` 保持了梯度流。在 joint training 中，Stage 2 的渲染 loss 梯度可以通过 base_verts 流回 seed_positions → offset_mlp → encoder。

#### 5.5.2 训练流程

**Phase 1 (仅 Stage 1, 快速预训练)**:
```python
# 使用 forward_seeds_only() 跳过 Delaunay, 直接在 seed 点上计算 loss
seed_positions, seed_normals, _ = stage1.forward_seeds_only(scan, normals)

L = curvature_weighted_chamfer(gt_points, seed_positions, curvature)
  + 100 * normal_consistency(seed_normals, gt_normals)
```

#### 5.5.3 失败原因

训练 ~20 epochs 后观察到：
1. **训练极慢** (~35 min/epoch, 3×GPU)
2. **大量破面** (mesh 中出现非流形边、退化三角形)
3. **Loss 不收敛** (除 normal consistency 外均不下降)
4. **无曲率自适应** (seed 分布仍然接近均匀)

**根本原因分析**:

| 问题 | 原因 | 严重程度 |
|------|------|----------|
| 训练慢 | 819k 点逐点通过 MLP backbone + BN = `819k × 4 layers × 2 ops` ≈ 6.5M 次操作/sample; KNN(seed, 819k, K=32) 搜索空间巨大 | 致命 |
| 破面 | Delaunay 对 ~1000 个稀疏 3D 点做四面体剖分，结果质量差。PoNQ 原始设计用 5k–10k+ 点（3D CNN voxel中心）且配合 min-cut 优化；我们仅用 ~1000 个 FPS 点且仅用简单投票阈值 | 致命 |
| Loss 不收敛 | Chamfer loss 的曲率加权项被平均到 ~800 个 seed 上（819k÷1000），每个 seed 的曲率信号 ≈ 1/800 × curvature_value，太弱。backward Chamfer + self-repulsion 的均匀化力远强于 forward Chamfer 的曲率信号 | 致命 |
| 无曲率自适应 | 上一条的直接后果：曲率加权对 seed 分布的影响 ≈ 0 | 严重 |

**与 PoNQ 原方法的关键差异**:

| 方面 | PoNQ (原始) | 我们的改造 |
|------|-------------|------------|
| 输入点数 | 5k–10k+ (3D CNN voxel 中心) | ~1000 (FPS seed) |
| 点的来源 | 3D CNN 在整个体积上预测 | FPS 采样 + 学习偏移 |
| Tet 分类 | Circumcenter voting + min-cut (图切割优化) | 仅 circumcenter voting + 简单阈值 |
| 训练方式 | 离线优化 / end-to-end 3D CNN | feed-forward encoder + 不可微拓扑 |
| 目标面数 | 50k–200k+ | ~1000–3000 |

核心矛盾：PoNQ 在高密度点集（>>5k）上工作良好，因为 Delaunay 的四面体质量随点密度提升。但我们的需求是极低面数（~2000 面 ≈ ~1000 顶点），在如此稀疏的点集上 Delaunay 产生大量狭长/退化四面体，法线投票分类的可靠性急剧下降。

**结论**: PoNQ 的 mesh extraction 算法不适合"从稀疏点直接生成低面数 mesh"的场景。它的 sweet spot 是中高密度点集生成中高面数 mesh。

### 5.6 EdgeRunner + LoRA + 曲率 Loss 微调: 可行性分析

[EdgeRunner](https://arxiv.org/abs/2409.18114) (NVIDIA, ICLR 2025) 是一个自回归 mesh 生成模型：将三角 mesh tokenize 为 1D token 序列，通过 auto-regressive auto-encoder (ArAE) 编码为固定长度 latent code，再自回归解码生成 mesh。可从点云或图像条件生成最多 ~4000 面的 mesh。

#### 架构概述

```
Input: point cloud (or image)
    │
    ▼
Lightweight Encoder → fixed-length latent z (e.g., 256-dim or 512-dim)
    │
    ▼
Auto-regressive Decoder (Transformer)
    │  Sequentially predicts mesh tokens:
    │    [edge_op, vertex_coord, ..., EOS]
    │  Token vocabulary: discretized coordinates + edge operations
    │  Sequence length: ~4000 tokens for ~4000 faces
    ▼
Triangle Mesh (up to 4000 faces, 512³ resolution)
```

#### 3×RTX 3090 训练可行性

| 项目 | 估计值 | 说明 |
|------|--------|------|
| **模型参数量** | ~100M–300M | Transformer decoder (12–24 layers, dim=512–1024) |
| **bf16 参数显存** | ~200–600MB | 参数本身不是瓶颈 |
| **LoRA 可训练参数** | ~2–5M (rank=16) | 占原始参数 1–3% |
| **LoRA 优化器状态** | ~40–100MB | AdamW: 2 × 可训练参数 |
| **自回归解码 KV cache** | ~2–8GB | 序列长度 ~4000 tokens, 累积的 KV 对 |
| **Curvature loss 额外开销** | ~1–2GB | KNN search on 819k points |
| **单卡总显存** | ~8–14GB | 可在单张 3090 (24GB) 上运行 |
| **3 卡 DDP** | **可行** | 每卡一个 sample, 有效 batch_size=3 |

**关键限制**:

1. **Flash Attention 兼容性**: EdgeRunner 依赖 `flash-attn`。RTX 3090 是 Ampere 架构 (SM 8.6)，**支持** Flash Attention 2，但性能不如 A100 (SM 8.0 with higher memory bandwidth)。

2. **自回归解码速度**: 生成 4000 面的 mesh 需要 ~4000 步 sequential decoding，每步需要完整的 Transformer forward pass + KV cache update。估计单个 mesh 生成需 5–15 秒，**训练一个 epoch (525 samples) 需 ~1–2 小时**（含 curvature loss 计算）。

3. **Curvature loss 集成**: 需要将 mesh token 序列解码为实际的 vertex/face 数组后才能计算 Chamfer distance 和曲率加权。这意味着每个 training step 都需要完整地 sample 出 mesh (teacher forcing 或 full generation)，开销大。**如果用 teacher forcing (ground truth tokens as input, predict next token)**，curvature loss 无法直接计算（因为 mesh 是逐 token 生成的，中间状态不是完整 mesh）。

4. **梯度流问题**: EdgeRunner 的 token 是离散的（quantized coordinates + edge operations）。Curvature loss 需要对顶点坐标求梯度，但 quantization 截断了梯度。解决方案：
   - Straight-Through Estimator (STE): 简单但梯度噪声大
   - Gumbel-Softmax: 适用于 categorical tokens
   - 仅在 continuous latent z 上优化 curvature loss (latent-space regularization)

#### 综合评估

| 维度 | 评分 | 说明 |
|------|------|------|
| 显存可行性 | ✅ 可行 | 单卡 24GB 足够, LoRA 大幅降低显存 |
| 训练速度 | ⚠️ 较慢 | 自回归解码是主要瓶颈, ~1–2h/epoch |
| 工程复杂度 | ❌ 高 | 需修改 EdgeRunner 训练逻辑, 集成 curvature loss, 处理离散→连续梯度 |
| 预期收益 | ⚠️ 不确定 | EdgeRunner 原始输出已被验证"不优于 QEM"；LoRA finetune 可能改善但上限不明 |
| 与 Stage 2 兼容性 | ⚠️ 中等 | EdgeRunner 生成的 mesh 拓扑不规则 (非均匀细分), 可能影响 face-anchored encoding 质量 |

**建议**: 在投入 EdgeRunner LoRA 微调的工程工作之前，应先验证 curvature-adaptive QEM (已实现的 Route C) 是否已能显著提升 Stage 2 质量。如果 adaptive QEM 就能带来足够提升，则无需引入 EdgeRunner 的复杂性。EdgeRunner 的优势在于生成全新拓扑（而非简化已有 mesh），但对于从 GT scan 简化到低面数 base mesh 的场景，基于简化的方法通常比生成方法更可靠。

---

## References

- [PoNQ: Points of Normal Quadrics](https://arxiv.org/abs/2404.07498) (CVPR 2024) — Delaunay + tet classification mesh extraction
- [High-Fidelity Lightweight Mesh Reconstruction from Point Clouds](https://arxiv.org/abs/2501.06285) (CVPR 2025) — Curvature-adaptive vertex placement
- [EdgeRunner: Auto-regressive Auto-encoder for Artistic Mesh Generation](https://arxiv.org/abs/2409.18114) (ICLR 2025) — Autoregressive mesh tokenization
