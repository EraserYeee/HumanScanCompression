这是一个非常清晰、模块化的技术路线。为了方便你直接将需求传递给 Cursor、Copilot 或用于 Vibe Coding，我将整个 Baseline 整理为**结构化的伪代码与技术规范文档**。

你可以直接复制下面的内容发给 AI 助手，它包含了明确的输入输出维度、类结构设计以及核心算法逻辑。

-----

## 📋 System Prompt / Technical Spec: Stage 2 Baseline

**Task:** Implement a 3D reconstruction pipeline ("Stage 2") that refines a coarse Base Mesh using features extracted from a dense Scan Point Cloud.
**Core Mechanism:** Mesh-anchored feature encoding (Local PointNet + Scatter) $\to$ Neural Subdivision $\to$ Differentiable Rendering Supervision.

### 1\. Data Pipeline: Anchor & Grouping (On-the-fly)

**Module Name:** `LocalPatchGrouper`
**Goal:** Assign scan points to mesh vertices and canonicalize coordinates.

  * **Input:**
      * `base_mesh_verts`: $(B, V, 3)$
      * `base_mesh_normals`: $(B, V, 3)$
      * `scan_points`: $(B, P, 3)$ (Dense point cloud)
  * **Logic:**
    1.  **KNN Assignment:** For every point in `scan_points`, find the nearest neighbor in `base_mesh_verts` ($K=1$).
          * Returns `cluster_idx`: $(B, P)$ indicating which vertex each point belongs to.
    2.  **Relative Coordinates:** $\Delta p = p_{scan} - v_{anchor}$.
    3.  **Rotation Alignment (Canonicalization):**
          * Construct rotation matrix $R$ for each vertex using its normal (Normal vector becomes Z-axis).
          * $p_{local} = \Delta p \times R^T$.
  * **Output:**
      * `p_local`: $(B, P, 3)$ (Local coordinates)
      * `cluster_idx`: $(B, P)$ (For scatter aggregation)

### 2\. Model Architecture

**Class Name:** `NeuralSubdivisionReconstructor`

#### A. Encoder (`LocalFeatureEncoder`)

  * **Input:** `p_local` $(B, P, 3)$, `cluster_idx` $(B, P)$
  * **Layers:**
    1.  **PointMLP:** Shared MLP `[3 -> 64 -> 128]`. Applied to every point.
    2.  **Scatter Aggregation:** `torch_scatter.scatter_max` (or `scatter_mean`).
          * Aggregates point features based on `cluster_idx` into vertex slots.
  * **Output:** `vertex_features` $(B, V, 128)$.

#### B. Decoder (`SubdivisionDecoder`)

  * **Input:** `base_mesh` (Topology), `vertex_features` $(B, V, 128)$
  * **Logic:**
    1.  **Topology Subdivision:** Perform Loop Subdivision on the mesh connectivity (e.g., 1 iteration: $V \to \approx 4V$).
    2.  **Feature Interpolation:**
          * Existing vertices keep features.
          * New vertices (on edges) get average features of endpoints.
    3.  **Displacement Head:**
          * MLP `[128 -> 64 -> 1]`.
          * Predicts scalar displacement $d$ for each vertex in the *subdivided* mesh.
    4.  **Geometry Update:** $v_{new} = v_{sub} + d \times n_{sub}$.
  * **Output:** `fine_mesh` (High-res vertices & faces).

### 3\. Training: Differentiable Rendering Loss

**Module Name:** `NormalConsistencyLoss`
**Library:** PyTorch3D / Kaolin

  * **Setup:**
      * Initialize a `MeshRenderer` with a `SoftPhongShader` or just a Normal Shader.
      * Define $K$ random cameras (Random azimuth/elevation) per batch item.
  * **Forward Process:**
    1.  **Render Prediction:** Render `fine_mesh` from camera $C_k$ to get Normal Map $I_{pred}$.
    2.  **Render GT:** Render `scan_mesh` (Ground Truth) from camera $C_k$ to get Normal Map $I_{gt}$.
          * *Note:* If GT is just points, use "Point Splatting" to render normals, or surface reconstruction (Poisson) beforehand to get a GT mesh.
  * **Loss Calculation:**
      * `loss_render = L1Loss(I_pred, I_gt)` or `CosineSimilarity(I_pred, I_gt)`.
      * **Auxiliary Regularization:**
          * `loss_laplacian`: Penalize rough surfaces on `fine_mesh`.
          * `loss_displacement`: L2 penalty on predicted $d$ (keep it minimal).

### 4\. Pseudo-Code Workflow (for Cursor)

```python
class Stage2Pipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.grouper = LocalPatchGrouper()
        self.encoder = LocalFeatureEncoder(emb_dim=128)
        self.decoder = SubdivisionDecoder(levels=1)
        self.renderer = DifferentiableNormalRenderer(img_size=512)

    def forward(self, base_mesh, scan_points, scan_mesh_gt=None, is_training=True):
        # 1. Grouping & Aligning
        # scan_points: (B, P, 3)
        # base_mesh.verts: (B, V, 3)
        p_local, cluster_idx = self.grouper(base_mesh, scan_points)

        # 2. Encoding
        # vertex_feats: (B, V, 128) - One feature vector per base vertex
        vertex_feats = self.encoder(p_local, cluster_idx, num_verts=base_mesh.num_verts)

        # 3. Decoding (Subdivision)
        # fine_mesh corresponds to the subdivided geometry
        fine_mesh = self.decoder(base_mesh, vertex_feats)

        if is_training and scan_mesh_gt is not None:
            # 4. Differentiable Rendering Loss
            # Render normals from random views
            pred_normals = self.renderer(fine_mesh)
            gt_normals = self.renderer(scan_mesh_gt)
            
            loss = F.l1_loss(pred_normals, gt_normals)
            return fine_mesh, loss
        
        return fine_mesh
```

-----

### 💡 关键提示给 AI 助手 (Tips for AI)

1.  **Handle Variable Sizes:** Be careful with batching. Use **Packed/Batch** format (e.g., PyTorch3D `Meshes` structure) because the number of points $P$ and vertices $V$ might vary across samples.
2.  **Rotation Matrix:** When constructing the local frame, use the vertex normal as the Z-axis. Use `pytorch3d.transforms.rotation_6d_to_matrix` or Gram-Schmidt process to find orthogonal X and Y axes deterministically.
3.  **Scatter Function:** Use `torch_scatter.scatter_max` for the encoder aggregation to preserve sharp geometric features (edges/corners).
4.  **Renderer:** Ensure the renderer is set up to output **Screen-space Normals** (pixel values represent normal vectors).