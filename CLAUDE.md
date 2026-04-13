# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HumanScanCompression is a two-stage neural mesh compression system for human scans:

- **Stage 1** (`models/stage1_pipeline.py`): Feed-forward mesh generator. Takes a dense point cloud (~819k points + normals) and produces a coarse **base mesh** (~1000–3000 faces) via learned seed point prediction + Delaunay mesh extraction.
- **Stage 2** (`models/pipeline.py`): Detail refinement. Takes the coarse base mesh + the dense point cloud and produces a **fine mesh** (1.3M+ vertices) via face-anchored encoding + neural subdivision.

Both stages are jointly trainable end-to-end. Stage 2's rendering loss provides gradient signal back to Stage 1's vertex positions, so Stage 1 learns to produce base meshes that are optimal for Stage 2.

## Common Commands

**Stage 2 only (original workflow):**
```bash
python train_stage2.py --config config.yaml
python infer_stage2.py --checkpoint /path/to/epoch_165.pth --config config.yaml
```

**Joint Stage 1 + Stage 2 training (three phases):**
```bash
# Phase 1: Pre-train Stage 1 with geometric losses (fast, no rendering)
python train_joint.py --config configs/config_joint.yaml --phase 1

# Phase 2: Pre-train Stage 2 with Stage 1 base meshes (Stage 1 frozen)
python train_joint.py --config configs/config_joint.yaml --phase 2 \
    --stage1-ckpt checkpoints/stage1_final.pth

# Phase 3: Joint fine-tuning of both stages
python train_joint.py --config configs/config_joint.yaml --phase 3 \
    --stage1-ckpt checkpoints/stage1_final.pth \
    --stage2-ckpt checkpoints/stage2_final.pth
```

**Data preprocessing:**
```bash
python preprocess_data.py           # Full preprocessing pipeline
python preprocess_base_meshes.py    # Pre-simplify base meshes
python create_lmdb.py               # Build LMDB database for fast I/O
```

There are no unit tests in this codebase.

## Architecture

### Stage 1: Feed-Forward Mesh Generator (`models/stage1_pipeline.py`)

Inspired by PoNQ (Delaunay mesh extraction) and LightWeightMR (curvature-adaptive vertex placement).

```
scan_points (1, 819200, 3) + scan_normals (1, 819200, 3)
    │
    ▼
Stage1Encoder  ── shared MLP backbone (6→64→128→256→512)
    │
    ├── per_point_feat (1, P, 128)
    └── global_feat (1, 512)
    │
    ▼
SeedPredictor
    FPS(819k → 50k subsample → N seeds)        [non-differentiable]
    KNN local aggregation per seed (K=32)
    offset_mlp → Δposition                      [differentiable]
    normal_mlp → unit normal                    [differentiable]
    seed_pos = fps_pos + Δpos
    │
    ▼
DelaunayMeshExtractor (models/mesh_extraction.py)
    scipy.Delaunay tetrahedralization           [non-differentiable]
    Normal-based tet classification (PoNQ)      [non-differentiable]
    Surface boundary extraction → faces
    base_verts = seed_pos[vert_indices]         [preserves grad!]
    │
    ▼
base_verts (1, V, 3)   base_faces (1, F, 3)   base_normals (1, V, 3)
```

**Key design properties:**
- **Curvature-adaptive density**: The curvature-weighted Chamfer loss (from LightWeightMR) penalizes high-curvature GT regions more, causing the network to cluster more seed points at fingers, face, and creases.
- **Gradient flow**: Topology is discrete (recomputed each forward pass), but vertex positions are differentiable. Stage 2's rendering loss flows through `base_verts → seed_positions → offset_mlp → encoder`.
- **Face-anchored compatibility**: Stage 1's output directly feeds Stage 2's `FaceLocalGrouper` + `FaceTriangleDecoder`. No adapter needed.

**Modules:**

| Module | File | Purpose |
|--------|------|---------|
| `Stage1Encoder` | `models/stage1_pipeline.py` | Shared MLP backbone → per-point + global features |
| `SeedPredictor` | `models/stage1_pipeline.py` | FPS selection + KNN aggregation + offset/normal MLPs |
| `DelaunayMeshExtractor` | `models/mesh_extraction.py` | Delaunay tet → normal voting → surface extraction |
| `Stage1Pipeline` | `models/stage1_pipeline.py` | Wraps all three; has `forward()` and `forward_seeds_only()` |

**Mesh extraction algorithm** (adapted from PoNQ `PoNQ_to_mesh.py`):
1. Add 8 bounding-box corner vertices (ensures closed Delaunay)
2. `scipy.spatial.Delaunay` tetrahedralization on seed positions
3. Circumcenter-normal voting: for each tet, check if vertex-to-circumcenter vectors align with vertex normals. 4 votes = inside, 0 = outside.
4. Barycenter correction: fix ambiguous tets using barycenter-to-vertex plane distances
5. Surface extraction: keep triangles at inside/outside tet boundaries
6. Filter corner vertices and degenerate faces

**Stage 1 losses** (`utils/stage1_losses.py`, adapted from LightWeightMR `losses.py`):

| Loss | Formula | Purpose |
|------|---------|---------|
| Curvature-weighted Chamfer (forward) | `w0 * mean(‖scan - nn_seed‖² × curvature) + w1 * mean(‖scan - nn_seed‖²)` | Adaptive coverage |
| Chamfer (backward) | `w2 * mean(‖seed - nn_scan‖²)` | Surface proximity |
| Self-repulsion | `-w3 * mean(clamp(‖seed - nn_other_seed‖²))` | Prevent collapse |
| Normal consistency | `mean(1 - cos_sim(seed_normal, nn_gt_normal))` | Accurate frames |
| Mesh regularity | `var(edge_lengths)/mean² + penalty(zero-area faces)` | Mesh quality |

**Curvature estimation** (`compute_curvature_from_normals` / `_compute_curvature_np_static`):
Precomputed in dataset `__getitem__` on a 50k subsample for speed, interpolated to full 819k via nearest-neighbor. Uses normal variation among KNN neighbors, Gaussian-smoothed, sigmoid-normalized to [0, 1]. Stored as `scan_curvature` in batch dict.

### Stage 2: Detail Refinement (`models/pipeline.py` → `Stage2Pipeline`)

```
base_verts/faces (from Stage 1 or QEM)  +  scan_points (B, P, 3+3)
        ↓
    Grouper  — assigns scan points to face centroids (FaceLocalGrouper)
        ↓
    Encoder  — extracts per-face features from local point neighborhoods
        ↓
  [Optional VAE bottleneck]
        ↓
    Decoder  — subdivides base mesh, interpolates features, predicts displacements
        ↓
  fine_verts (B, N_fine, 3),  fine_faces (B, M_fine, 3)
        ↓
  DifferentiableNormalRenderer  →  normal map losses  →  backprop to Stage 1 + Stage 2
```

### Swappable Components

Each component has multiple implementations selectable via `config.yaml`:

| Stage | `encoder_type` | Implementation |
|-------|----------------|----------------|
| Encoder | `attentive` | Attention-weighted scatter pooling |
| | `cross_attention` | Cross-attention (scan→vertex queries) |
| | `pt_sa_attentive` | Point Transformer self-attention |
| | `standard` | Simple MLP + scatter max |

| Stage | `decoder_type` | Implementation |
|-------|----------------|----------------|
| Decoder | `sum_of_feature` | Sums interpolated features before MLP |
| | `standard` | Per-vertex MLP on interpolated features |
| | `face_triangle` | Face-local coordinate system |

Groupers in `models/grouper.py`: `LocalPatchGrouper` (KNN around vertices), `FaceLocalGrouper` (KNN around face centroids).

### Subdivision (`utils/subdivision.py`)

`BarycentricSubdivision` subdivides the base mesh edges by a configurable rate (`subdivision_rate`, default 16). Each new vertex is represented by its barycentric coordinates w.r.t. the original triangle, enabling feature interpolation from the three anchor vertices. This is the core mechanism linking coarse features to fine geometry.

### Losses (`utils/train_helpers.py`)

The primary loss is **differentiable rendering** (`utils/render.py`): normal maps are rendered from ~20 random camera views and compared to GT normal maps via L1 + SSIM. Additional terms:

- `w_normal_l1` / `w_normal_ssim`: pixel-space normal losses
- `w_laplacian`: uniform Laplacian smoothness (NGF-style)
- `w_disp`: L2 displacement magnitude penalty
- `w_mat`: T-Net feature transform regularization
- `w_chamfer`: Chamfer distance (expensive, off by default)
- `vae_beta`: KL divergence when VAE is enabled

The normal gradient loss (∂N/∂x, ∂N/∂y) is computed only on the AND-mask of prediction and GT silhouettes, because the loss has no "push-back" force on over-extended regions outside the GT silhouette.

### Data (`data/stage2_dataset.py`)

`ScanToMeshDataset` performs online mesh simplification each epoch (using PyFQMR or Open3D) to randomly sample a base mesh with `base_mesh_faces_min`–`base_mesh_faces_max` faces. Supports:

- **LMDB** (`lmdb_path`): fast binary database for scan data
- **Pre-simplified base meshes** (`use_preprocess_base_mesh`): avoid recomputing simplification
- **Sharp-edge biased sampling** (`sharp_edge_sampling`): higher point density on creases

Dataset is THuman2.0 (no normalization applied).

### Training Infrastructure

- **Hugging Face Accelerate** for distributed/mixed-precision training (bf16)
- **AdamW** + cosine annealing LR scheduler
- **wandb** for experiment tracking
- Timing hooks (`models/timing_hooks.py`) for profiling bottlenecks
- Resume via `resume_path` in config

## Joint Training (`train_joint.py`)

Three-phase training strategy for end-to-end Stage 1 + Stage 2:

### Phase 1: Pre-train Stage 1 (~300ms/sample)
- Trains: `Stage1Encoder` + `SeedPredictor`
- Skips: mesh extraction, Stage 2, rendering
- Uses `forward_seeds_only()` (fast, no Delaunay)
- Loss: curvature-weighted Chamfer + normal consistency
- LR: 1e-3, cosine anneal

### Phase 2: Pre-train Stage 2 (~4s/sample)
- Trains: Stage 2 only (Stage 1 frozen in eval mode)
- Stage 1 generates base meshes (with Delaunay extraction)
- Stage 2 trains on these neural base meshes instead of QEM-simplified ones
- Loss: existing Stage 2 rendering losses
- LR: 5e-4, cosine anneal

### Phase 3: Joint fine-tuning (~4.5s/sample)
- Trains: both stages, dual-LR optimizer
  - Stage 1: lr=1e-5 (small, topology changes cause noise)
  - Stage 2: lr=1e-4
- Loss: `L_stage2_rendering + λ * L_stage1_geometric`, where λ decays from 1.0 to 0.1
- Delaunay recomputed every forward pass (topology adapts as vertex positions change)

### End-to-End Gradient Flow

```
fine_verts rendering loss
  │  ∂L/∂fine_verts (differentiable)
  ▼
FaceTriangleDecoder: fine_verts = lp + disp
  lp depends on v0, e1, e2 from _compute_face_basis(base_verts, base_faces)
  disp depends on face_features, which uses face centroids from base_verts
  │  ∂L/∂base_verts (through BOTH subdivision positions AND face features)
  ▼
base_verts = seed_positions[vert_indices]      ← simple indexing, preserves grad
  │  ∂L/∂seed_positions
  ▼
seed_positions = fps_points + offset_mlp(...)  ← fps_points is constant
  │  ∂L/∂offset_mlp, ∂L/∂encoder
  ▼
Stage1Encoder (backbone weights)

NOT differentiable (discrete, recomputed each forward pass):
  - FPS selection (initial seed indices)
  - Delaunay topology (which faces exist)
  - Tet classification (inside/outside)

NOTE: base_normals from Stage 1 are NOT used by FaceTriangleDecoder
(it computes face normals from vertex positions). The normal MLP is
trained only via Stage 1's normal consistency loss.
```

## Configuration

The active config is `config.yaml`. Experiment variants live in `configs/`:

- `configs/default.yaml` — minimal baseline
- `configs/config_ptsa.yaml` — Point Transformer self-attention encoder
- `configs/config_ca.yaml` — cross-attention encoder
- `configs/config_ptsa_progress.yaml` — PTSA with progressive training
- `configs/config_joint.yaml` — Joint Stage 1 + Stage 2 training (three phases)

Key tunable parameters:
- `model.encoder_type` / `model.decoder_type` — swap components
- `render.views_per_sample` — more views = better supervision, more VRAM
- `render.view_chunk_size` — trade throughput for VRAM
- `model.subdivision_levels` — positional encoding frequency bands
- `model.subdivision_rate` — edge subdivision factor (controls output mesh resolution)
- `stage1.num_seeds_min` / `stage1.num_seeds_max` — compression rate control (more seeds = more faces)

## Reference Implementations

Two external repositories are bundled for reference:

- **`PoNQ/`** (CVPR 2024): Feed-forward 3D CNN for mesh generation via Delaunay + min-cut surface extraction. We adapted the tet classification algorithm (`PoNQ_to_mesh.py:get_init_tet_color`, `correct_tet_color`, `get_surface`) for our `DelaunayMeshExtractor`. PoNQ is feed-forward (not per-scan optimized).
- **`LightweightMR/`** (CVPR 2025): Curvature-adaptive vertex placement via curvature-weighted Chamfer loss. We adapted `cal_curvature_with_normal`, `cal_chamfer_loss`, and `gaussian_kernel` for our `stage1_losses.py`. LightWeightMR is per-scan optimized; our Stage 1 amortizes it into a feed-forward network.

## Research Context

The main open challenge is **uneven vertex displacement on protruding regions** (hands, feet, face). The current best result uses face-anchored encoding (face-local coordinate system) + attentive encoder + sum-of-features decoder + normal gradient loss.

Key findings (from `output2.md`):
- Linear feature interpolation (replacing the full sum-of-features decoder) fails because vertices at the same barycentric coordinates in triangles of different sizes receive identical features despite different spatial offsets.
- Normal gradient loss helps but causes instability at silhouette boundaries without the AND-mask restriction.
- VAE was removed as it did not help with high-frequency detail recovery.
- EdgeRunner-generated base meshes were found to be no better than QEM simplification, motivating the neural Stage 1 approach.
- SMPL template deformation was rejected for Stage 1 because clothing and scan incompleteness (e.g., arm-body contact) make fixed-topology approaches infeasible.
