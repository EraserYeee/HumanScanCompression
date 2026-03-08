"""
可视化注意力机制：将每个注意力头的注意力分数映射到点云颜色上。

用法:
    python visualize_attention.py \
        --fine_mesh path/to/fine.obj \
        --base_mesh path/to/base.obj \
        --checkpoint path/to/checkpoint.pth \
        --config path/to/config.yaml \
        --output_dir path/to/output
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import argparse
import torch
import trimesh
import numpy as np
import yaml
from matplotlib import cm
from tqdm import tqdm
# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.pipeline import Stage2Pipeline


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_mesh(mesh_path):
    """加载 mesh 文件"""
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected Trimesh, got {type(mesh)}")
    return mesh


def sample_points_from_mesh(mesh, num_points):
    """从 mesh 表面均匀采样点云"""
    points, face_indices = mesh.sample(num_points, return_index=True)
    
    # 计算法线（从采样点所在面的法线）
    face_normals = mesh.face_normals[face_indices]
    # 归一化
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normals = face_normals / norms
    
    return points.astype(np.float32), normals.astype(np.float32)


def compute_vertex_normals(vertices, faces):
    """计算顶点法线"""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh.vertex_normals.astype(np.float32)


def attention_to_color(attention_values, colormap='viridis'):
    """
    将注意力分数映射到 RGB 颜色
    
    Args:
        attention_values: (N,) numpy array, 注意力分数 [0, 1]
        colormap: str, matplotlib colormap 名称
    
    Returns:
        colors: (N, 3) numpy array, RGB 颜色 [0, 255]
    """
    cmap = cm.get_cmap(colormap)
    colors = cmap(attention_values)[:, :3]  # (N, 3) in [0, 1]
    colors = (colors * 255).astype(np.uint8)  # Convert to [0, 255]
    return colors


def save_point_cloud_ply(points, colors, output_path):
    """
    保存点云为 PLY 格式（带颜色）
    
    Args:
        points: (N, 3) numpy array
        colors: (N, 3) numpy array, RGB [0, 255]
        output_path: str
    """
    # 确保 points 和 colors 形状匹配
    assert points.shape[0] == colors.shape[0], \
        f"Points ({points.shape[0]}) and colors ({colors.shape[0]}) must have same length"
    
    # 创建 PLY 文件
    with open(output_path, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Data
        for i in range(len(points)):
            f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} "
                   f"{colors[i, 0]} {colors[i, 1]} {colors[i, 2]}\n")


def main():
    parser = argparse.ArgumentParser(description="Visualize attention scores from encoder")
    parser.add_argument('--fine_mesh', type=str, required=True,
                       help='Path to fine mesh (.obj or .ply)')
    parser.add_argument('--base_mesh', type=str, required=True,
                       help='Path to base mesh (.obj or .ply)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pth)')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file (.yaml)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for visualization files')
    parser.add_argument('--num_points', type=int, default=819200,
                       help='Number of points to sample from fine mesh (default: 819200)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')
    parser.add_argument('--colormap', type=str, default='viridis',
                       help='Matplotlib colormap name (default: viridis)')
    parser.add_argument('--normalize_mode', type=str, default='global',
                       choices=['global', 'cluster'],
                       help='Normalization mode: global (min-max across all points) or cluster (min-max per cluster) (default: global)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load config
    print(f"Loading config from {args.config}...")
    config = load_config(args.config)
    
    # Ensure encoder_type is 'attentive'
    if config['model'].get('encoder_type', 'standard') != 'attentive':
        print("[Warning] encoder_type is not 'attentive'. Setting to 'attentive' for visualization.")
        config['model']['encoder_type'] = 'attentive'
    
    # Load meshes
    print(f"Loading fine mesh from {args.fine_mesh}...")
    fine_mesh = load_mesh(args.fine_mesh)
    fine_verts = np.array(fine_mesh.vertices, dtype=np.float32)
    fine_faces = np.array(fine_mesh.faces, dtype=np.int64)
    
    print(f"Loading base mesh from {args.base_mesh}...")
    base_mesh = load_mesh(args.base_mesh)
    base_verts = np.array(base_mesh.vertices, dtype=np.float32)
    base_faces = np.array(base_mesh.faces, dtype=np.int64)
    
    # Validate face indices
    print("Validating face indices...")
    max_face_idx = base_faces.max()
    num_verts = len(base_verts)
    if max_face_idx >= num_verts:
        print(f"[Warning] Invalid face indices: max index {max_face_idx} >= num vertices {num_verts}")
        print(f"  This usually means the mesh uses 1-based indexing. Converting to 0-based...")
        base_faces = base_faces - 1
        if base_faces.min() < 0:
            print(f"[Error] After conversion, found negative indices. Mesh may be corrupted.")
            raise ValueError("Invalid face indices in base mesh")
        print(f"  Converted: max index now {base_faces.max()}, num vertices {num_verts}")
    
    # Normalize to unit sphere (same as training dataset)
    # Training uses: center -> scale by max norm (unit sphere, radius=1)
    # Use fine mesh (GT) for normalization, as in training
    bbox_min = fine_verts.min(axis=0)
    bbox_max = fine_verts.max(axis=0)
    center = (bbox_min + bbox_max) / 2
    
    fine_verts_centered = fine_verts - center
    base_verts_centered = base_verts - center
    
    # Scale to unit sphere (max norm = 1) - use fine mesh to determine scale
    fine_norms = np.linalg.norm(fine_verts_centered, axis=1)
    scale = fine_norms.max()
    if scale < 1e-6:
        print("[Warning] Scale is too small, using 1.0")
        scale = 1.0
    
    fine_verts_norm = fine_verts_centered / scale
    base_verts_norm = base_verts_centered / scale
    
    print(f"Normalization: center={center}, scale={scale:.6f}")
    print(f"  Fine mesh: {len(fine_verts)} vertices, normalized range: [{fine_verts_norm.min():.3f}, {fine_verts_norm.max():.3f}]")
    print(f"  Base mesh: {len(base_verts)} vertices, normalized range: [{base_verts_norm.min():.3f}, {base_verts_norm.max():.3f}]")
    
    # Compute base mesh normals (after normalization)
    print("Computing base mesh normals...")
    base_normals = compute_vertex_normals(base_verts_norm, base_faces)
    
    # Sample points from fine mesh (use normalized fine mesh for sampling)
    print(f"Sampling {args.num_points} points from fine mesh...")
    fine_mesh_norm = trimesh.Trimesh(vertices=fine_verts_norm, faces=fine_faces, process=False)
    scan_points_norm, sampled_face_idx = fine_mesh_norm.sample(args.num_points, return_index=True)
    scan_points_norm = scan_points_norm.astype(np.float32)
    
    # Denormalize scan points for visualization (back to original coordinates)
    scan_points_viz = scan_points_norm * scale + center
    
    # Compute scan normals from face normals
    if args.num_points > 0:
        face_normals = fine_mesh_norm.face_normals[sampled_face_idx]
        norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        scan_normals = (face_normals / norms).astype(np.float32)
    else:
        scan_normals = np.zeros((0, 3), dtype=np.float32)
    
    # Convert to tensors (normalized coordinates for model)
    device = torch.device(args.device)
    base_verts_t = torch.from_numpy(base_verts_norm).unsqueeze(0).to(device)  # (1, V, 3)
    base_faces_t = torch.from_numpy(base_faces).unsqueeze(0).to(device)  # (1, F, 3)
    base_normals_t = torch.from_numpy(base_normals).unsqueeze(0).to(device)  # (1, V, 3)
    scan_points_t = torch.from_numpy(scan_points_norm).unsqueeze(0).to(device)  # (1, P, 3)
    scan_normals_t = torch.from_numpy(scan_normals).unsqueeze(0).to(device)  # (1, P, 3)
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = Stage2Pipeline(config=config['model'])
    
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model = model.eval().to(device)
    
    # Forward pass with attention return
    print("Running forward pass to extract attention scores...")
    with torch.no_grad():
        result = model(
            base_verts_t, base_faces_t, base_normals_t,
            scan_points_t, scan_normals=scan_normals_t,
            return_attention=True
        )
    
    # Unpack results
    if len(result) == 8:
        fine_verts, fine_faces, displacements, trans_feat, vertex_features, kl_loss, attention_scores, global_cluster_idx = result
    else:
        raise ValueError(f"Expected 8 return values, got {len(result)}. Make sure encoder_type='attentive' and return_attention=True")
    
    # attention_scores: (B*P, H)
    # global_cluster_idx: (B*P,)
    attention_scores = attention_scores.cpu().numpy()  # (B*P, H)
    global_cluster_idx_np = global_cluster_idx.cpu().numpy()  # (B*P,)
    num_heads = attention_scores.shape[1]
    
    print(f"Extracted attention scores: shape {attention_scores.shape}, {num_heads} heads")
    
    # Print statistics for each head
    for head_idx in range(num_heads):
        alpha_h = attention_scores[:, head_idx]
        print(f"  Head {head_idx}: min={alpha_h.min():.6f}, max={alpha_h.max():.6f}, mean={alpha_h.mean():.6f}, std={alpha_h.std():.6f}")
    
    # Use original coordinates for visualization
    base_verts_viz = base_verts  # Original coordinates (before normalization)
    # scan_points_viz already set above
    
    # For each attention head, create visualization
    print(f"Generating visualizations for {num_heads} attention heads...")
    for head_idx in range(num_heads):
        print(f"  Processing head {head_idx}...")
        
        # Extract attention scores for this head
        alpha_h = attention_scores[:, head_idx]  # (B*P,)
        
        # Normalize attention scores to [0, 1] for better visualization
        if args.normalize_mode == 'cluster':
            # Option 2: Per-cluster min-max normalization (方案B)
            # This highlights relative importance within each cluster
            alpha_h_norm = np.zeros_like(alpha_h)
            unique_clusters = np.unique(global_cluster_idx_np)
            for cluster_id in unique_clusters:
                mask = global_cluster_idx_np == cluster_id
                cluster_alpha = alpha_h[mask]
                if len(cluster_alpha) > 0:
                    c_min = cluster_alpha.min()
                    c_max = cluster_alpha.max()
                    if c_max > c_min:
                        alpha_h_norm[mask] = (cluster_alpha - c_min) / (c_max - c_min)
                    else:
                        alpha_h_norm[mask] = cluster_alpha  # All same value
            print(f"    Cluster-normalized: min={alpha_h_norm.min():.6f}, max={alpha_h_norm.max():.6f}, mean={alpha_h_norm.mean():.6f}")
        else:
            # Option 1: Global min-max normalization
            alpha_min = alpha_h.min()
            alpha_max = alpha_h.max()
            if alpha_max > alpha_min:
                alpha_h_norm = (alpha_h - alpha_min) / (alpha_max - alpha_min)
            else:
                alpha_h_norm = alpha_h  # All values are the same
            print(f"    Global-normalized: min={alpha_h_norm.min():.6f}, max={alpha_h_norm.max():.6f}, mean={alpha_h_norm.mean():.6f}")
        
        # Map to colors
        scan_colors = attention_to_color(alpha_h_norm, colormap=args.colormap)
        
        # Base vertices: black
        num_base_verts = base_verts_viz.shape[0]
        base_colors = np.zeros((num_base_verts, 3), dtype=np.uint8)
        
        # Combine points and colors
        all_points = np.vstack([base_verts_viz, scan_points_viz])  # (V + P, 3)
        all_colors = np.vstack([base_colors, scan_colors])  # (V + P, 3)
        
        # Save PLY file
        output_path = os.path.join(args.output_dir, f"attention_head_{head_idx}.ply")
        save_point_cloud_ply(all_points, all_colors, output_path)
        print(f"    Saved to {output_path}")
    
    print(f"\nVisualization complete! Output files saved to {args.output_dir}")
    print(f"  - {num_heads} PLY files (one per attention head)")
    print(f"  - Base mesh vertices shown in black")
    print(f"  - Scan points colored by attention scores (colormap: {args.colormap})")


if __name__ == '__main__':
    main()
