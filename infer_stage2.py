import os
import argparse
import torch
import trimesh
import numpy as np
import open3d as o3d
import yaml
from tqdm import tqdm

from models.pipeline import Stage2Pipeline
from preprocess_data import normalize_mesh

# EdgeRunner/core/utils.py style normalization
def normalize_mesh_o3d(mesh, bound=0.95):
    # Convert to numpy to use the same logic
    verts = np.asarray(mesh.vertices)
    verts = normalize_mesh(verts, bound=bound) # Use the function from preprocess_data
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    return mesh

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to config.yaml (from checkpoint)")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument('--input_scan', type=str, required=True, help="Path to input scan (.obj/.ply)")
    parser.add_argument('--output_dir', type=str, default="results", help="Output directory")
    parser.add_argument('--base_verts', type=int, default=2000, help="Target base mesh vertices")
    parser.add_argument('--point_num', type=int, default=320000, help="Number of scan points to sample")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Config & Model
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Loading model from {args.checkpoint}...")
    model = Stage2Pipeline(config={
        'feature_dim': config['model']['feature_dim'],
        'enc_hidden_dim': config['model']['enc_hidden_dim'],
        'dec_hidden_dim': config['model']['dec_hidden_dim'],
        'subdivision_levels': config['model']['subdivision_levels'],
        'subdivision_rate': config['model']['subdivision_rate']
    }).to(device)
    
    ckpt = torch.load(args.checkpoint, map_location=device)
    # Handle if checkpoint saves 'model_state_dict' or just dict
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    # 2. Process Input Scan
    print(f"Processing scan: {args.input_scan}")
    
    # Load using Open3D for consistency with dataset simplification
    scan_mesh = o3d.io.read_triangle_mesh(args.input_scan)
    
    # Normalize (Critical!)
    # Note: This assumes input is raw scan. If input is already normalized, skip this.
    scan_mesh = normalize_mesh_o3d(scan_mesh, bound=0.95)
    
    # 3. Generate Base Mesh (Simplify)
    print(f"Simplifying to ~{args.base_verts} vertices...")
    base_mesh = scan_mesh.simplify_quadric_decimation(target_number_of_triangles=args.base_verts * 2)
    base_mesh.compute_vertex_normals()
    
    # 4. Sample Points (using Trimesh for consistency)
    # Convert Open3D -> Trimesh
    # (Or just write temp and read back to be safe)
    v_base = np.asarray(base_mesh.vertices).astype(np.float32)
    f_base = np.asarray(base_mesh.triangles).astype(np.int64)
    n_base = np.asarray(base_mesh.vertex_normals).astype(np.float32)
    
    # For sampling, we need the high-res scan as Trimesh
    v_scan = np.asarray(scan_mesh.vertices)
    f_scan = np.asarray(scan_mesh.triangles)
    tm_scan = trimesh.Trimesh(vertices=v_scan, faces=f_scan, process=False)
    
    print(f"Sampling {args.point_num} points...")
    scan_points, _ = trimesh.sample.sample_surface(tm_scan, args.point_num)
    scan_points = scan_points.astype(np.float32)
    
    # 5. Prepare Tensors
    # Add batch dim
    base_verts_t = torch.from_numpy(v_base).unsqueeze(0).to(device)
    base_faces_t = torch.from_numpy(f_base).unsqueeze(0).to(device)
    base_normals_t = torch.from_numpy(n_base).unsqueeze(0).to(device)
    scan_points_t = torch.from_numpy(scan_points).unsqueeze(0).to(device)
    
    # 6. Inference
    print("Running inference...")
    with torch.no_grad():
        # Forward
        fine_verts, fine_faces, disp = model(base_verts_t, base_faces_t, base_normals_t, scan_points_t)
        
    # 7. Export Results
    # fine_verts: (1, V_fine, 3)
    # fine_faces: (F_fine, 3) (Shared topology)
    
    out_verts = fine_verts[0].cpu().numpy()
    out_faces = fine_faces.cpu().numpy()
    
    # Export Base Mesh
    o3d.io.write_triangle_mesh(os.path.join(args.output_dir, "base_mesh.obj"), base_mesh)
    
    # Export Fine Mesh
    fine_mesh = trimesh.Trimesh(vertices=out_verts, faces=out_faces, process=False)
    fine_mesh.export(os.path.join(args.output_dir, "fine_mesh.obj"))
    
    # Export GT (Normalized) for comparison
    o3d.io.write_triangle_mesh(os.path.join(args.output_dir, "gt_normalized.obj"), scan_mesh)
    
    print(f"Results saved to {args.output_dir}")
    print(f"Base Verts: {len(v_base)}, Fine Verts: {len(out_verts)}")

if __name__ == '__main__':
    main()

