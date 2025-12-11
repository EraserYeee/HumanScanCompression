import os
import glob
import argparse
import torch
import numpy as np
import trimesh
import pymeshlab
from multiprocessing import Pool
from tqdm import tqdm
import json
import random

def get_pymeshlab_mesh_set(verts_np, faces_np):
    """
    Create a pymeshlab MeshSet from numpy arrays.
    """
    ms = pymeshlab.MeshSet()
    m = pymeshlab.Mesh(vertex_matrix=verts_np, face_matrix=faces_np)
    ms.add_mesh(m)
    return ms

def simplify_mesh_pymeshlab(verts_np, faces_np, target_faces):
    """
    Simplify mesh using PyMeshLab (QEM) to target face count.
    
    Args:
        verts_np: (V, 3) float32
        faces_np: (F, 3) int32/int64
        target_faces: int
        
    Returns:
        (V_new, 3), (F_new, 3)
    """
    try:
        # Pymeshlab expects float64 usually for coords, but handles float32. Faces should be int32.
        ms = get_pymeshlab_mesh_set(verts_np.astype(np.float64), faces_np.astype(np.int32))
        
        # High quality simplification settings
        # targetfacenum: target face count
        # preservenormal: preserve normals direction
        # preservetopology: avoid creating non-manifold edges
        # qualitythr: quality threshold (0-1), 1 is highest (slower)
        # optimalplacement: use optimal placement for collapsed vertices
        # planarsimp: simplification on planar regions
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum = target_faces, targetperc = 0.000000, qualitythr = 0.500000, preservenormal = False, preservetopology = False, planarquadric = False)
        ms.meshing_remove_folded_faces()
        # ms.meshing_decimation_quadric_edge_collapse(
        #     targetfacenum=target_faces,
        #     preservenormal=True,
        #     preservetopology=True,
        #     qualitythr=1.0,  # Max quality
        #     optimalplacement=True,
        #     planarsimp=True,
        #     planarquadric=True
        # )
        
        m = ms.current_mesh()
        new_verts = m.vertex_matrix().astype(np.float32)
        new_faces = m.face_matrix().astype(np.int64)
        
        return new_verts, new_faces
    
    except Exception as e:
        # If simplification fails (e.g. mesh too small), return original
        print(f"[Warning] Simplification failed: {e}")
        return verts_np, faces_np

def process_single_pt_file(args):
    """
    Process a single .pt file:
    1. Load GT mesh
    2. Create 3 random simplifications
    3. Save them in a subdirectory
    """
    pt_path, output_dir, min_faces, max_faces = args
    
    try:
        # Load GT
        data = torch.load(pt_path, map_location='cpu')
        gt_verts = data['gt_verts'].numpy()
        gt_faces = data['gt_faces'].numpy()
        
        # Filename info
        filename = os.path.basename(pt_path)
        name, _ = os.path.splitext(filename)
        
        # Create output subdir for this mesh
        mesh_out_dir = os.path.join(output_dir, name)
        os.makedirs(mesh_out_dir, exist_ok=True)
        
        results = []
        
        # Generate 3 random target face counts
        target_counts = [random.randint(min_faces, max_faces) for _ in range(5)]
        
        for i, target_f in enumerate(target_counts):
            # Simplify
            base_verts, base_faces = simplify_mesh_pymeshlab(gt_verts, gt_faces, target_f)
            
            # Save as .pt
            save_name = f"{name}_base_{i}_f{target_f}.pt"
            save_path = os.path.join(mesh_out_dir, save_name)
            
            torch.save({
                'base_verts': torch.from_numpy(base_verts),
                'base_faces': torch.from_numpy(base_faces),
                'gt_path': pt_path, # Link back to original GT
                'target_faces': target_f
            }, save_path)
            
            results.append({
                'base_pt_path': os.path.join(name, save_name), # Relative path inside processed_dir
                'gt_pt_path': pt_path,
                'base_faces': base_faces.shape[0]
            })
            
        return results
        
    except Exception as e:
        print(f"[Error] Failed to process {pt_path}: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Pre-generate simplified base meshes using PyMeshLab")
    parser.add_argument('--data_root', type=str, required=True, help='Root directory containing train.json/test.json and .pt files')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory for base meshes')
    parser.add_argument('--workers', type=int, default=8, help='Number of worker processes')
    parser.add_argument('--min_faces', type=int, default=1000)
    parser.add_argument('--max_faces', type=int, default=3000)
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    splits = ['train', 'test']
    
    for split in splits:
        json_path = os.path.join(args.data_root, f"{split}.json")
        if not os.path.exists(json_path):
            print(f"[Warning] {split}.json not found in {args.data_root}, skipping.")
            continue
            
        print(f"Processing split: {split}")
        with open(json_path, 'r') as f:
            file_list = json.load(f)
            
        # Extract full paths
        # Assuming entries in json look like {'pt_path': 'filename.pt'}
        # and they are located in args.data_root
        pt_files = []
        for entry in file_list:
            # Handle if pt_path is absolute or relative
            path = entry['pt_path']
            if not os.path.isabs(path):
                path = os.path.join(args.data_root, path)
            pt_files.append(path)
            
        # Output directory for this split (optional, or just put all in out_dir)
        # Let's put them in out_dir directly as requested "subfolders for each mesh"
        
        process_args = [(f, args.out_dir, args.min_faces, args.max_faces) for f in pt_files]
        
        all_results = []
        with Pool(processes=args.workers) as pool:
            for res in tqdm(pool.imap_unordered(process_single_pt_file, process_args), total=len(pt_files)):
                all_results.extend(res)
                
        # Save new index file
        new_json_path = os.path.join(args.out_dir, f"{split}_base.json")
        with open(new_json_path, 'w') as f:
            json.dump(all_results, f, indent=4)
            
        print(f"Saved index to {new_json_path}")
        print(f"Processed {len(pt_files)} meshes -> {len(all_results)} base meshes.")

if __name__ == '__main__':
    main()

