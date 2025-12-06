import os
import glob
import argparse
import json
import torch
import trimesh
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def process_mesh(args):
    """
    处理单个 Mesh 文件：加载并转换为 Tensor 保存为 .pt
    """
    path, output_dir = args
    try:
        # Load mesh
        # process=False 保持原始顶点和面片，不进行合并或重排序
        mesh = trimesh.load(path, process=False, force='mesh')
        
        # Handle scene object
        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) == 0:
                return None
            # Concatenate all geometries if it's a scene
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
            
        # Convert to Tensor
        # Float32 for vertices, Int64 for faces
        gt_verts = torch.from_numpy(np.array(mesh.vertices)).float()
        gt_faces = torch.from_numpy(np.array(mesh.faces)).long()
        
        # Generate save path
        filename = os.path.basename(path)
        name, _ = os.path.splitext(filename)
        save_filename = f"{name}.pt"
        save_path = os.path.join(output_dir, save_filename)
        
        # Save
        torch.save({
            'gt_verts': gt_verts,
            'gt_faces': gt_faces
        }, save_path)
        
        # Return entry for json
        return {'pt_path': save_filename}
        
    except Exception as e:
        print(f"[Error] Failed to process {path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Preprocess OBJ/PLY meshes to PT files for fast loading.")
    parser.add_argument('--src_dir', type=str, required=True, help='Source directory containing .obj/.ply files')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory to save .pt files and index json')
    parser.add_argument('--split', type=str, default='train', help='Split name for index file (e.g. train -> train.json)')
    parser.add_argument('--workers', type=int, default=8, help='Number of worker processes')
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Find files
    print(f"Scanning {args.src_dir} for meshes...")
    files = glob.glob(os.path.join(args.src_dir, "**/*.obj"), recursive=True)
    files += glob.glob(os.path.join(args.src_dir, "**/*.ply"), recursive=True)
    
    if len(files) == 0:
        print("No .obj or .ply files found.")
        return
        
    print(f"Found {len(files)} files. Processing with {args.workers} workers...")
    
    # Prepare arguments for multiprocessing
    process_args = [(f, args.out_dir) for f in files]
    
    results = []
    # Use multiprocessing to speed up
    with Pool(processes=args.workers) as pool:
        for res in tqdm(pool.imap_unordered(process_mesh, process_args), total=len(files)):
            if res is not None:
                results.append(res)
                
    # Save index JSON
    json_path = os.path.join(args.out_dir, f"{args.split}.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Successfully processed {len(results)} samples.")
    print(f"Index file saved to: {json_path}")
    print(f"Please update your config['data']['processed_dir'] to: {args.out_dir}")

if __name__ == '__main__':
    main()
