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
from pathlib import Path

def convert_stl_to_pt(args):
    """
    将单个 STL 文件转换为 .pt 格式（参考 preprocess_data.py）
    
    Args:
        args: (stl_path, pt_output_dir) 元组
    """
    stl_path, pt_output_dir = args
    try:
        # Load mesh with trimesh
        # process=False 保持原始顶点和面片，不进行合并或重排序
        mesh = trimesh.load(stl_path, process=False, force='mesh')
        
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
        filename = os.path.basename(stl_path)
        name, _ = os.path.splitext(filename)
        save_filename = f"{name}.pt"
        save_path = os.path.join(pt_output_dir, save_filename)
        
        # Save
        torch.save({
            'gt_verts': gt_verts,
            'gt_faces': gt_faces
        }, save_path)
        
        # Delete original STL file
        try:
            os.remove(stl_path)
        except Exception as e:
            print(f"[Warning] Failed to delete {stl_path}: {e}")
        
        # Return entry for json
        return {'pt_path': save_filename, 'file_id': int(name) if name.isdigit() else None}
        
    except Exception as e:
        print(f"[Error] Failed to process {stl_path}: {e}")
        return None

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
            
            # 保存相对路径（相对于 base_output_dir）
            results.append({
                'base_pt_path': os.path.join(name, save_name), # Relative path inside base_output_dir
                'gt_pt_path': os.path.basename(pt_path), # 只保存文件名，假设 GT 文件在 pt_output_dir
                'base_faces': base_faces.shape[0]
            })
            
        return results
        
    except Exception as e:
        print(f"[Error] Failed to process {pt_path}: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="处理 thingi10k 数据：STL转PT，然后生成base meshes")
    parser.add_argument('--thingi10k_dir', type=str, required=True, 
                       help='Thingi10k 数据根目录（包含 extracted/raw_meshes 目录）')
    parser.add_argument('--pt_output_dir', type=str, required=True, 
                       help='输出 .pt 文件的目录（GT meshes）')
    parser.add_argument('--base_output_dir', type=str, required=True, 
                       help='输出 base meshes 的目录')
    parser.add_argument('--workers', type=int, default=8, help='并行处理的工作进程数')
    parser.add_argument('--min_faces', type=int, default=1000, help='Base mesh 最小面数')
    parser.add_argument('--max_faces', type=int, default=3000, help='Base mesh 最大面数')
    parser.add_argument('--skip_stl_to_pt', action='store_true', 
                       help='跳过 STL 转 PT 步骤（如果已经转换过）')
    args = parser.parse_args()
    
    os.makedirs(args.pt_output_dir, exist_ok=True)
    os.makedirs(args.base_output_dir, exist_ok=True)
    
    # 步骤 1: 将 STL 文件转换为 .pt 格式
    if not args.skip_stl_to_pt:
        print(f"\n步骤 1: 将 STL 文件转换为 .pt 格式...")
        
        # 查找 STL 文件
        # thingi10k 的 raw variant 解压后通常在 extracted/Thingi10K/raw_meshes/ 目录下
        raw_meshes_dirs = [
            os.path.join(args.thingi10k_dir, 'extracted', 'Thingi10K', 'raw_meshes'),
            os.path.join(args.thingi10k_dir, 'extracted', 'raw_meshes'),
            os.path.join(args.thingi10k_dir, 'raw_meshes'),
        ]
        
        raw_meshes_dir = None
        for d in raw_meshes_dirs:
            if os.path.exists(d):
                raw_meshes_dir = d
                print(f"找到 raw_meshes 目录: {raw_meshes_dir}")
                break
        
        if raw_meshes_dir is None:
            # 尝试搜索整个 extracted 目录
            print("在 extracted 目录中搜索 STL 文件...")
            extracted_dir = os.path.join(args.thingi10k_dir, 'extracted')
            if os.path.exists(extracted_dir):
                stl_files = list(Path(extracted_dir).rglob('*.stl'))
                stl_files += list(Path(extracted_dir).rglob('*.STL'))
                if stl_files:
                    print(f"找到 {len(stl_files)} 个 STL 文件")
                    # 使用第一个文件的目录作为参考
                    raw_meshes_dir = os.path.dirname(str(stl_files[0]))
                else:
                    raise FileNotFoundError(f"在 {extracted_dir} 中未找到 STL 文件")
            else:
                raise FileNotFoundError(f"未找到 extracted 目录: {extracted_dir}")
        
        # 直接扫描 raw_meshes_dir 中的所有 STL 文件
        print(f"扫描 {raw_meshes_dir} 中的所有 STL 文件...")
        stl_files = list(Path(raw_meshes_dir).glob('*.stl'))
        stl_files += list(Path(raw_meshes_dir).glob('*.STL'))
        
        if len(stl_files) == 0:
            # 如果当前目录没有，递归搜索子目录
            stl_files = list(Path(raw_meshes_dir).rglob('*.stl'))
            stl_files += list(Path(raw_meshes_dir).rglob('*.STL'))
        
        print(f"找到 {len(stl_files)} 个 STL 文件")
        
        # 准备转换任务
        convert_tasks = [(str(stl_file), args.pt_output_dir) for stl_file in stl_files]
        
        print(f"准备转换 {len(convert_tasks)} 个 STL 文件...")
        
        # 使用多进程转换
        pt_entries = []
        with Pool(processes=args.workers) as pool:
            for res in tqdm(pool.imap_unordered(convert_stl_to_pt, convert_tasks), 
                          total=len(convert_tasks), desc="转换 STL 到 PT"):
                if res is not None:
                    pt_entries.append(res)
        
        print(f"成功转换 {len(pt_entries)} 个文件")
        
        # 保存 train.json
        train_json = os.path.join(args.pt_output_dir, 'train.json')
        with open(train_json, 'w', encoding='utf-8') as f:
            json.dump([{'pt_path': e['pt_path']} for e in pt_entries], 
                     f, indent=2, ensure_ascii=False)
        print(f"训练索引文件已保存到: {train_json}")
    else:
        print(f"\n跳过 STL 转 PT 步骤（使用已存在的 .pt 文件）")
        # 读取已存在的 train.json
        train_json = os.path.join(args.pt_output_dir, 'train.json')
        if os.path.exists(train_json):
            with open(train_json, 'r', encoding='utf-8') as f:
                pt_entries = json.load(f)
            print(f"从 {train_json} 读取了 {len(pt_entries)} 个 .pt 文件")
        else:
            # 如果没有 train.json，尝试从目录中查找所有 .pt 文件
            pt_files = glob.glob(os.path.join(args.pt_output_dir, "*.pt"))
            pt_entries = [{'pt_path': os.path.basename(f)} for f in pt_files]
            print(f"从目录中找到 {len(pt_entries)} 个 .pt 文件")
    
    # 步骤 3: 生成 base meshes
    print(f"\n步骤 2: 生成 base meshes...")
    
    # 准备 base mesh 生成任务
    base_tasks = []
    for entry in pt_entries:
        pt_filename = entry['pt_path']
        pt_file = os.path.join(args.pt_output_dir, pt_filename)
        if os.path.exists(pt_file):
            base_tasks.append((pt_file, args.base_output_dir, args.min_faces, args.max_faces))
    
    print(f"准备为 {len(base_tasks)} 个模型生成 base meshes...")
    
    # 使用多进程生成 base meshes
    all_base_results = []
    with Pool(processes=args.workers) as pool:
        for res in tqdm(pool.imap_unordered(process_single_pt_file, base_tasks),
                      total=len(base_tasks), desc="生成 base meshes"):
            if res:
                all_base_results.extend(res)
    
    print(f"成功生成 {len(all_base_results)} 个 base meshes")
    
    # 保存 base meshes 索引
    base_json = os.path.join(args.base_output_dir, 'train_base.json')
    with open(base_json, 'w', encoding='utf-8') as f:
        json.dump(all_base_results, f, indent=2, ensure_ascii=False)
    print(f"Base meshes 索引文件已保存到: {base_json}")
    
    print(f"\n所有处理完成！")
    print(f"GT meshes (.pt): {args.pt_output_dir}")
    print(f"Base meshes: {args.base_output_dir}")

if __name__ == '__main__':
    main()

