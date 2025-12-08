import os
import torch
import json
import trimesh
import numpy as np
import open3d as o3d
import time
from torch.utils.data import Dataset
from pytorch3d.structures import Meshes

# Try to import pyfqmr
try:
    import pyfqmr
    _HAS_PYFQMR = True
except ImportError:
    _HAS_PYFQMR = False

class ScanToMeshDataset(Dataset):
    """
    Stage 2 训练数据集 (Online Simplification & Sampling).
    
    1. 加载预处理好的 GT Mesh (.pt).
    2. Online: 使用 Open3D 或 PyFQMR 随机简化 GT Mesh 得到 Base Mesh.
    3. Online: 使用 Trimesh 从 GT Mesh 表面采样 Scan Points (320k).
    """
    def __init__(self, data_root, split='train', point_num=320000, 
                 base_faces_min=2000, base_faces_max=6000, 
                 backend='open3d', debug_export=False):
        """
        Args:
            data_root: 预处理数据目录
            point_num: 采样点数
            base_faces_min/max: 随机简化目标面数范围
            backend: 'open3d' or 'pyfqmr'
            debug_export: 是否导出第0个样本的中间结果用于检查
        """
        self.data_root = data_root
        self.split = split
        self.point_num = point_num
        self.base_faces_range = (base_faces_min, base_faces_max)
        self.backend = backend
        self.debug_export = debug_export
        
        if self.backend == 'pyfqmr' and not _HAS_PYFQMR:
            print("[Warning] pyfqmr not found, falling back to open3d")
            self.backend = 'open3d'
        
        # Pre-instantiate simplifier for pyfqmr if needed
        if self.backend == 'pyfqmr':
             self.simplifier = pyfqmr.Simplify()
        
        json_path = os.path.join(data_root, f"{split}.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Index file not found: {json_path}")
            
        with open(json_path, 'r') as f:
            self.file_list = json.load(f)
            
        print(f"[Dataset] Loaded {len(self.file_list)} samples for {split} | Backend: {self.backend}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        t0 = time.time()
        item = self.file_list[idx]
        pt_path = os.path.join(self.data_root, item['pt_path'])
        
        # Load GT Data
        data = torch.load(pt_path, weights_only=False)
        gt_verts_np = data['gt_verts'].numpy()
        gt_faces_np = data['gt_faces'].numpy()
        t1 = time.time()
        
        # 1. Online Sampling (Trimesh is good for sampling)
        tm_mesh = trimesh.Trimesh(vertices=gt_verts_np, faces=gt_faces_np, process=False)
        scan_points, _ = trimesh.sample.sample_surface(tm_mesh, self.point_num)
        scan_points = scan_points.astype(np.float32)
        t2 = time.time()
        
        # 2. Online Simplification
        # Random target faces
        target_faces = np.random.randint(self.base_faces_range[0], self.base_faces_range[1] + 1)
        
        if self.backend == 'pyfqmr':
            # PyFQMR
            # Re-using the simplifier object if possible (though it's stateful per mesh, so setMesh resets it)
            self.simplifier.setMesh(gt_verts_np, gt_faces_np)
            self.simplifier.simplify_mesh(target_count=target_faces, aggressiveness=7, preserve_border=True, verbose=0)
            base_verts_np, base_faces_np, base_normals_np = self.simplifier.getMesh()
            base_verts_np = base_verts_np.astype(np.float32)
            base_faces_np = base_faces_np.astype(np.int64)
            # PyFQMR might return normals, but let's ensure they are float32
            base_normals_np = base_normals_np.astype(np.float32)
            
        else:
            # Open3D
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(gt_verts_np)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(gt_faces_np.astype(np.int32))
            
            base_mesh_o3d = o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
            base_mesh_o3d.compute_vertex_normals()
            
            base_verts_np = np.asarray(base_mesh_o3d.vertices, dtype=np.float32)
            base_faces_np = np.asarray(base_mesh_o3d.triangles, dtype=np.int64)
            base_normals_np = np.asarray(base_mesh_o3d.vertex_normals, dtype=np.float32)
            
        t3 = time.time()
        
        # print(f"[Dataset Worker] Idx {idx}: Total {t3-t0:.4f}s | Load {t1-t0:.4f}s | Sample {t2-t1:.4f}s | Simplify ({self.backend}) {t3-t2:.4f}s")
        
        # Debug Export (First item only)
        if self.debug_export and idx == 0:
            os.makedirs("debug_dataset", exist_ok=True)
            # For saving, reuse Open3D
            dbg_mesh = o3d.geometry.TriangleMesh()
            dbg_mesh.vertices = o3d.utility.Vector3dVector(base_verts_np)
            dbg_mesh.triangles = o3d.utility.Vector3iVector(base_faces_np.astype(np.int32))
            o3d.io.write_triangle_mesh("debug_dataset/base_mesh_debug.obj", dbg_mesh)
            
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(scan_points)
            o3d.io.write_point_cloud("debug_dataset/scan_points_debug.ply", pc)
            print(f"[Dataset Debug] Exported base mesh (V={len(base_verts_np)}, F={len(base_faces_np)}) and scan points.")
        
        return {
            'scan_points': torch.from_numpy(scan_points), # (P, 3)
            'base_verts': torch.from_numpy(base_verts_np), # (V, 3)
            'base_faces': torch.from_numpy(base_faces_np), # (F, 3)
            'base_normals': torch.from_numpy(base_normals_np), # (V, 3)
            'gt_verts': data['gt_verts'],
            'gt_faces': data['gt_faces']
        }

def stage2_collate_fn(batch):
    """
    Collate function for variable size meshes.
    Scan points are stacked.
    Meshes are kept as lists (or could be packed Meshes).
    """
    scan_points = torch.stack([item['scan_points'] for item in batch])
    
    base_verts_list = [item['base_verts'] for item in batch]
    base_faces_list = [item['base_faces'] for item in batch]
    base_normals_list = [item['base_normals'] for item in batch]
    
    gt_verts_list = [item['gt_verts'] for item in batch]
    gt_faces_list = [item['gt_faces'] for item in batch]
    
    return {
        'scan_points': scan_points,
        'base_verts': base_verts_list,
        'base_faces': base_faces_list,
        'base_normals': base_normals_list,
        'gt_verts': gt_verts_list,
        'gt_faces': gt_faces_list
    }
