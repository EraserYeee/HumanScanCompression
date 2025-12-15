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
                 backend='open3d', debug_export=False,
                 preprocessed_base_mesh_dir=None, use_preprocess_base_mesh=False):
        """
        Args:
            data_root: 预处理数据目录
            point_num: 采样点数
            base_faces_min/max: 随机简化目标面数范围
            backend: 'open3d' or 'pyfqmr'
            debug_export: 是否导出第0个样本的中间结果用于检查
            preprocessed_base_mesh_dir: Base Mesh 预处理目录 (可选)
            use_preprocess_base_mesh: 是否使用预处理的 Base Mesh
        """
        self.data_root = data_root
        self.split = split
        self.point_num = point_num
        self.base_faces_range = (base_faces_min, base_faces_max)
        self.backend = backend
        self.debug_export = debug_export
        self.use_preprocess_base_mesh = use_preprocess_base_mesh
        self.preprocessed_base_mesh_dir = preprocessed_base_mesh_dir
        
        if self.backend == 'pyfqmr' and not _HAS_PYFQMR:
            print("[Warning] pyfqmr not found, falling back to open3d")
            self.backend = 'open3d'
        
        # Pre-instantiate simplifier for pyfqmr if needed (only if not using preprocessed)
        if not self.use_preprocess_base_mesh and self.backend == 'pyfqmr':
             self.simplifier = pyfqmr.Simplify()
        
        # If using preprocessed base meshes, load the base mesh index
        if self.use_preprocess_base_mesh:
             if self.preprocessed_base_mesh_dir is None:
                 raise ValueError("use_preprocess_base_mesh is True but preprocessed_base_mesh_dir is None")
             
             base_json_path = os.path.join(self.preprocessed_base_mesh_dir, f"{split}_base.json")
             if not os.path.exists(base_json_path):
                 raise FileNotFoundError(f"Base mesh index not found: {base_json_path}")
             
             with open(base_json_path, 'r') as f:
                 self.base_file_list = json.load(f)
             
             # Group by GT path to allow random selection for the same GT
             # We want to maintain the same length as the original dataset (one sample per GT mesh)
             # But we can pick a random base mesh for it.
             # Wait, the dataset length should probably match the GT dataset length.
             # Let's load the original GT index first.
        
        json_path = os.path.join(data_root, f"{split}.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Index file not found: {json_path}")
            
        with open(json_path, 'r') as f:
            self.file_list = json.load(f)
            
        # Build a mapping from GT pt filename to list of base meshes
        if self.use_preprocess_base_mesh:
            self.gt_to_base = {}
            for entry in self.base_file_list:
                # entry['gt_pt_path'] might be absolute or relative
                # We need to match it with self.file_list['pt_path']
                # Let's use basename for matching to be safe
                gt_basename = os.path.basename(entry['gt_pt_path'])
                if gt_basename not in self.gt_to_base:
                    self.gt_to_base[gt_basename] = []
                self.gt_to_base[gt_basename].append(entry)
            
        print(f"[Dataset] Loaded {len(self.file_list)} samples for {split} | Backend: {self.backend} | Preprocessed Base: {self.use_preprocess_base_mesh}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        t0 = time.time()
        item = self.file_list[idx]
        pt_path = os.path.join(self.data_root, item['pt_path'])
        
        # Load GT Data
        # GT Data is needed for sampling scan points (always)
        data = torch.load(pt_path, weights_only=False)
        gt_verts_np = data['gt_verts'].numpy()
        gt_faces_np = data['gt_faces'].numpy()
        
        # --- Normalization Check & Implementation ---
        # Normalize GT mesh to unit sphere (or bbox)
        # Center the mesh
        bbox_min = gt_verts_np.min(axis=0)
        bbox_max = gt_verts_np.max(axis=0)
        center = (bbox_min + bbox_max) / 2
        gt_verts_np = gt_verts_np - center
        
        # Scale to unit sphere (radius = 1)
        scale = np.max(np.linalg.norm(gt_verts_np, axis=1))
        # Add small epsilon to avoid div by zero
        if scale < 1e-6: scale = 1.0
        gt_verts_np = gt_verts_np / scale
        # --------------------------------------------
        
        t1 = time.time()
        
        # 1. Online Sampling (Trimesh is good for sampling)
        tm_mesh = trimesh.Trimesh(vertices=gt_verts_np, faces=gt_faces_np, process=False)
        scan_points, _ = trimesh.sample.sample_surface(tm_mesh, self.point_num)
        scan_points = scan_points.astype(np.float32)
        t2 = time.time()
        
        base_verts_np = None
        base_faces_np = None
        base_normals_np = None
        
        if self.use_preprocess_base_mesh:
            # 2a. Load Preprocessed Base Mesh
            # Find candidate base meshes for this GT
            gt_basename = os.path.basename(item['pt_path'])
            candidates = self.gt_to_base.get(gt_basename, [])
            
            if len(candidates) > 0:
                # Randomly select one using a seed based on index and time? 
                # Or just random.choice (which is seeded by global random state)
                # Requirement: "randomly select one ... use seed to ensure reproducibility"
                # If we rely on np.random, we should ensure the seed is set in the main worker process.
                # In PyTorch dataloader workers, seeds are usually handled if worker_init_fn is set.
                # Here we just use np.random.choice.
                selected = np.random.choice(candidates)
                
                # Load the base mesh .pt
                # base_pt_path is relative to preprocessed_base_mesh_dir
                # Note: in preprocess_base_meshes.py, we stored relative path: 'meshname/xxx.pt'
                base_pt_full_path = os.path.join(self.preprocessed_base_mesh_dir, selected['base_pt_path'])
                
                try:
                    base_data = torch.load(base_pt_full_path, weights_only=False)
                    base_verts_np = base_data['base_verts'].numpy().astype(np.float32)
                    base_faces_np = base_data['base_faces'].numpy().astype(np.int64)
                    
                    # --- Normalization for Base Mesh (Apply SAME transform as GT) ---
                    # Note: We must use the SAME center and scale as calculated from GT
                    base_verts_np = (base_verts_np - center) / scale
                    # --------------------------------------------------------------
                    
                    # Compute normals for base mesh (Trimesh or Open3D)
                    # Preprocessed file might not have normals saved.
                    # Let's compute them on the fly.
                    mesh_o3d = o3d.geometry.TriangleMesh()
                    mesh_o3d.vertices = o3d.utility.Vector3dVector(base_verts_np)
                    mesh_o3d.triangles = o3d.utility.Vector3iVector(base_faces_np.astype(np.int32))
                    mesh_o3d.compute_vertex_normals()
                    base_normals_np = np.asarray(mesh_o3d.vertex_normals, dtype=np.float32)
                    
                except Exception as e:
                    print(f"[Error] Failed to load base mesh {base_pt_full_path}: {e}. Fallback to online.")
                    # Fallback will happen below if variables are None
            else:
                print(f"[Warning] No preprocessed base mesh found for {gt_basename}. Fallback to online.")

        if base_verts_np is None:
            # 2b. Online Simplification (Fallback or Default)
            # Random target faces
            # print("Fallback to online simplification")
            # exit()
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
                
                # --- Normalization for Online Base Mesh ---
                # Since we simplified the *normalized* GT mesh, the base mesh is already normalized!
                # We passed `gt_verts_np` (which is now normalized) to simplifier.
                # So no extra action needed here.
                # ------------------------------------------
                
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
            'gt_verts': torch.from_numpy(gt_verts_np), # Need to return normalized GT!
            'gt_faces': data['gt_faces'] # Faces unchanged
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
