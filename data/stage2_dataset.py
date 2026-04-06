import os
import glob
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
                 preprocessed_base_mesh_dir=None, use_preprocess_base_mesh=False, 
                 preload_ram=False, lmdb_path=None, use_scan_normal=False,
                 dataset_type='human',
                 sharp_edge_sampling=False,
                 sharp_edge_angle_threshold=10.0,
                 sharp_edge_ratio=0.5):
        """
        Args:
            data_root: 预处理数据目录
            point_num: 采样点数
            base_faces_min/max: 随机简化目标面数范围
            backend: 'open3d' or 'pyfqmr'
            debug_export: 是否导出第0个样本的中间结果用于检查
            preprocessed_base_mesh_dir: Base Mesh 预处理目录 (可选)
            use_preprocess_base_mesh: 是否使用预处理的 Base Mesh
            preload_ram: 是否将所有数据预加载到内存中 (解决 IO 瓶颈)
            lmdb_path: 预打包的 LMDB 数据库路径 (推荐使用)
            dataset_type: 'human' (默认) 或 'thingi10k'
            sharp_edge_sampling: 是否启用 sharp edge biased sampling
            sharp_edge_angle_threshold: 二面角阈值 (度)
            sharp_edge_ratio: sharp 点占总采样点的比例
        """
        self.data_root = data_root
        self.split = split
        self.dataset_type = dataset_type
        self.point_num = point_num
        self.base_faces_range = (base_faces_min, base_faces_max)
        self.backend = backend
        self.debug_export = debug_export
        self.use_preprocess_base_mesh = use_preprocess_base_mesh
        self.preprocessed_base_mesh_dir = preprocessed_base_mesh_dir
        self.preload_ram = preload_ram
        self.lmdb_path = lmdb_path
        self.use_scan_normal = use_scan_normal
        self.sharp_edge_sampling = sharp_edge_sampling
        self.sharp_edge_angle_threshold = sharp_edge_angle_threshold
        self.sharp_edge_ratio = sharp_edge_ratio
        self.lmdb_env = None
        
        if self.sharp_edge_sampling:
            print(f"[Dataset] Sharp edge sampling ENABLED: threshold={self.sharp_edge_angle_threshold}°, ratio={self.sharp_edge_ratio}")
        
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
            if self.dataset_type == 'thingi10k':
                # Thingi10k: 如果没有索引文件，自动扫描目录中的 .pt 文件生成列表
                print(f"[Dataset] {json_path} 不存在，自动扫描 {data_root} 中的 .pt 文件...")
                pt_files = sorted(glob.glob(os.path.join(data_root, "*.pt")))
                if len(pt_files) == 0:
                    raise FileNotFoundError(f"No .pt files found in {data_root}")
                self.file_list = [{'pt_path': os.path.basename(f)} for f in pt_files]
                print(f"[Dataset] 自动扫描到 {len(self.file_list)} 个 .pt 文件")
            else:
                raise FileNotFoundError(f"Index file not found: {json_path}")
        else:
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
            
        print(f"[Dataset] Loaded {len(self.file_list)} samples for {split} | Type: {self.dataset_type} | Backend: {self.backend} | Preprocessed Base: {self.use_preprocess_base_mesh}")

        # --- LMDB Setup ---
        if self.lmdb_path and os.path.exists(self.lmdb_path):
             import lmdb
             # Open read-only, no lock
             self.lmdb_env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
             print(f"[Dataset] Using LMDB: {self.lmdb_path}")

        # --- Preload to RAM ---
        self.gt_cache = {}
        self.base_cache = {}
        
        # Only preload if NOT using LMDB (or if user explicitly wants to load LMDB into RAM which is usually unnecessary as OS does it)
        # But if preload_ram is True, we can load FROM LMDB to RAM to be even faster.
        if self.preload_ram:
             from tqdm import tqdm
             if self.lmdb_env:
                 print("[Dataset] Preloading all data from LMDB to RAM...")
                 import pickle
                 with self.lmdb_env.begin(write=False) as txn:
                     # Load GT
                     for i in tqdm(range(len(self.file_list))):
                         key = f"gt_{i}".encode()
                         buf = txn.get(key)
                         if buf:
                             pt_path = os.path.join(self.data_root, self.file_list[i]['pt_path'])
                             self.gt_cache[pt_path] = pickle.loads(buf)
                     
                     # Load Base
                     if self.use_preprocess_base_mesh and self.base_file_list:
                         for entry in tqdm(self.base_file_list):
                             rel_path = entry['base_pt_path']
                             key = f"base_{rel_path}".encode()
                             buf = txn.get(key)
                             if buf:
                                 full_path = os.path.join(self.preprocessed_base_mesh_dir, rel_path)
                                 self.base_cache[full_path] = pickle.loads(buf)
                 print(f"[Dataset] LMDB Preload complete.")
                 
             else:
                print("[Dataset] Preloading all data to RAM from DISK... This may take a while.")
                from tqdm import tqdm
                
                # 1. Preload GT
                print("  - Loading GT Meshes...")
                for item in tqdm(self.file_list):
                    pt_path = os.path.join(self.data_root, item['pt_path'])
                    if pt_path not in self.gt_cache:
                        try:
                            self.gt_cache[pt_path] = torch.load(pt_path, weights_only=False)
                        except Exception as e:
                            print(f"[Warning] Failed to load {pt_path}: {e}")
                
                # 2. Preload Base (if used)
                if self.use_preprocess_base_mesh and self.base_file_list:
                    print("  - Loading Base Meshes...")
                    for entry in tqdm(self.base_file_list):
                        base_pt_full_path = os.path.join(self.preprocessed_base_mesh_dir, entry['base_pt_path'])
                        if base_pt_full_path not in self.base_cache:
                            try:
                                self.base_cache[base_pt_full_path] = torch.load(base_pt_full_path, weights_only=False)
                            except Exception as e:
                                print(f"[Warning] Failed to load {base_pt_full_path}: {e}")
                                
                print(f"[Dataset] Preload complete. GT Cache: {len(self.gt_cache)}, Base Cache: {len(self.base_cache)}")

    @staticmethod
    def _sample_on_sharp_edges(mesh, sharp_edge_verts, sharp_face_pairs, target_num):
        """Interpolate points along sharp edges (Dora-style)."""
        verts = mesh.vertices
        face_normals = mesh.face_normals
        E = len(sharp_edge_verts)
        if E == 0:
            return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)

        unique_idx = np.unique(sharp_edge_verts.ravel())
        known_pts = verts[unique_idx]
        known_nrm = mesh.vertex_normals[unique_idx]
        num_known = len(known_pts)
        num_need = max(target_num - num_known, 0)

        n1 = face_normals[sharp_face_pairs[:, 0]]
        n2 = face_normals[sharp_face_pairs[:, 1]]
        edge_normals = 0.5 * (n1 + n2)
        edge_normals /= np.linalg.norm(edge_normals, axis=1, keepdims=True).clip(1e-8)

        start = verts[sharp_edge_verts[:, 0]]
        end = verts[sharp_edge_verts[:, 1]]

        interp_pts, interp_nrm = [], []
        if num_need > 0:
            if num_need >= E:
                per_edge = num_need // E
                for j in range(1, per_edge + 1):
                    t = j / (per_edge + 1)
                    interp_pts.append((1 - t) * start + t * end)
                    interp_nrm.append(edge_normals)
                remainder = num_need % E
            else:
                remainder = num_need

            if remainder > 0:
                rng = np.random.default_rng()
                sel = rng.choice(E, remainder, replace=(remainder > E))
                t = np.random.rand(remainder, 1).astype(np.float32)
                interp_pts.append((1 - t) * start[sel] + t * end[sel])
                interp_nrm.append(edge_normals[sel])

        all_pts = [known_pts] + interp_pts
        all_nrm = [known_nrm] + interp_nrm
        all_pts = np.concatenate(all_pts, axis=0).astype(np.float32)
        all_nrm = np.concatenate(all_nrm, axis=0).astype(np.float32)

        if len(all_pts) > target_num:
            idx = np.random.choice(len(all_pts), target_num, replace=False)
            all_pts, all_nrm = all_pts[idx], all_nrm[idx]
        return all_pts, all_nrm

    def _sample_scan_points(self, tm_mesh):
        """Sample scan points: sharp-edge biased or uniform."""
        if not self.sharp_edge_sampling:
            pts, fid = trimesh.sample.sample_surface(tm_mesh, self.point_num)
            pts = pts.astype(np.float32)
            nrm = tm_mesh.face_normals[fid].astype(np.float32) if self.use_scan_normal else None
            return pts, nrm

        threshold_rad = np.deg2rad(self.sharp_edge_angle_threshold)
        angles = tm_mesh.face_adjacency_angles
        mask = angles >= threshold_rad

        num_sharp_target = int(self.point_num * self.sharp_edge_ratio)
        num_uniform_target = self.point_num - num_sharp_target

        sharp_edge_verts = tm_mesh.face_adjacency_edges[mask]
        sharp_face_pairs = tm_mesh.face_adjacency[mask]
        num_sharp_edges = len(sharp_edge_verts)

        min_sharp_fallback = 0.05
        if num_sharp_edges == 0 or num_sharp_target == 0:
            pts, fid = trimesh.sample.sample_surface(tm_mesh, self.point_num)
            pts = pts.astype(np.float32)
            nrm = tm_mesh.face_normals[fid].astype(np.float32) if self.use_scan_normal else None
            return pts, nrm

        sharp_pts, sharp_nrm = self._sample_on_sharp_edges(
            tm_mesh, sharp_edge_verts, sharp_face_pairs, num_sharp_target
        )

        if len(sharp_pts) < num_sharp_target * min_sharp_fallback:
            pts, fid = trimesh.sample.sample_surface(tm_mesh, self.point_num)
            pts = pts.astype(np.float32)
            nrm = tm_mesh.face_normals[fid].astype(np.float32) if self.use_scan_normal else None
            return pts, nrm

        num_uniform_target = self.point_num - len(sharp_pts)
        uni_pts, fid = trimesh.sample.sample_surface(tm_mesh, num_uniform_target)
        uni_pts = uni_pts.astype(np.float32)

        scan_points = np.concatenate([sharp_pts, uni_pts], axis=0)
        if self.use_scan_normal:
            uni_nrm = tm_mesh.face_normals[fid].astype(np.float32)
            scan_normals = np.concatenate([sharp_nrm, uni_nrm], axis=0)
        else:
            scan_normals = None
        return scan_points, scan_normals

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        t0 = time.time()
        item = self.file_list[idx]
        pt_path = os.path.join(self.data_root, item['pt_path'])
        
        # Load GT Data
        # GT Data is needed for sampling scan points (always)
        if self.preload_ram and pt_path in self.gt_cache:
            data = self.gt_cache[pt_path]
        elif self.lmdb_env:
            import pickle
            with self.lmdb_env.begin(write=False) as txn:
                key = f"gt_{idx}".encode()
                buf = txn.get(key)
                if buf:
                    data = pickle.loads(buf)
                else:
                    # Fallback
                    data = torch.load(pt_path, weights_only=False)
        else:
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
        
        # 1. Online Sampling
        tm_mesh = trimesh.Trimesh(vertices=gt_verts_np, faces=gt_faces_np, process=False)
        scan_points, scan_normals = self._sample_scan_points(tm_mesh)
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
                    if self.preload_ram and base_pt_full_path in self.base_cache:
                        base_data = self.base_cache[base_pt_full_path]
                    elif self.lmdb_env:
                         import pickle
                         with self.lmdb_env.begin(write=False) as txn:
                             key = f"base_{selected['base_pt_path']}".encode()
                             buf = txn.get(key)
                             if buf:
                                 base_data = pickle.loads(buf)
                             else:
                                 base_data = torch.load(base_pt_full_path, weights_only=False)
                    else:
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
        
        # # Debug Export (First item only)
        # if self.debug_export and idx == 0:
        #     os.makedirs("debug_dataset", exist_ok=True)
        #     # For saving, reuse Open3D
        #     dbg_mesh = o3d.geometry.TriangleMesh()
        #     dbg_mesh.vertices = o3d.utility.Vector3dVector(base_verts_np)
        #     dbg_mesh.triangles = o3d.utility.Vector3iVector(base_faces_np.astype(np.int32))
        #     o3d.io.write_triangle_mesh("debug_dataset/base_mesh_debug.obj", dbg_mesh)
            
        #     pc = o3d.geometry.PointCloud()
        #     pc.points = o3d.utility.Vector3dVector(scan_points)
        #     o3d.io.write_point_cloud("debug_dataset/scan_points_debug.ply", pc)
        #     print(f"[Dataset Debug] Exported base mesh (V={len(base_verts_np)}, F={len(base_faces_np)}) and scan points.")
        
        result = {
            'scan_points': torch.from_numpy(scan_points), # (P, 3)
            'base_verts': torch.from_numpy(base_verts_np), # (V, 3)
            'base_faces': torch.from_numpy(base_faces_np), # (F, 3)
            'base_normals': torch.from_numpy(base_normals_np), # (V, 3)
            'gt_verts': torch.from_numpy(gt_verts_np), # Need to return normalized GT!
            'gt_faces': data['gt_faces'] # Faces unchanged
        }
        if self.use_scan_normal and scan_normals is not None:
            result['scan_normals'] = torch.from_numpy(scan_normals)  # (P, 3)
        return result

def stage2_collate_fn(batch):
    """
    Collate function for variable size meshes.
    Scan points are stacked.
    Meshes are kept as lists (or could be packed Meshes).
    """
    scan_points = torch.stack([item['scan_points'] for item in batch])
    has_scan_normal = ('scan_normals' in batch[0])
    scan_normals = torch.stack([item['scan_normals'] for item in batch]) if has_scan_normal else None
    
    base_verts_list = [item['base_verts'] for item in batch]
    base_faces_list = [item['base_faces'] for item in batch]
    base_normals_list = [item['base_normals'] for item in batch]
    
    gt_verts_list = [item['gt_verts'] for item in batch]
    gt_faces_list = [item['gt_faces'] for item in batch]
    
    result = {
        'scan_points': scan_points,
        'base_verts': base_verts_list,
        'base_faces': base_faces_list,
        'base_normals': base_normals_list,
        'gt_verts': gt_verts_list,
        'gt_faces': gt_faces_list
    }
    if has_scan_normal:
        result['scan_normals'] = scan_normals
    return result