import os
import sys
import argparse
import torch
import trimesh
import numpy as np
import yaml
import json
import pickle
from tqdm import tqdm
from PIL import Image

# Add current directory to path for models and utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import Stage 2 components
from models.pipeline import Stage2Pipeline
from data.stage2_dataset import ScanToMeshDataset, stage2_collate_fn
from utils.render import DifferentiableNormalRenderer

# --- EdgeRunner Integration ---
EDGE_RUNNER_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'EdgeRunner')
sys.path.insert(0, EDGE_RUNNER_ROOT)
sys.path.insert(0, os.path.join(EDGE_RUNNER_ROOT, 'models'))

try:
    from core.models import LMM
    from core.options import config_defaults
    from core.utils import load_mesh, get_tokenizer, monkey_patch_transformers
    from core.transformer.point import PointEncoderEmbed
    from kiui.mesh_utils import clean_mesh
    from safetensors.torch import load_file
    monkey_patch_transformers()
    _HAS_EDGERUNNER = True
except ImportError as e:
    print(f"[Warning] EdgeRunner dependencies not found: {e}")
    _HAS_EDGERUNNER = False

class EdgeRunnerWrapper:
    def __init__(self, checkpoint_path, config_name='ArAE', device='cuda'):
        self.device = torch.device(device)
        self.opt = config_defaults[config_name]
        self.opt.checkpointing = False
        
        print(f'[EdgeRunner] Loading model from {checkpoint_path}')
        self.model = LMM(self.opt)
        
        if checkpoint_path.endswith('safetensors'):
            ckpt = load_file(checkpoint_path, device='cpu')
        else:
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        self.model.load_state_dict(ckpt, strict=False)
        self.model = self.model.half().eval().to(self.device)
        self.tokenizer, _ = get_tokenizer(self.opt)
        
        self.encoder = PointEncoderEmbed(
            hidden_dim=self.opt.point_hidden_dim,
            num_heads=self.opt.point_num_heads,
            latent_size=self.opt.point_latent_size,
            latent_dim=self.opt.point_latent_dim,
            gradient_checkpointing=False,
        )
        encoder_state_dict = {}
        for key, value in ckpt.items():
            if key.startswith('point_encoder.'):
                new_key = key.replace('point_encoder.', '')
                encoder_state_dict[new_key] = value
        self.encoder.load_state_dict(encoder_state_dict, strict=True)
        self.encoder = self.encoder.half().eval().to(self.device)

    @torch.no_grad()
    def encode_points(self, points):
        points_tensor = torch.from_numpy(points).unsqueeze(0).float().to(self.device)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            posterior = self.encoder(points_tensor)
            embedding = posterior.mode()
        return embedding

    @torch.no_grad()
    def generate_mesh(self, embedding, num_faces=2000):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            original_cond_mode = self.model.opt.cond_mode
            self.model.opt.cond_mode = 'point_latent'
            meshes, _ = self.model.generate(
                embedding.half(),
                num_faces=num_faces,
                tokenizer=self.tokenizer,
                clean=True
            )
            self.model.opt.cond_mode = original_cond_mode
        return meshes[0]

# --- Utilities ---

def stage2_normalize(verts_np):
    bbox_min = verts_np.min(axis=0)
    bbox_max = verts_np.max(axis=0)
    center = (bbox_min + bbox_max) / 2
    verts_centered = verts_np - center
    scale = np.max(np.linalg.norm(verts_centered, axis=1))
    if scale < 1e-6: scale = 1.0
    verts_norm = verts_centered / scale
    return verts_norm, center, scale

def apply_normalization(verts_np, center, scale):
    return (verts_np - center) / scale

def load_stage2_config(checkpoint_path):
    config_path = os.path.join(os.path.dirname(checkpoint_path), "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description="Stage 2 Inference Script")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to Stage 2 checkpoint (.pth)")
    parser.add_argument(
        '--mode',
        type=str,
        choices=['gt_simplified', 'edgerunner', 'given_simplified'],
        default='gt_simplified'
    )
    parser.add_argument('--data_source', type=str, choices=['dataset', 'single'], default='dataset')
    parser.add_argument('--input_path', type=str, help="Path to dataset processed_dir or single obj")
    parser.add_argument('--edgerunner_ckpt', type=str, help="Path to EdgeRunner checkpoint")
    parser.add_argument('--test_num_face', type=int, default=2000)
    parser.add_argument('--use_cached', action='store_true')
    parser.add_argument('--output_dir', type=str, default="test_results")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--render_size', type=int)
    parser.add_argument('--render_views', type=int)
    
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Config & Model
    config = load_stage2_config(args.checkpoint)
    exp_name = config.get('experiment_name', 'default_exp')
    
    # Final Output Structure
    final_output_dir = os.path.join(args.output_dir, exp_name)
    fine_mesh_out_dir = os.path.join(final_output_dir, "fine_meshs")
    os.makedirs(fine_mesh_out_dir, exist_ok=True)
    
    # Special Folders
    EXAMPLES_DIR = "/mnt/lab/data/yeruisi/data/compression/examples"
    GT_EXPORT_DIR = os.path.join(EXAMPLES_DIR, "GT_meshes")
    ER_CACHE_DIR = os.path.join(EXAMPLES_DIR, "EdgeRunner_basemesh")
    
    SHOULD_EXPORT_GT = False
    if not os.path.exists(GT_EXPORT_DIR):
        os.makedirs(GT_EXPORT_DIR, exist_ok=True)
        SHOULD_EXPORT_GT = True

    if args.render_size: config['render']['image_size'] = args.render_size
    if args.render_views: config['render']['views_per_sample'] = args.render_views
    
    model = Stage2Pipeline(config=config['model']).to(device)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    
    er_wrapper = None
    if args.mode == 'edgerunner':
        if not _HAS_EDGERUNNER:
            raise ImportError("EdgeRunner dependencies not found.")
        if not args.use_cached:
            if not args.edgerunner_ckpt:
                raise ValueError("edgerunner mode requires --edgerunner_ckpt if not using cache")
            er_wrapper = EdgeRunnerWrapper(args.edgerunner_ckpt, device=args.device)

    # 3. Setup Data
    data_root = args.input_path if args.input_path else config['data']['processed_dir']
    samples = []
    base_usage_info = []
    
    if args.data_source == 'dataset':
        if args.mode == 'gt_simplified':
            test_base_json = os.path.join(data_root, 'test_base.json')
            if os.path.exists(test_base_json):
                with open(test_base_json, 'r') as f:
                    file_list = json.load(f)
                for item in file_list:
                    if "_base_0_" in item['base_pt_path']:
                        samples.append({
                            'name': os.path.splitext(os.path.basename(item['gt_pt_path']))[0],
                            'gt_path': item['gt_pt_path'],
                            'base_path': os.path.join(config['data']['preprocessed_base_mesh_dir'], item['base_pt_path'])
                        })
            else:
                test_json = os.path.join(data_root, 'test.json')
                with open(test_json, 'r') as f:
                    file_list = json.load(f)
                for item in file_list:
                    samples.append({
                        'name': os.path.splitext(os.path.basename(item['pt_path']))[0],
                        'gt_path': os.path.join(data_root, item['pt_path']),
                        'base_path': None
                    })
        elif args.mode == 'edgerunner':
            if args.use_cached:
                er_json = os.path.join(ER_CACHE_DIR, "base_meshs.json")
                if os.path.exists(er_json):
                    with open(er_json, 'r') as f:
                        samples = json.load(f)
                else:
                    raise FileNotFoundError(f"EdgeRunner cache not found at {er_json}")
            else:
                test_json = os.path.join(data_root, 'test.json')
                with open(test_json, 'r') as f:
                    file_list = json.load(f)
                for item in file_list:
                    samples.append({
                        'name': os.path.splitext(os.path.basename(item['pt_path']))[0],
                        'gt_path': os.path.join(data_root, item['pt_path'])
                    })
        elif args.mode == 'given_simplified':
            raise ValueError("mode=given_simplified 目前仅支持 data_source=single，请使用 --data_source single")
    else:
        # 单样本模式
        if args.mode == 'given_simplified':
            if not args.input_path:
                raise ValueError("mode=given_simplified 需要指定 --input_path=gt_step_xxxx.obj")
            gt_path = os.path.abspath(args.input_path)
            if not os.path.exists(gt_path):
                raise FileNotFoundError(f"输入的 gt mesh 不存在: {gt_path}")
            name = os.path.splitext(os.path.basename(gt_path))[0]
            # 推断对应的 base mesh 文件名
            suffix = ""
            if name.startswith("gt_step_"):
                suffix = name[len("gt_step_"):]
            elif name.startswith("gt_"):
                suffix = name[len("gt_"):]
            if suffix:
                base_filename = f"base_step_{suffix}.obj"
            else:
                base_filename = name.replace("gt_", "base_", 1) + ".obj" if name.startswith("gt_") else f"base_{name}.obj"
            base_path = os.path.join(os.path.dirname(gt_path), base_filename)
            base_path = os.path.abspath(base_path)
            if not os.path.exists(base_path):
                raise FileNotFoundError(
                    f"mode=given_simplified 期望在与 gt 同目录下找到 {os.path.basename(base_path)}，但未找到。"
                )
            samples.append({
                'name': name,
                'gt_path': gt_path,
                'base_path': base_path
            })
        else:
            # 原有单样本逻辑：只给一个扫描，基于扫描在线简化 base mesh
            samples.append({
                'name': os.path.splitext(os.path.basename(args.input_path))[0],
                'gt_path': args.input_path,
                'base_path': None
            })

    renderer = DifferentiableNormalRenderer(
        image_size=config['render']['image_size'],
        device=device,
        cameras_per_batch=config['render']['views_per_sample'],
        dist_multiplier=config['render'].get('dist_multiplier', 1.0)
    )

    # 5. Inference Loop
    for sample in tqdm(samples, desc="Inference"):
        name = sample['name']
        gt_path = sample['gt_path']
        
        # Load GT
        if gt_path.endswith('.pt'):
            gt_data = torch.load(gt_path, map_location='cpu', weights_only=False)
            gt_v, gt_f = gt_data['gt_verts'].numpy(), gt_data['gt_faces'].numpy()
        else:
            gt_mesh = trimesh.load(gt_path, process=False)
            gt_v, gt_f = gt_mesh.vertices, gt_mesh.faces
            
        if SHOULD_EXPORT_GT:
            gt_obj_path = os.path.join(GT_EXPORT_DIR, f"{name}.obj")
            if not os.path.exists(gt_obj_path):
                trimesh.Trimesh(vertices=gt_v, faces=gt_f).export(gt_obj_path)

        gt_v_norm, center, scale = stage2_normalize(gt_v)
        
        # Get Base Mesh
        if args.mode == 'gt_simplified':
            base_path = sample.get('base_path')
            if base_path and os.path.exists(base_path):
                base_data = torch.load(base_path, map_location='cpu', weights_only=False)
                base_v = apply_normalization(base_data['base_verts'].numpy().astype(np.float32), center, scale)
                base_f = base_data['base_faces'].numpy().astype(np.int64)
                base_usage_info.append({"name": name, "base_path": base_path})
            else:
                import open3d as o3d
                o3d_m = o3d.geometry.TriangleMesh()
                o3d_m.vertices, o3d_m.triangles = o3d.utility.Vector3dVector(gt_v_norm), o3d.utility.Vector3iVector(gt_f.astype(np.int32))
                base_mesh_o3d = o3d_m.simplify_quadric_decimation(target_number_of_triangles=args.test_num_face)
                base_v, base_f = np.asarray(base_mesh_o3d.vertices, dtype=np.float32), np.asarray(base_mesh_o3d.triangles, dtype=np.int64)
                base_usage_info.append({"name": name, "base_path": "simplified_on_the_fly"})
        elif args.mode == 'given_simplified':
            base_path = sample.get('base_path', None)
            if not base_path or not os.path.exists(base_path):
                raise FileNotFoundError(
                    f"mode=given_simplified 需要有效的 base_path，但在样本 {name} 中未找到或文件不存在: {base_path}"
                )
            # 这里的 base mesh 是已经给定好的简化网格（OBJ 或其他几何格式）
            if base_path.endswith('.pt'):
                base_data = torch.load(base_path, map_location='cpu', weights_only=False)
                base_v = apply_normalization(base_data['base_verts'].numpy().astype(np.float32), center, scale)
                base_f = base_data['base_faces'].numpy().astype(np.int64)
            else:
                base_mesh = trimesh.load(base_path, process=False)
                base_v_raw, base_f_raw = base_mesh.vertices, base_mesh.faces
                base_v = apply_normalization(base_v_raw.astype(np.float32), center, scale)
                base_f = base_f_raw.astype(np.int64)
            base_usage_info.append({"name": name, "base_path": base_path, "mode": "given_simplified"})
        elif args.mode == 'edgerunner':
            os.makedirs(ER_CACHE_DIR, exist_ok=True)
            c_obj, c_pkl = os.path.join(ER_CACHE_DIR, f"{name}_base.obj"), os.path.join(ER_CACHE_DIR, f"{name}_emb.pkl")
            
            if args.use_cached and os.path.exists(c_obj):
                bm = trimesh.load(c_obj, process=False)
                base_v, base_f = apply_normalization(bm.vertices, center, scale), bm.faces
            else:
                gt_tm = trimesh.Trimesh(vertices=gt_v, faces=gt_f, process=False)
                points = gt_tm.sample(160000)
                pm, pM = points.min(0), points.max(0)
                pc, ps = (pM + pm) / 2, 2 * 0.95 / np.max(pM - pm)
                emb = er_wrapper.encode_points((points - pc) * ps)
                bm_er = er_wrapper.generate_mesh(emb, num_faces=args.test_num_face)
                with open(c_pkl, 'wb') as f: pickle.dump(emb.cpu().numpy(), f)
                world_v = bm_er.vertices / ps + pc
                trimesh.Trimesh(vertices=world_v, faces=bm_er.faces).export(c_obj)
                base_v, base_f = apply_normalization(world_v, center, scale), bm_er.faces
            base_usage_info.append({"name": name, "gt_path": gt_path, "base_path": c_obj, "emb_path": c_pkl})

        # Stage 2 Inference
        import open3d as o3d
        o3d_m = o3d.geometry.TriangleMesh()
        o3d_m.vertices, o3d_m.triangles = o3d.utility.Vector3dVector(base_v), o3d.utility.Vector3iVector(base_f.astype(np.int32))
        o3d_m.compute_vertex_normals()
        base_n = np.asarray(o3d_m.vertex_normals, dtype=np.float32)

        # 构建 GT mesh 用于采样点和法线
        gt_mesh_tm = trimesh.Trimesh(vertices=gt_v_norm, faces=gt_f, process=False)
        scan_points, face_ids = trimesh.sample.sample_surface(
            gt_mesh_tm, config['data']['point_num']
        )

        scan_normals_np = None
        if config['model'].get('use_scan_normal', False):
            # 使用对应三角形法线作为采样点的法线
            face_normals = gt_mesh_tm.face_normals  # (F, 3)
            scan_normals_np = face_normals[face_ids].astype(np.float32)
        
        # Prepare Tensors
        # Ensure float32 for vertex/normal/scan data, and long for faces
        b_v = torch.from_numpy(base_v).float().unsqueeze(0).to(device).contiguous()
        b_f = torch.from_numpy(base_f).long().unsqueeze(0).to(device).contiguous()
        b_n = torch.from_numpy(base_n).float().unsqueeze(0).to(device).contiguous()
        b_s = torch.from_numpy(scan_points).float().unsqueeze(0).to(device).contiguous()
        if scan_normals_np is not None:
            b_s_n = torch.from_numpy(scan_normals_np).float().unsqueeze(0).to(device).contiguous()
        else:
            b_s_n = None

        with torch.no_grad():
            if b_s_n is not None:
                fv, ff, _, _, _, _ = model(b_v, b_f, b_n, b_s, scan_normals=b_s_n)
            else:
                fv, ff, _, _, _, _ = model(b_v, b_f, b_n, b_s)
            
        trimesh.Trimesh(vertices=fv[0].cpu().numpy(), faces=ff.cpu().numpy(), process=False).export(os.path.join(fine_mesh_out_dir, f"{name}_fine.obj"))

        if args.render_views and args.render_views > 0:
            gv, gf = torch.from_numpy(gt_v_norm).unsqueeze(0).to(device), torch.from_numpy(gt_f).unsqueeze(0).to(device)
            pi, gi = renderer(fv, ff.unsqueeze(0), gv, gf)
            rd = os.path.join(final_output_dir, "renders", name)
            os.makedirs(rd, exist_ok=True)
            for i in range(min(4, pi.shape[0])):
                img = Image.fromarray(((pi[i].cpu().numpy() + 1.0) * 0.5 * 255).clip(0, 255).astype(np.uint8))
                img.save(os.path.join(rd, f"view_{i}.png"))

    with open(os.path.join(final_output_dir, "base_meshs.json"), 'w') as f: json.dump(base_usage_info, f, indent=4)
    if args.mode == 'edgerunner' and not args.use_cached:
        with open(os.path.join(ER_CACHE_DIR, "base_meshs.json"), 'w') as f: json.dump(base_usage_info, f, indent=4)
    print(f"Inference complete. Results in {final_output_dir}")

if __name__ == '__main__': main()
