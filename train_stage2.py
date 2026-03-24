import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
import yaml
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from datetime import datetime
import trimesh
import numpy as np
from PIL import Image
from accelerate import Accelerator
from accelerate.utils import set_seed

from models.pipeline import Stage2Pipeline
from models.tnet import feature_transform_regularizer # Import loss function
from data.stage2_dataset import ScanToMeshDataset, stage2_collate_fn
from utils.render import DifferentiableNormalRenderer

# 尝试导入 PyTorch3D Loss
try:
    from pytorch3d.loss import mesh_laplacian_smoothing, chamfer_distance
    from pytorch3d.structures import Meshes
except ImportError:
    print("PyTorch3D not found. Laplacian loss and Chamfer distance will be disabled.")
    def mesh_laplacian_smoothing(meshes, method="uniform"):
        return torch.tensor(0.0, device=meshes.device)
    def chamfer_distance(x, y, x_lengths=None, y_lengths=None, **kwargs):
        print("Chamfer distance not found")
        exit()
        return torch.tensor(0.0, device=x.device), torch.tensor(0.0, device=x.device)

# 尝试导入 SSIM (pytorch-msssim)
try:
    from pytorch_msssim import ssim
    HAS_SSIM = True
except ImportError:
    HAS_SSIM = False
    print("[Warn] pytorch-msssim not found. SSIM loss will be disabled.")
    def ssim(*args, **kwargs):
        return torch.tensor(0.0, device=args[0].device if args else 'cuda')

# 尝试导入 LPIPS
try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    print("[Warn] lpips not found. LPIPS loss will be disabled.")
    lpips = None


def compute_normal_loss_geo(pred_normal, gt_normal, mask):
    """
    Geometric normal loss from:
      L = (1 - N_hat·N) + ||N_hat - N||^2 + ||∇N_hat - ∇N||^2
    
    Args:
        pred_normal: (B, H, W, 3) predicted normal map
        gt_normal:   (B, H, W, 3) ground-truth normal map
        mask:        (B, H, W, 1) boolean mask for valid pixels
    Returns:
        scalar loss (mean over valid pixels)
    """
    mask_f = mask.float()
    num_valid = mask_f.sum().clamp(min=1.0)

    pred_n = torch.nn.functional.normalize(pred_normal, dim=-1, p=2)
    gt_n = torch.nn.functional.normalize(gt_normal, dim=-1, p=2)

    # Term 1: cosine distance
    cos_dist = (1.0 - (pred_n * gt_n).sum(dim=-1, keepdim=True)) * mask_f
    loss_cos = cos_dist.sum() / num_valid

    # Term 2: L2 distance
    l2_dist = ((pred_n - gt_n) ** 2).sum(dim=-1, keepdim=True) * mask_f
    loss_l2 = l2_dist.sum() / num_valid

    # Term 3: spatial gradient consistency (finite differences along H and W)
    # dx: difference along W; dy: difference along H
    pred_dx = pred_n[:, :, 1:, :] - pred_n[:, :, :-1, :]
    gt_dx = gt_n[:, :, 1:, :] - gt_n[:, :, :-1, :]
    pred_dy = pred_n[:, 1:, :, :] - pred_n[:, :-1, :, :]
    gt_dy = gt_n[:, 1:, :, :] - gt_n[:, :-1, :, :]

    mask_dx = mask_f[:, :, 1:, :] * mask_f[:, :, :-1, :]
    mask_dy = mask_f[:, 1:, :, :] * mask_f[:, :-1, :, :]

    grad_loss_x = (((pred_dx - gt_dx) ** 2).sum(dim=-1, keepdim=True) * mask_dx).sum() / mask_dx.sum().clamp(min=1.0)
    grad_loss_y = (((pred_dy - gt_dy) ** 2).sum(dim=-1, keepdim=True) * mask_dy).sum() / mask_dy.sum().clamp(min=1.0)
    loss_grad = grad_loss_x + grad_loss_y

    return loss_cos + loss_l2 + loss_grad


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config, save_path):
    with open(save_path, 'w') as f:
        yaml.dump(config, f)

def debug_export_meshes(base_v, base_f, fine_v, fine_f, gt_v, gt_f, step, batch_idx, tag=""):
    """
    导出训练过程中的 Mesh 用于调试
    """
    save_dir = f"/mnt/lab/data/yeruisi/data/compression/debug_train_meshes/step_{step:04d}_b{batch_idx}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Base Mesh
    if base_v is not None:
        trimesh.Trimesh(
            vertices=base_v.detach().cpu().numpy(), 
            faces=base_f.detach().cpu().numpy()
        ).export(os.path.join(save_dir, f"base_{tag}.obj"))
        
    # Fine Mesh (Pred)
    if fine_v is not None:
        trimesh.Trimesh(
            vertices=fine_v.detach().cpu().numpy(), 
            faces=fine_f.detach().cpu().numpy()
        ).export(os.path.join(save_dir, f"fine_{tag}.obj"))
        
    # GT Mesh
    if gt_v is not None:
        trimesh.Trimesh(
            vertices=gt_v.detach().cpu().numpy(), 
            faces=gt_f.detach().cpu().numpy()
        ).export(os.path.join(save_dir, f"gt_{tag}.obj"))
    
    print(f"[Debug] Exported meshes to {save_dir}")

def debug_export_images(pred_img, gt_img, step, batch_idx, tag=""):
    """
    导出渲染结果用于调试
    pred_img, gt_img: (K, H, W, 3) or (B*K, H, W, 3)
    """
    save_dir = f"/mnt/lab/data/yeruisi/data/compression/debug_train_images/step_{step:04d}_b{batch_idx}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Take first few views
    count = min(4, pred_img.shape[0])
    
    for i in range(count):
        # Pred
        p = pred_img[i].detach().cpu().numpy() # (H, W, 3)
        # Map [-1, 1] to [0, 255] if output is normal map-like
        # But our shader output is already normalized vector [-1, 1] or 0 for bg
        # Map to 0-255
        p = (p + 1.0) * 0.5 * 255.0
        p = np.clip(p, 0, 255).astype(np.uint8)
        Image.fromarray(p).save(os.path.join(save_dir, f"pred_view_{i}_{tag}.png"))
        
        # GT
        if gt_img is not None:
            g = gt_img[i].detach().cpu().numpy()
            g = (g + 1.0) * 0.5 * 255.0
            g = np.clip(g, 0, 255).astype(np.uint8)
            Image.fromarray(g).save(os.path.join(save_dir, f"gt_view_{i}_{tag}.png"))
            
    print(f"[Debug] Exported rendered images to {save_dir}")

def train(config, args):
    # 1. Setup Accelerator
    accelerator = Accelerator(
        mixed_precision=config['train'].get('mixed_precision', 'bf16'),
        log_with="wandb" if not args.no_wandb else None
    )
    set_seed(42)
    
    # Init WandB (on main process)
    run_name = f"{config['experiment_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if accelerator.is_main_process:
        # Create checkpoints dir
        ckpt_dir = os.path.join("/mnt/lab/data/yeruisi/data/compression/checkpoints", run_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        save_config(config, os.path.join(ckpt_dir, "config.yaml"))

    if not args.no_wandb:
        accelerator.init_trackers(
            project_name="mesh_compression_stage2", 
            config=config,
            init_kwargs={"wandb": {"name": run_name}}
        )

    # 2. Setup Data
    if accelerator.is_main_process:
        print(f"Loading dataset from {config['data']['processed_dir']}...")
        
    dataset_type = config['data'].get('dataset_type', 'human')
    
    dataset = ScanToMeshDataset(
        data_root=config['data']['processed_dir'], 
        split='train',
        point_num=config['data']['point_num'],
        base_faces_min=config['data']['base_mesh_faces_min'],
        base_faces_max=config['data']['base_mesh_faces_max'],
        backend=config['data'].get('simplification_backend', 'open3d'),
        preprocessed_base_mesh_dir=config['data'].get('preprocessed_base_mesh_dir', None),
        use_preprocess_base_mesh=config['data'].get('use_preprocess_base_mesh', False),
        preload_ram=config['data'].get('preload_ram', False), # Controlled by config
        lmdb_path=config['data'].get('lmdb_path', None),
        use_scan_normal=config['model'].get('use_scan_normal', False),
        dataset_type=dataset_type
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config['data']['batch_size'], 
        shuffle=True, 
        num_workers=config['data']['num_workers'],
        collate_fn=stage2_collate_fn,
        pin_memory=True
    )

    # 3. Setup Model
    if accelerator.is_main_process:
        print("Initializing model...")
        
    model = Stage2Pipeline(config=config['model'])
    # No need for .to(device), accelerate handles it
    
    # 4. Setup Renderer (Loss)
    # View chunking: split total views into smaller chunks to save VRAM
    total_views_per_sample = config['render']['views_per_sample']
    view_chunk_size = config['render'].get('view_chunk_size', total_views_per_sample)
    num_view_chunks = max(1, total_views_per_sample // view_chunk_size)
    if accelerator.is_main_process:
        print(f"Rendering: {total_views_per_sample} views/sample, "
              f"chunk_size={view_chunk_size}, num_chunks={num_view_chunks}")
    
    renderer = DifferentiableNormalRenderer(
        image_size=config['render']['image_size'], 
        device=accelerator.device, 
        cameras_per_batch=view_chunk_size,
        dist_multiplier=config['render'].get('dist_multiplier', 1.0)
    )
    
    # Setup LPIPS loss model if available
    lpips_model = None
    if HAS_LPIPS:
        lpips_model = lpips.LPIPS(net='vgg').to(accelerator.device)
        lpips_model.eval()
        for param in lpips_model.parameters():
            param.requires_grad = False
    else:
        assert("no lpips model found")

    # 5. Optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=float(config['train']['lr']),
        weight_decay=float(config['train']['weight_decay'])
    )
    
    # LR Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config['train']['epochs'], 
        eta_min=1e-6
    )
    
    # Prepare everything with Accelerator
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    
    # train.profile_timing: 耗时观测；profile_timing_interval: 每隔多少个 batch / forward 打印一次
    profile_timing = bool(config.get("train", {}).get("profile_timing", False))
    profile_interval = max(1, int(config.get("train", {}).get("profile_timing_interval", 50)))
    _unwrap = accelerator.unwrap_model(model)
    _unwrap._profile_timing = profile_timing
    _unwrap._profile_interval = profile_interval
    train_profile_batch_i = 0
    
    # Resume from checkpoint if specified
    start_epoch = 0
    resume_path = config.get('resume_path', None)
    resume_use_config_lr = bool(
        config.get('train', {}).get('resume_use_config_lr', False) or
        getattr(args, 'resume_use_config_lr', False)
    )
    if resume_path and os.path.exists(resume_path):
        if accelerator.is_main_process:
            print(f"Resuming training from {resume_path}...")
        
        checkpoint = torch.load(resume_path, map_location='cpu')
        
        # Load model weights
        # Unwrap if necessary, though accelerator usually handles loading state dict to wrapped model
        # But here we load manually. Accelerator.load_state is for its own format.
        # We used torch.save on unwrapped model, so we should load to unwrapped or handle prefix
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer and scheduler
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Optional: ignore resumed LR and force config LR after loading states
        if resume_use_config_lr:
            target_lr = float(config['train']['lr'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = target_lr
                if 'initial_lr' in param_group:
                    param_group['initial_lr'] = target_lr
            if hasattr(scheduler, 'base_lrs'):
                scheduler.base_lrs = [target_lr for _ in scheduler.base_lrs]
            if hasattr(scheduler, '_last_lr'):
                scheduler._last_lr = [target_lr for _ in scheduler._last_lr]
            if accelerator.is_main_process:
                print(f"Resume LR override enabled: using config lr = {target_lr}")
        
        start_epoch = checkpoint['epoch'] + 1
        if accelerator.is_main_process:
            print(f"Resumed from epoch {start_epoch}")
    
    # 6. Training Loop
    epochs = config['train']['epochs']
    if accelerator.is_main_process:
        print(f"Start training for {epochs} epochs...")
    
    model.train()
    
    step = start_epoch * len(dataloader)
    for epoch in range(start_epoch, epochs):
        oom_count = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", disable=not accelerator.is_local_main_process)
        
        t_end = time.time()
        for batch in pbar:
            try:
                t_data_avail = time.time()

                # Unpack batch data (List of Tensors)
                # Accelerate handles the main batch dict move if it can, but for lists of tensors 
                # in a custom collate, we ensure they are on the right device.
                
                scan_points = batch['scan_points'].to(accelerator.device) # (B, P, 3)
                scan_normals = batch['scan_normals'].to(accelerator.device) if 'scan_normals' in batch else None
                
                base_verts_list = [v.to(accelerator.device) for v in batch['base_verts']]
                base_faces_list = [f.to(accelerator.device) for f in batch['base_faces']]
                base_normals_list = [n.to(accelerator.device) for n in batch['base_normals']]
                
                gt_verts_list = [v.to(accelerator.device) for v in batch['gt_verts']]
                gt_faces_list = [f.to(accelerator.device) for f in batch['gt_faces']]
                t_tensors_on_device = time.time()
                if profile_timing:
                    train_profile_batch_i += 1

                optimizer.zero_grad(set_to_none=True)
                
                total_loss_batch = 0
                loss_render_batch = 0
                loss_chamfer_batch = 0
                loss_lap_batch = 0
                loss_disp_batch = 0
                loss_mat_batch = 0 # Matrix regularization loss
                loss_kl_batch = 0  # VAE KL divergence loss
                
                # Diagnostics: attentive (full) or cross_attention (CA residual stats)
                _et = config['model'].get('encoder_type', 'standard')
                enable_diagnostics = _et in ('attentive', 'cross_attention') and config.get(
                    'enable_attention_diagnostics', False
                )
                diagnostics_list = []  # Collect diagnostics from all batch items
                
                # Iterate over batch (Gradient Accumulation logic effectively)
                for b in range(len(base_verts_list)):
                    # Skip if empty mesh
                    if base_faces_list[b].shape[0] == 0:
                        continue
                    
                    # Single item forward
                    # Unsqueeze inputs to fake batch=1
                    # Ensure contiguous memory for .view() operations in model
                    b_verts = base_verts_list[b].unsqueeze(0).contiguous() # (1, V, 3)
                    b_faces = base_faces_list[b].unsqueeze(0).contiguous() # (1, F, 3)
                    b_normals = base_normals_list[b].unsqueeze(0).contiguous()
                    b_scan = scan_points[b].unsqueeze(0).contiguous() # (1, P, 3)
                    b_scan_normals = scan_normals[b].unsqueeze(0).contiguous() if scan_normals is not None else None
                    pred_img = None
                    gt_img = None
                    f_faces_expanded = None
                    loss = None
                
                    # Forward
                    if enable_diagnostics:
                        result = model(
                            b_verts, b_faces, b_normals, b_scan, scan_normals=b_scan_normals,
                            return_diagnostics=True
                        )
                        if len(result) == 7:
                            f_verts, f_faces, disp, trans_feat, vertex_features, model_kl_loss, diagnostics = result
                            diagnostics_list.append(diagnostics)
                        else:
                            # Fallback if diagnostics not available
                            f_verts, f_faces, disp, trans_feat, vertex_features, model_kl_loss = result[:6]
                    else:
                        f_verts, f_faces, disp, trans_feat, vertex_features, model_kl_loss = model(
                            b_verts, b_faces, b_normals, b_scan, scan_normals=b_scan_normals
                        )
                
                    # GT
                    g_verts = gt_verts_list[b].unsqueeze(0)
                    g_faces = gt_faces_list[b].unsqueeze(0)
                
                    # Check indices validity
                    is_valid = True
                    if f_faces.max() >= f_verts.shape[1]:
                        if accelerator.is_main_process:
                            print(f"[Error] Pred Faces max index {f_faces.max()} >= Pred Verts count {f_verts.shape[1]}")
                        is_valid = False
                    if g_faces.max() >= g_verts.shape[1]:
                        if accelerator.is_main_process:
                            print(f"[Error] GT Faces max index {g_faces.max()} >= GT Verts count {g_verts.shape[1]}")
                        is_valid = False
                
                    # Export Debug Mesh on error OR every 10 steps
                    should_export = (step % 3000 == 0 and b == 0) or (not is_valid)
                    # should_export = False
                    if should_export and accelerator.is_main_process:
                        tag = "error" if not is_valid else f"step_{step}"
                        debug_export_meshes(
                            b_verts[0], b_faces[0], 
                            f_verts[0], f_faces, 
                            g_verts[0], g_faces[0],
                            step, b, tag=tag
                        )
                        if not is_valid:
                            # Release tensors before skipping this sample
                            del f_verts, f_faces, disp, trans_feat, vertex_features
                            if model_kl_loss is not None:
                                del model_kl_loss
                            if enable_diagnostics and 'diagnostics' in locals():
                                # Remove diagnostics for this broken sample
                                if diagnostics_list and len(diagnostics_list) > 0:
                                    diagnostics_list.pop()
                            del b_verts, b_faces, b_normals, b_scan, g_verts, g_faces
                            if b_scan_normals is not None:
                                del b_scan_normals
                            continue # Skip loss calculation for broken mesh

                    # Loss for this item
                
                    # Check if using Chamfer Loss
                    loss_chamfer = torch.tensor(0.0, device=accelerator.device)
                
                    use_chamfer = config['loss'].get('use_chamfer', False)
                
                    if use_chamfer:
                        chamfer_result = chamfer_distance(f_verts, b_scan)
                        loss_chamfer = chamfer_result[0]
                        if len(chamfer_result) > 1:
                            del chamfer_result[1]
                        del chamfer_result
                
                    f_faces_expanded = f_faces.unsqueeze(0)
                
                    # Non-render losses (computed once, shared across view chunks)
                    loss_lap = torch.tensor(0.0, device=accelerator.device)
                    loss_disp = torch.tensor(0.0, device=accelerator.device)
                    loss_mat = torch.tensor(0.0, device=accelerator.device)
                    loss_kl = torch.tensor(0.0, device=accelerator.device)
                    if model_kl_loss is not None:
                        vae_beta = config['loss'].get('vae_beta', 0.001)
                        vae_warmup_epochs = config['loss'].get('vae_warmup_epochs', 50)
                        if epoch < vae_warmup_epochs:
                            beta = vae_beta * (epoch / max(vae_warmup_epochs, 1))
                        else:
                            beta = vae_beta
                        loss_kl = beta * model_kl_loss
                
                    w_mat = config['loss'].get('w_mat', 0.001)
                    if use_chamfer:
                        w_chamfer = config['loss']['w_chamfer']
                        w_render_weight = 0.0
                    else:
                        w_chamfer = 0.0
                        w_render_weight = config['loss']['w_render']
                
                    loss_non_render = (
                        w_chamfer * loss_chamfer +
                        config['loss']['w_laplacian'] * loss_lap +
                        config['loss']['w_disp'] * loss_disp +
                        w_mat * loss_mat +
                        loss_kl
                    )
                
                    # View-chunked rendering with per-chunk backward to save VRAM.
                    # Each chunk renders view_chunk_size views, computes render loss,
                    # and calls backward (with retain_graph for non-last chunks).
                    # This frees rendering intermediates between chunks while
                    # accumulating gradients on model parameters.
                    should_render = not use_chamfer or w_render_weight > 0
                    loss_render_accum = 0.0
                
                    if should_render and num_view_chunks > 0:
                        w_depth_l1 = config['loss'].get('w_depth_l1', 10.0)
                        normal_loss_type = config['loss'].get('normal_loss_type', 'l1')
                        w_normal_l1 = config['loss'].get('w_normal_l1', 4.0)
                        w_normal_ssim = config['loss'].get('w_normal_ssim', 0.5)
                        w_normal_lpips = config['loss'].get('w_normal_lpips', 0.5)
                        w_normal_geo = config['loss'].get('w_normal_geo', 4.0)
                    
                        for vc in range(num_view_chunks):
                            is_last_chunk = (vc == num_view_chunks - 1)
                        
                            pred_img, gt_img, pred_depth, gt_depth = renderer(
                                f_verts, f_faces_expanded, g_verts, g_faces
                            )
                        
                            loss_depth_l1 = torch.tensor(0.0, device=accelerator.device)
                        
                            if w_depth_l1 > 0 and pred_depth is not None and gt_depth is not None:
                                depth_mask = (pred_depth.abs() > 1e-6) & (gt_depth.abs() > 1e-6)
                                if depth_mask.any():
                                    loss_depth_l1 = torch.nn.functional.l1_loss(
                                        pred_depth[depth_mask], gt_depth[depth_mask]
                                    )
                        
                            if normal_loss_type == 'geo':
                                loss_normal_geo_val = torch.tensor(0.0, device=accelerator.device)
                                if w_normal_geo > 0 and pred_img is not None and gt_img is not None:
                                    normal_mask = (pred_img.abs().sum(dim=-1, keepdim=True) > 1e-6) | \
                                                 (gt_img.abs().sum(dim=-1, keepdim=True) > 1e-6)
                                    if normal_mask.any():
                                        loss_normal_geo_val = compute_normal_loss_geo(pred_img, gt_img, normal_mask)
                                chunk_render_loss = (
                                    w_depth_l1 * loss_depth_l1 +
                                    w_normal_geo * loss_normal_geo_val
                                )
                            else:
                                loss_normal_l1_val = torch.tensor(0.0, device=accelerator.device)
                                loss_normal_ssim = torch.tensor(0.0, device=accelerator.device)
                                loss_normal_lpips = torch.tensor(0.0, device=accelerator.device)

                                if w_normal_l1 > 0 and pred_img is not None and gt_img is not None:
                                    normal_mask = (pred_img.abs().sum(dim=-1, keepdim=True) > 1e-6) | \
                                                 (gt_img.abs().sum(dim=-1, keepdim=True) > 1e-6)
                                    if normal_mask.any():
                                        pred_masked = pred_img * normal_mask
                                        gt_masked = gt_img * normal_mask
                                        loss_normal_l1_val = torch.nn.functional.l1_loss(pred_masked, gt_masked)

                                if w_normal_ssim > 0 and HAS_SSIM and pred_img is not None and gt_img is not None:
                                    pred_normal_norm = (pred_img + 1.0) * 0.5
                                    gt_normal_norm = (gt_img + 1.0) * 0.5
                                    pred_normal_norm = pred_normal_norm.permute(0, 3, 1, 2)
                                    gt_normal_norm = gt_normal_norm.permute(0, 3, 1, 2)
                                    ssim_val = ssim(pred_normal_norm, gt_normal_norm, data_range=1.0)
                                    loss_normal_ssim = 1.0 - ssim_val

                                if w_normal_lpips > 0 and HAS_LPIPS and lpips_model is not None and pred_img is not None and gt_img is not None:
                                    pred_normal_lpips_in = pred_img.permute(0, 3, 1, 2)
                                    gt_normal_lpips_in = gt_img.permute(0, 3, 1, 2)
                                    loss_normal_lpips = lpips_model(pred_normal_lpips_in, gt_normal_lpips_in).mean()

                                chunk_render_loss = (
                                    w_depth_l1 * loss_depth_l1 +
                                    w_normal_l1 * loss_normal_l1_val +
                                    w_normal_ssim * loss_normal_ssim +
                                    w_normal_lpips * loss_normal_lpips
                                )
                        
                            loss_render_accum += chunk_render_loss.item()
                        
                            # Render loss for this chunk, averaged over total chunks
                            chunk_loss = w_render_weight * chunk_render_loss / num_view_chunks
                            if is_last_chunk:
                                chunk_loss = chunk_loss + loss_non_render
                            chunk_loss = chunk_loss / len(base_verts_list)
                        
                            accelerator.backward(chunk_loss, retain_graph=not is_last_chunk)
                        
                            if vc == 0 and should_export and accelerator.is_main_process:
                                debug_export_images(pred_img, gt_img, step, b, tag=f"step_{step}")
                        
                            del pred_img, gt_img, pred_depth, gt_depth
                            del chunk_render_loss, chunk_loss
                    else:
                        loss_nr_scaled = loss_non_render / len(base_verts_list)
                        accelerator.backward(loss_nr_scaled)
                        del loss_nr_scaled
                
                    loss_render_avg = loss_render_accum / max(num_view_chunks, 1)
                    total_loss_batch += (w_render_weight * loss_render_avg + loss_non_render.item()) / len(base_verts_list)
                    loss_render_batch += loss_render_avg
                    loss_chamfer_batch += loss_chamfer.item()
                    loss_lap_batch += loss_lap.item()
                    loss_disp_batch += loss_disp.item()
                    loss_mat_batch += loss_mat.item()
                    loss_kl_batch += loss_kl.item()
                
                    # Delete large tensors to free memory immediately
                    del f_verts, f_faces, disp, trans_feat, vertex_features
                    if model_kl_loss is not None:
                        del model_kl_loss
                    del b_verts, b_faces, b_normals, b_scan
                    del g_verts, g_faces
                    if b_scan_normals is not None:
                        del b_scan_normals
                    if f_faces_expanded is not None:
                        del f_faces_expanded
                    del loss_non_render, loss_chamfer, loss_lap, loss_disp, loss_mat, loss_kl
                
                optimizer.step()

                if accelerator.is_main_process and profile_timing and (train_profile_batch_i % profile_interval == 0):
                    gap_ms = (t_data_avail - t_end) * 1000
                    h2d_ms = (t_tensors_on_device - t_data_avail) * 1000
                    print(f"[timing/train] batch={train_profile_batch_i} dataloader_gap={gap_ms:.2f}ms to_device={h2d_ms:.2f}ms")

                # Collect gradient information for diagnostics (after optimizer.step())
                if enable_diagnostics and len(diagnostics_list) > 0:
                    unwrapped_model = accelerator.unwrap_model(model)
                    if hasattr(unwrapped_model, 'encoder') and hasattr(unwrapped_model.encoder, 'score_linear'):
                        score_linear = unwrapped_model.encoder.score_linear
                        if score_linear.weight.grad is not None:
                            grad_norm = score_linear.weight.grad.norm().item()
                            weight_norm = score_linear.weight.norm().item()
                            # Update diagnostics with gradient info
                            for diag in diagnostics_list:
                                diag['score_linear_grad_norm'] = grad_norm
                                diag['score_linear_weight_norm'] = weight_norm

                # Save batch_len before releasing batch tensors (needed for logging)
                batch_len = len(base_verts_list)
                
                step += 1
                
                # Log
                if step % config['train']['log_interval'] == 0:
                    log_dict = {
                        "loss/total": total_loss_batch,
                        "loss/render": loss_render_batch / batch_len,
                        "loss/chamfer": loss_chamfer_batch / batch_len,
                        "loss/laplacian": loss_lap_batch / batch_len,
                        "loss/disp": loss_disp_batch / batch_len,
                        "loss/mat": loss_mat_batch / batch_len,
                        "loss/kl": loss_kl_batch / batch_len,
                        "lr": optimizer.param_groups[0]['lr'],
                        "epoch": epoch
                    }
                    
                    # Add diagnostics if available
                    if enable_diagnostics and len(diagnostics_list) > 0:
                        # Average diagnostics across batch items
                        avg_diag = {}
                        for key in diagnostics_list[0].keys():
                            if isinstance(diagnostics_list[0][key], dict):
                                # Nested dict (e.g., 'point_feats', 'scores', etc.)
                                avg_diag[key] = {}
                                for subkey in diagnostics_list[0][key].keys():
                                    if subkey != 'score_linear_grad_norm':  # This is set after backward
                                        values = [d[key][subkey] for d in diagnostics_list if key in d and subkey in d[key]]
                                        if values:
                                            avg_diag[key][subkey] = sum(values) / len(values)
                            elif key.startswith('head_'):
                                # Per-head statistics
                                avg_diag[key] = {}
                                for subkey in diagnostics_list[0][key].keys():
                                    values = [d[key][subkey] for d in diagnostics_list if key in d and subkey in d[key]]
                                    if values:
                                        avg_diag[key][subkey] = sum(values) / len(values)
                            elif key in ['num_points', 'num_clusters']:
                                # Use first value (should be same across batch)
                                avg_diag[key] = diagnostics_list[0][key]
                            elif isinstance(diagnostics_list[0][key], (int, float)):
                                values = [d[key] for d in diagnostics_list if key in d]
                                if values:
                                    avg_diag[key] = sum(values) / len(values)
                        
                        # Get gradient norm from first item (should be same after optimizer.step())
                        if diagnostics_list[0].get('score_linear_grad_norm') is not None:
                            avg_diag['score_linear_grad_norm'] = diagnostics_list[0]['score_linear_grad_norm']
                        if diagnostics_list[0].get('score_linear_weight_norm') is not None:
                            avg_diag['score_linear_weight_norm'] = diagnostics_list[0]['score_linear_weight_norm']
                        
                        # Flatten nested dicts for wandb logging
                        for key, value in avg_diag.items():
                            if isinstance(value, dict):
                                for subkey, subvalue in value.items():
                                    log_dict[f"diagnostics/{key}/{subkey}"] = subvalue
                            else:
                                log_dict[f"diagnostics/{key}"] = value
                        
                        # Print diagnostics summary
                        if accelerator.is_main_process:
                            print(f"\n[Diagnostics] Step {step}:")
                            if 'point_feats' in avg_diag:
                                pf = avg_diag['point_feats']
                                print(f"  Point Features: mean={pf.get('mean', 0):.6f}, std={pf.get('std', 0):.6f}, "
                                      f"min={pf.get('min', 0):.6f}, max={pf.get('max', 0):.6f}")
                            if 'scores' in avg_diag:
                                sc = avg_diag['scores']
                                print(f"  Scores (logits, raw): mean={sc.get('mean', 0):.6f}, std={sc.get('std', 0):.6f}, "
                                      f"min={sc.get('min', 0):.6f}, max={sc.get('max', 0):.6f}")
                            if 'scores_scaled' in avg_diag:
                                scs = avg_diag['scores_scaled']
                                print(f"  Scores (logits, scaled): mean={scs.get('mean', 0):.6f}, std={scs.get('std', 0):.6f}, "
                                      f"min={scs.get('min', 0):.6f}, max={scs.get('max', 0):.6f}")
                            if 'attention_scores' in avg_diag:
                                att = avg_diag['attention_scores']
                                print(f"  Attention (alpha): mean={att.get('mean', 0):.6f}, std={att.get('std', 0):.6f}, "
                                      f"min={att.get('min', 0):.6f}, max={att.get('max', 0):.6f}")
                            for h in range(4):  # Assuming 4 heads
                                head_key = f'head_{h}'
                                if head_key in avg_diag:
                                    hd = avg_diag[head_key]
                                    print(f"  Head {h}: mean={hd.get('mean', 0):.6f}, std={hd.get('std', 0):.6f}, "
                                          f"min={hd.get('min', 0):.6f}, max={hd.get('max', 0):.6f}")
                            if 'score_linear_grad_norm' in avg_diag:
                                print(f"  Score Linear Grad Norm: {avg_diag['score_linear_grad_norm']:.8f}")
                            if 'score_linear_weight_norm' in avg_diag:
                                print(f"  Score Linear Weight Norm: {avg_diag['score_linear_weight_norm']:.8f}")
                            if 'ca_residual_mean_abs' in avg_diag:
                                print(f"  CA residual |mean|: {avg_diag.get('ca_residual_mean_abs', 0):.6f}, "
                                      f"coarse |mean|: {avg_diag.get('ca_coarse_mean_abs', 0):.6f}, "
                                      f"ratio: {avg_diag.get('ca_residual_to_coarse_ratio', 0):.6f}")
                            print()
                    
                    accelerator.log(log_dict, step=step)
                    
                    # Clear diagnostics list after logging
                    diagnostics_list.clear()
                    
                pbar.set_postfix({
                    'loss': f"{total_loss_batch:.4f}",
                    'cham': f"{loss_chamfer_batch / batch_len:.4f}",
                    'rend': f"{loss_render_batch / batch_len:.4f}",
                    'mat': f"{loss_mat_batch / batch_len:.4f}",
                    'kl': f"{loss_kl_batch / batch_len:.6f}"
                })
                
                # Release batch-level tensors after logging
                del scan_points
                if scan_normals is not None:
                    del scan_normals
                del base_verts_list, base_faces_list, base_normals_list, gt_verts_list, gt_faces_list
                if torch.cuda.is_available() and (step % 50 == 0):
                    torch.cuda.empty_cache()

                t_end = time.time()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    oom_count += 1
                    if accelerator.is_main_process:
                        print(f"[Warn] CUDA OOM at epoch {epoch}, step {step}. Skipping this batch. OOM count in this epoch: {oom_count}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if oom_count >= 3:
                        raise RuntimeError(f"CUDA OOM occurred {oom_count} times in epoch {epoch}. Exiting training.")
                    continue
                else:
                    raise
            
        # Update Scheduler
        scheduler.step()
            
        # Save Checkpoint
        if (epoch + 1) % config['train']['save_interval'] == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': unwrapped_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'config': config
                }, os.path.join(ckpt_dir, f"epoch_{epoch+1}.pth"))
            
    accelerator.end_training()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml', help="Path to config file")
    parser.add_argument('--no_wandb', action='store_true', help="Disable wandb logging")
    parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument('--resume_use_config_lr', action='store_true',
                        help="When resuming, ignore checkpoint LR and use config['train']['lr']")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # If resuming, update config with resume path if not present (or handle logic in train)
    if args.resume:
        config['resume_path'] = args.resume
        
    train(config, args)
