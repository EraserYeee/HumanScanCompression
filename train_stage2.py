import argparse
import os
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
from data.stage2_dataset import ScanToMeshDataset, stage2_collate_fn
from utils.render import DifferentiableNormalRenderer

# 尝试导入 PyTorch3D Loss
try:
    from pytorch3d.loss import mesh_laplacian_smoothing
    from pytorch3d.structures import Meshes
except ImportError:
    print("PyTorch3D not found. Laplacian loss will be disabled.")
    def mesh_laplacian_smoothing(meshes, method="uniform"):
        return torch.tensor(0.0, device=meshes.device)

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
    save_dir = f"debug_train_meshes/step_{step:04d}_b{batch_idx}"
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
    save_dir = f"debug_train_images/step_{step:04d}_b{batch_idx}"
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
        mixed_precision=config['train'].get('mixed_precision', 'no'),
        log_with="wandb" if not args.no_wandb else None
    )
    set_seed(42)
    
    # Init WandB (on main process)
    run_name = f"{config['experiment_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if accelerator.is_main_process:
        # Create checkpoints dir
        ckpt_dir = os.path.join("checkpoints", run_name)
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
        
    dataset = ScanToMeshDataset(
        data_root=config['data']['processed_dir'], 
        split='train',
        base_faces_min=config['data']['base_mesh_faces_min'],
        base_faces_max=config['data']['base_mesh_faces_max'],
        backend=config['data'].get('simplification_backend', 'open3d')
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
        
    model = Stage2Pipeline(config={
        'feature_dim': config['model']['feature_dim'],
        'enc_hidden_dim': config['model']['enc_hidden_dim'],
        'dec_hidden_dim': config['model']['dec_hidden_dim'],
        'subdivision_levels': config['model']['subdivision_levels'],
        'subdivision_rate': config['model']['subdivision_rate']
    })
    # No need for .to(device), accelerate handles it
    
    # 4. Setup Renderer (Loss)
    # Renderer needs correct device for camera generation
    renderer = DifferentiableNormalRenderer(
        image_size=config['render']['image_size'], 
        device=accelerator.device, 
        cameras_per_batch=config['render']['views_per_sample'],
        dist_multiplier=config['render'].get('dist_multiplier', 1.0)
    )

    # 5. Optimizer
    optimizer = optim.Adam(
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
    
    # 6. Training Loop
    epochs = config['train']['epochs']
    if accelerator.is_main_process:
        print(f"Start training for {epochs} epochs...")
    
    model.train()
    
    step = 0
    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", disable=not accelerator.is_local_main_process)
        
        t_end = time.time()
        for batch in pbar:
            t_data_avail = time.time()

            # Unpack batch data (List of Tensors)
            # Accelerate handles the main batch dict move if it can, but for lists of tensors 
            # in a custom collate, we ensure they are on the right device.
            
            scan_points = batch['scan_points'].to(accelerator.device) # (B, P, 3)
            
            base_verts_list = [v.to(accelerator.device) for v in batch['base_verts']]
            base_faces_list = [f.to(accelerator.device) for f in batch['base_faces']]
            base_normals_list = [n.to(accelerator.device) for n in batch['base_normals']]
            
            gt_verts_list = [v.to(accelerator.device) for v in batch['gt_verts']]
            gt_faces_list = [f.to(accelerator.device) for f in batch['gt_faces']]
            
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t_to_device = time.time()

            optimizer.zero_grad()
            
            total_loss_batch = 0
            loss_render_batch = 0
            loss_lap_batch = 0
            loss_disp_batch = 0
            
            # Iterate over batch (Gradient Accumulation logic effectively)
            for b in range(len(base_verts_list)):
                # Skip if empty mesh
                if base_faces_list[b].shape[0] == 0:
                    continue
                    
                # Single item forward
                # Unsqueeze inputs to fake batch=1
                b_verts = base_verts_list[b].unsqueeze(0) # (1, V, 3)
                b_faces = base_faces_list[b].unsqueeze(0) # (1, F, 3)
                b_normals = base_normals_list[b].unsqueeze(0)
                b_scan = scan_points[b].unsqueeze(0) # (1, P, 3)
                
                # Forward
                f_verts, f_faces, disp = model(b_verts, b_faces, b_normals, b_scan)
                
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
                
                # Export Debug Mesh on error OR first step
                if (not is_valid or step == 0) and accelerator.is_main_process:
                    tag = "error" if not is_valid else "init"
                    debug_export_meshes(
                        b_verts[0], b_faces[0], 
                        f_verts[0], f_faces, 
                        g_verts[0], g_faces[0],
                        step, b, tag=tag
                    )
                    if not is_valid:
                        continue # Skip loss calculation for broken mesh

                # Loss for this item
                # 1. Render Loss
                # Render Pred vs GT
                # f_faces is (F_fine, 3), need (1, F_fine, 3)
                f_faces_expanded = f_faces.unsqueeze(0)
                
                pred_img, gt_img = renderer(f_verts, f_faces_expanded, g_verts, g_faces)
                
                # Debug Export Images (First step)
                if step == 0 and b == 0 and accelerator.is_main_process:
                    debug_export_images(pred_img, gt_img, step, b, tag="init")
                
                loss_render = torch.nn.functional.l1_loss(pred_img, gt_img)
                
                # 2. Laplacian
                f_mesh = Meshes(verts=f_verts, faces=f_faces_expanded)
                # loss_lap = mesh_laplacian_smoothing(f_mesh)
                loss_lap = torch.tensor(0.0, device=accelerator.device)
                
                # 3. Disp Reg
                # loss_disp = torch.mean(disp ** 2)
                loss_disp = torch.tensor(0.0, device=accelerator.device)
                
                # Weighted Sum
                loss = (
                    config['loss']['w_render'] * loss_render +
                    config['loss']['w_laplacian'] * loss_lap +
                    config['loss']['w_disp'] * loss_disp
                )
                
                # Accumulate (average later)
                loss = loss / len(base_verts_list)
                
                # Use accelerator for backward
                accelerator.backward(loss)
                
                total_loss_batch += loss.item()
                loss_render_batch += loss_render.item()
                loss_lap_batch += loss_lap.item()
                loss_disp_batch += loss_disp.item()
            
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t_forward_backward = time.time()

            optimizer.step()
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t_step_end = time.time()

            if accelerator.is_main_process:
                print(f"Step {step} | Data: {t_data_avail - t_end:.4f}s | Device: {t_to_device - t_data_avail:.4f}s | Fwd+Bwd: {t_forward_backward - t_to_device:.4f}s | Step: {t_step_end - t_forward_backward:.4f}s")

            step += 1
            
            # Log
            if step % config['train']['log_interval'] == 0:
                accelerator.log({
                    "loss/total": total_loss_batch,
                    "loss/render": loss_render_batch / len(base_verts_list),
                    "loss/laplacian": loss_lap_batch / len(base_verts_list),
                    "loss/disp": loss_disp_batch / len(base_verts_list),
                    "lr": optimizer.param_groups[0]['lr'],
                    "epoch": epoch
                }, step=step)
                
            pbar.set_postfix({
                'loss': f"{total_loss_batch:.4f}",
                'rend': f"{loss_render_batch / len(base_verts_list):.4f}"
            })

            t_end = time.time()
            
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
    args = parser.parse_args()
    
    config = load_config(args.config)
    train(config, args)
