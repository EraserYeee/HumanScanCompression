"""Joint Stage 1 + Stage 2 training script.

Phase 1: Pre-train Stage 1 with geometric losses (no mesh extraction, no Stage 2)
Phase 2: Pre-train Stage 2 with Stage 1 output (Stage 1 frozen)
Phase 3: Joint fine-tuning of both stages

Usage:
    python train_joint.py --config configs/config_joint.yaml --phase 1
    python train_joint.py --config configs/config_joint.yaml --phase 2 --stage1-ckpt checkpoints/stage1_phase1.pth
    python train_joint.py --config configs/config_joint.yaml --phase 3 --stage1-ckpt ... --stage2-ckpt ...
"""

import argparse
import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from datetime import datetime
from accelerate import Accelerator
from accelerate.utils import set_seed

from models.stage1_pipeline import Stage1Pipeline
from models.pipeline import Stage2Pipeline
from models.tnet import feature_transform_regularizer
from data.stage2_dataset import ScanToMeshDataset, stage2_collate_fn
from utils.render import DifferentiableNormalRenderer
from utils.train_helpers import (
    load_config,
    save_config,
    debug_export_meshes,
    compute_uniform_laplacian_l1,
    compute_normal_loss_geo,
)
from utils.stage1_losses import (
    curvature_weighted_chamfer,
    normal_consistency_loss,
    mesh_regularity_loss,
)

try:
    from pytorch3d.loss import chamfer_distance
    from pytorch3d.structures import Meshes
except ImportError:
    pass

import trimesh
import numpy as np


def _export_stage1_mesh(stage1_model, scan_points, scan_normals, scan_curvature,
                        export_dir, epoch, tag="train", device='cuda'):
    """Export a Stage 1 base mesh + seed points for visual inspection.

    Exports:
      - {tag}_epoch{epoch}_base_mesh.obj  (extracted mesh)
      - {tag}_epoch{epoch}_seeds.ply      (seed points, colored by curvature)
      - {tag}_epoch{epoch}_scan.ply       (subsampled scan for reference)
    """
    os.makedirs(export_dir, exist_ok=True)
    stage1_model.eval()
    with torch.no_grad():
        base_verts, base_faces, base_normals, aux = stage1_model(
            scan_points, scan_normals
        )
    stage1_model.train()

    prefix = os.path.join(export_dir, f"{tag}_epoch{epoch:04d}")

    # Export base mesh
    v = base_verts[0].cpu().numpy()
    f = base_faces[0].cpu().numpy()
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    mesh.export(f"{prefix}_base_mesh.obj")

    # Export seeds colored by nearest-GT curvature
    seeds_np = aux['seed_positions'][0].cpu().numpy()
    if scan_curvature is not None:
        from pytorch3d.ops import knn_points
        with torch.no_grad():
            knn = knn_points(
                aux['seed_positions'].to(device),
                scan_points.to(device), K=1,
            )
        nn_idx = knn.idx[0, :, 0].cpu().numpy()
        cur = scan_curvature[0].cpu().numpy()[nn_idx].flatten()
        # Red=high curvature, blue=low
        colors = np.zeros((len(seeds_np), 4), dtype=np.uint8)
        colors[:, 0] = (cur * 255).clip(0, 255).astype(np.uint8)
        colors[:, 2] = ((1 - cur) * 255).clip(0, 255).astype(np.uint8)
        colors[:, 3] = 255
        pc = trimesh.PointCloud(seeds_np, colors=colors)
    else:
        pc = trimesh.PointCloud(seeds_np)
    pc.export(f"{prefix}_seeds.ply")

    # Export subsampled scan for reference
    sub_idx = np.random.choice(scan_points.shape[1], min(50000, scan_points.shape[1]), replace=False)
    scan_sub = scan_points[0, sub_idx].cpu().numpy()
    trimesh.PointCloud(scan_sub).export(f"{prefix}_scan_ref.ply")

    return f"{prefix}_base_mesh.obj"

try:
    from pytorch_msssim import ssim
    HAS_SSIM = True
except ImportError:
    HAS_SSIM = False
    def ssim(*args, **kwargs):
        return torch.tensor(0.0, device=args[0].device if args else 'cuda')


def compute_stage1_loss(seed_positions, seed_normals, scan_points, scan_normals,
                        scan_curvature, config_s1, extracted_faces=None,
                        extracted_verts=None):
    """Compute Stage 1 losses on seed points (and optionally extracted mesh).

    Returns:
        loss: scalar
        loss_dict: dict for logging
    """
    loss_dict = {}

    # Curvature-weighted Chamfer
    loss_chamfer, chamfer_parts = curvature_weighted_chamfer(
        scan_points[0], scan_curvature[0],
        seed_positions[0],
        w_curv=config_s1.get('w_chamfer_curv', 3e3),
        w_uniform=config_s1.get('w_chamfer_uniform', 1e2),
        w_backward=config_s1.get('w_chamfer_backward', 1e3),
        w_repulsion=config_s1.get('w_chamfer_repulsion', 1e2),
        repulsion_clamp=config_s1.get('repulsion_clamp', 3.0),
    )
    loss_dict.update({f's1/{k}': v for k, v in chamfer_parts.items()})
    loss_dict['s1/chamfer_total'] = loss_chamfer.item()

    # Normal consistency
    w_normal = config_s1.get('w_normal', 100.0)
    loss_normal = normal_consistency_loss(
        seed_positions[0], seed_normals[0],
        scan_points[0], scan_normals[0]
    )
    loss_dict['s1/normal_consistency'] = loss_normal.item()

    loss = loss_chamfer + w_normal * loss_normal

    # Mesh regularity (only when mesh is extracted)
    if extracted_faces is not None and extracted_verts is not None:
        w_mesh = config_s1.get('w_mesh_reg', 1.0)
        loss_mesh = mesh_regularity_loss(extracted_verts, extracted_faces)
        loss += w_mesh * loss_mesh
        loss_dict['s1/mesh_regularity'] = loss_mesh.item()

    loss_dict['s1/total'] = loss.item()
    return loss, loss_dict


def train_phase1(config, args):
    """Phase 1: Pre-train Stage 1 with geometric losses only."""
    accelerator = Accelerator(
        mixed_precision='bf16',
        log_with="wandb" if not args.no_wandb else None,
    )
    set_seed(config.get('seed', 42))

    run_name = f"{config['experiment_name']}_phase1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if accelerator.is_main_process:
        ckpt_dir = os.path.join("/mnt/lab/data/yeruisi/data/compression/checkpoints", run_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        save_config(config, os.path.join(ckpt_dir, "config.yaml"))

    if not args.no_wandb:
        accelerator.init_trackers(
            project_name="mesh_compression_joint",
            config=config,
            init_kwargs={"wandb": {"name": run_name}},
        )

    # Dataset (with curvature, no base mesh needed)
    dataset = ScanToMeshDataset(
        data_root=config['data']['processed_dir'],
        split='train',
        point_num=config['data']['point_num'],
        base_faces_min=config['data'].get('base_mesh_faces_min', 1000),
        base_faces_max=config['data'].get('base_mesh_faces_max', 3000),
        backend=config['data'].get('simplification_backend', 'pyfqmr'),
        use_preprocess_base_mesh=config['data'].get('use_preprocess_base_mesh', False),
        preprocessed_base_mesh_dir=config['data'].get('preprocessed_base_mesh_dir', None),
        lmdb_path=config['data'].get('lmdb_path', None),
        use_scan_normal=True,
        compute_curvature=True,
        curvature_knn=config['data'].get('curvature_knn', 30),
        curvature_subsample=config['data'].get('curvature_subsample', 50000),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=config['data'].get('num_workers', 2),
        collate_fn=stage2_collate_fn,
        pin_memory=True,
        persistent_workers=config['data'].get('num_workers', 2) > 0,
    )

    # Stage 1 model
    stage1 = Stage1Pipeline(config['stage1'])

    optimizer = optim.AdamW(
        stage1.parameters(),
        lr=float(config['train']['phase1_lr']),
        weight_decay=float(config['train']['phase1_weight_decay']),
    )

    epochs = config['train']['phase1_epochs']
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    stage1, optimizer, dataloader, scheduler = accelerator.prepare(
        stage1, optimizer, dataloader, scheduler
    )

    # Resume
    start_epoch = 0
    if args.stage1_ckpt and os.path.exists(args.stage1_ckpt):
        ckpt = torch.load(args.stage1_ckpt, map_location='cpu')
        accelerator.unwrap_model(stage1).load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt.get('epoch', -1) + 1
        if accelerator.is_main_process:
            print(f"Resumed Phase 1 from epoch {start_epoch}")

    if accelerator.is_main_process:
        print(f"Phase 1: Pre-training Stage 1 for {epochs} epochs (start={start_epoch})")
        print(f"  Seeds: [{config['stage1']['num_seeds_min']}, {config['stage1']['num_seeds_max']}]")

    step = start_epoch * len(dataloader)
    for epoch in range(start_epoch, epochs):
        pbar = tqdm(dataloader, desc=f"P1 Epoch {epoch+1}/{epochs}",
                    disable=not accelerator.is_local_main_process)

        for batch in pbar:
            try:
                scan_points = batch['scan_points'].to(accelerator.device)
                scan_normals = batch['scan_normals'].to(accelerator.device)
                scan_curvature = batch['scan_curvature'].to(accelerator.device)

                optimizer.zero_grad(set_to_none=True)

                # Forward (seeds only, no mesh extraction)
                seed_positions, seed_normals, _ = accelerator.unwrap_model(stage1).forward_seeds_only(
                    scan_points, scan_normals
                )

                # Loss
                loss, loss_dict = compute_stage1_loss(
                    seed_positions, seed_normals,
                    scan_points, scan_normals, scan_curvature,
                    config['stage1'],
                )

                accelerator.backward(loss)
                torch.nn.utils.clip_grad_norm_(stage1.parameters(), max_norm=1.0)
                optimizer.step()

                if accelerator.is_main_process and step % config['train']['log_interval'] == 0:
                    log_data = {**loss_dict, 's1/lr': optimizer.param_groups[0]['lr']}
                    if not args.no_wandb:
                        accelerator.log(log_data, step=step)
                    pbar.set_postfix(loss=f"{loss.item():.4f}")

                step += 1

            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    if accelerator.is_main_process:
                        print(f"[OOM] Phase 1 step {step}, skipping")
                    continue
                raise

        scheduler.step()

        # Per-epoch: export a random sample's base mesh for visual inspection
        if accelerator.is_main_process:
            try:
                vis_batch = next(iter(dataloader))
                vis_scan = vis_batch['scan_points'].to(accelerator.device)
                vis_norm = vis_batch['scan_normals'].to(accelerator.device)
                vis_cur = vis_batch.get('scan_curvature')
                if vis_cur is not None:
                    vis_cur = vis_cur.to(accelerator.device)
                export_dir = os.path.join(ckpt_dir, "vis")
                path = _export_stage1_mesh(
                    accelerator.unwrap_model(stage1),
                    vis_scan, vis_norm, vis_cur,
                    export_dir, epoch + 1, device=accelerator.device,
                )
                print(f"Exported vis: {path} (V={0}, F={0})")
            except Exception as e:
                print(f"[Vis export failed] {e}")

        # Save checkpoint
        if (epoch + 1) % config['train']['save_interval'] == 0 and accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(stage1)
            save_path = os.path.join(ckpt_dir, f"stage1_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': unwrapped.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, save_path)
            print(f"Saved: {save_path}")

    if accelerator.is_main_process:
        final_path = os.path.join(ckpt_dir, "stage1_final.pth")
        torch.save({
            'epoch': epochs - 1,
            'model_state_dict': accelerator.unwrap_model(stage1).state_dict(),
        }, final_path)
        print(f"Phase 1 complete. Final checkpoint: {final_path}")

    accelerator.end_training()


def train_phase2(config, args):
    """Phase 2: Pre-train Stage 2 with Stage 1 base meshes (Stage 1 frozen)."""
    accelerator = Accelerator(
        mixed_precision='bf16',
        log_with="wandb" if not args.no_wandb else None,
    )
    set_seed(config.get('seed', 42))

    run_name = f"{config['experiment_name']}_phase2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if accelerator.is_main_process:
        ckpt_dir = os.path.join("/mnt/lab/data/yeruisi/data/compression/checkpoints", run_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        save_config(config, os.path.join(ckpt_dir, "config.yaml"))

    if not args.no_wandb:
        accelerator.init_trackers(
            project_name="mesh_compression_joint",
            config=config,
            init_kwargs={"wandb": {"name": run_name}},
        )

    # Dataset
    dataset = ScanToMeshDataset(
        data_root=config['data']['processed_dir'],
        split='train',
        point_num=config['data']['point_num'],
        base_faces_min=config['data'].get('base_mesh_faces_min', 1000),
        base_faces_max=config['data'].get('base_mesh_faces_max', 3000),
        backend=config['data'].get('simplification_backend', 'pyfqmr'),
        use_preprocess_base_mesh=config['data'].get('use_preprocess_base_mesh', False),
        preprocessed_base_mesh_dir=config['data'].get('preprocessed_base_mesh_dir', None),
        lmdb_path=config['data'].get('lmdb_path', None),
        use_scan_normal=config['stage2'].get('use_scan_normal', True),
    )

    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True,
        num_workers=config['data'].get('num_workers', 2),
        collate_fn=stage2_collate_fn, pin_memory=True,
        persistent_workers=config['data'].get('num_workers', 2) > 0,
    )

    # Load Stage 1 (frozen)
    stage1 = Stage1Pipeline(config['stage1'])
    assert args.stage1_ckpt, "Phase 2 requires --stage1-ckpt"
    ckpt = torch.load(args.stage1_ckpt, map_location='cpu')
    stage1.load_state_dict(ckpt['model_state_dict'])
    stage1.eval()
    for p in stage1.parameters():
        p.requires_grad_(False)
    stage1 = stage1.to(accelerator.device)

    # Stage 2 model
    stage2 = Stage2Pipeline(config=config['stage2'])

    # Renderer
    total_views = config['render']['views_per_sample']
    view_chunk = config['render'].get('view_chunk_size', total_views)
    num_view_chunks = max(1, total_views // view_chunk)
    renderer = DifferentiableNormalRenderer(
        image_size=config['render']['image_size'],
        device=accelerator.device,
        cameras_per_batch=view_chunk,
        dist_multiplier=config['render'].get('dist_multiplier', 1.0),
    )

    optimizer = optim.AdamW(
        stage2.parameters(),
        lr=float(config['train']['phase2_lr']),
        weight_decay=float(config['train']['phase2_weight_decay']),
    )

    epochs = config['train']['phase2_epochs']
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    stage2, optimizer, dataloader, scheduler = accelerator.prepare(
        stage2, optimizer, dataloader, scheduler
    )

    if accelerator.is_main_process:
        print(f"Phase 2: Pre-training Stage 2 for {epochs} epochs (Stage 1 frozen)")

    step = 0
    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"P2 Epoch {epoch+1}/{epochs}",
                    disable=not accelerator.is_local_main_process)

        for batch in pbar:
            try:
                scan_points = batch['scan_points'].to(accelerator.device)
                scan_normals = batch.get('scan_normals')
                if scan_normals is not None:
                    scan_normals = scan_normals.to(accelerator.device)

                gt_verts_list = [v.to(accelerator.device) for v in batch['gt_verts']]
                gt_faces_list = [f.to(accelerator.device) for f in batch['gt_faces']]

                optimizer.zero_grad(set_to_none=True)

                # Generate base mesh with frozen Stage 1
                with torch.no_grad():
                    base_verts, base_faces, base_normals, _ = stage1(
                        scan_points, scan_normals
                    )

                if base_faces[0].shape[0] == 0:
                    continue

                # Stage 2 forward
                f_verts, f_faces, disp, trans_feat, vertex_features, kl_loss = \
                    accelerator.unwrap_model(stage2)(
                        base_verts, base_faces, base_normals,
                        scan_points, scan_normals=scan_normals,
                    )

                # Rendering loss (same as train_stage2.py)
                g_verts = gt_verts_list[0].unsqueeze(0)
                g_faces = gt_faces_list[0].unsqueeze(0)

                if f_faces.max() >= f_verts.shape[1] or g_faces.max() >= g_verts.shape[1]:
                    if accelerator.is_main_process:
                        print("[Error] Invalid face indices, skipping")
                    continue

                f_faces_expanded = f_faces.unsqueeze(0)
                loss = _compute_stage2_rendering_loss(
                    f_verts, f_faces, f_faces_expanded,
                    g_verts, g_faces,
                    trans_feat, kl_loss, disp,
                    renderer, num_view_chunks, view_chunk,
                    config, epoch, accelerator,
                )

                if loss is not None:
                    accelerator.backward(loss)
                    torch.nn.utils.clip_grad_norm_(stage2.parameters(), max_norm=1.0)
                    optimizer.step()

                    if accelerator.is_main_process and step % config['train']['log_interval'] == 0:
                        log_data = {'s2/total_loss': loss.item(), 's2/lr': optimizer.param_groups[0]['lr']}
                        if not args.no_wandb:
                            accelerator.log(log_data, step=step)
                        pbar.set_postfix(loss=f"{loss.item():.4f}")

                step += 1

            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    if accelerator.is_main_process:
                        print(f"[OOM] Phase 2 step {step}, skipping")
                    continue
                raise

        scheduler.step()

        if (epoch + 1) % config['train']['save_interval'] == 0 and accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(stage2)
            save_path = os.path.join(ckpt_dir, f"stage2_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': unwrapped.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, save_path)
            print(f"Saved: {save_path}")

    if accelerator.is_main_process:
        final_path = os.path.join(ckpt_dir, "stage2_final.pth")
        torch.save({
            'epoch': epochs - 1,
            'model_state_dict': accelerator.unwrap_model(stage2).state_dict(),
        }, final_path)
        print(f"Phase 2 complete. Final checkpoint: {final_path}")

    accelerator.end_training()


def train_phase3(config, args):
    """Phase 3: Joint fine-tuning of Stage 1 + Stage 2."""
    accelerator = Accelerator(
        mixed_precision='bf16',
        log_with="wandb" if not args.no_wandb else None,
    )
    set_seed(config.get('seed', 42))

    run_name = f"{config['experiment_name']}_phase3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if accelerator.is_main_process:
        ckpt_dir = os.path.join("/mnt/lab/data/yeruisi/data/compression/checkpoints", run_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        save_config(config, os.path.join(ckpt_dir, "config.yaml"))

    if not args.no_wandb:
        accelerator.init_trackers(
            project_name="mesh_compression_joint",
            config=config,
            init_kwargs={"wandb": {"name": run_name}},
        )

    # Dataset (with curvature for Stage 1 loss)
    dataset = ScanToMeshDataset(
        data_root=config['data']['processed_dir'],
        split='train',
        point_num=config['data']['point_num'],
        base_faces_min=config['data'].get('base_mesh_faces_min', 1000),
        base_faces_max=config['data'].get('base_mesh_faces_max', 3000),
        backend=config['data'].get('simplification_backend', 'pyfqmr'),
        use_preprocess_base_mesh=config['data'].get('use_preprocess_base_mesh', False),
        preprocessed_base_mesh_dir=config['data'].get('preprocessed_base_mesh_dir', None),
        lmdb_path=config['data'].get('lmdb_path', None),
        use_scan_normal=config['stage2'].get('use_scan_normal', True),
        compute_curvature=True,
        curvature_knn=config['data'].get('curvature_knn', 30),
        curvature_subsample=config['data'].get('curvature_subsample', 50000),
    )

    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True,
        num_workers=config['data'].get('num_workers', 2),
        collate_fn=stage2_collate_fn, pin_memory=True,
        persistent_workers=config['data'].get('num_workers', 2) > 0,
    )

    # Load both stages
    stage1 = Stage1Pipeline(config['stage1'])
    if args.stage1_ckpt:
        ckpt = torch.load(args.stage1_ckpt, map_location='cpu')
        stage1.load_state_dict(ckpt['model_state_dict'])
        if accelerator.is_main_process:
            print(f"Loaded Stage 1 from {args.stage1_ckpt}")

    stage2 = Stage2Pipeline(config=config['stage2'])
    if args.stage2_ckpt:
        ckpt = torch.load(args.stage2_ckpt, map_location='cpu')
        stage2.load_state_dict(ckpt['model_state_dict'])
        if accelerator.is_main_process:
            print(f"Loaded Stage 2 from {args.stage2_ckpt}")

    # Renderer
    total_views = config['render']['views_per_sample']
    view_chunk = config['render'].get('view_chunk_size', total_views)
    num_view_chunks = max(1, total_views // view_chunk)
    renderer = DifferentiableNormalRenderer(
        image_size=config['render']['image_size'],
        device=accelerator.device,
        cameras_per_batch=view_chunk,
        dist_multiplier=config['render'].get('dist_multiplier', 1.0),
    )

    # Dual-LR optimizer
    optimizer = optim.AdamW([
        {'params': stage1.parameters(), 'lr': float(config['train']['phase3_lr_stage1'])},
        {'params': stage2.parameters(), 'lr': float(config['train']['phase3_lr_stage2'])},
    ], weight_decay=float(config['train']['phase3_weight_decay']))

    epochs = config['train']['phase3_epochs']
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    stage1, stage2, optimizer, dataloader, scheduler = accelerator.prepare(
        stage1, stage2, optimizer, dataloader, scheduler
    )

    lambda_init = config['train'].get('phase3_lambda_init', 1.0)
    lambda_final = config['train'].get('phase3_lambda_final', 0.1)

    if accelerator.is_main_process:
        print(f"Phase 3: Joint fine-tuning for {epochs} epochs")
        print(f"  Stage 1 LR: {config['train']['phase3_lr_stage1']}")
        print(f"  Stage 2 LR: {config['train']['phase3_lr_stage2']}")
        print(f"  Lambda decay: {lambda_init} → {lambda_final}")

    step = 0
    for epoch in range(epochs):
        # Lambda decay for Stage 1 regularization
        progress = epoch / max(epochs - 1, 1)
        lambda_s1 = lambda_init + (lambda_final - lambda_init) * progress

        pbar = tqdm(dataloader, desc=f"P3 Epoch {epoch+1}/{epochs}",
                    disable=not accelerator.is_local_main_process)

        for batch in pbar:
            try:
                scan_points = batch['scan_points'].to(accelerator.device)
                scan_normals = batch.get('scan_normals')
                if scan_normals is not None:
                    scan_normals = scan_normals.to(accelerator.device)
                scan_curvature = batch['scan_curvature'].to(accelerator.device)

                gt_verts_list = [v.to(accelerator.device) for v in batch['gt_verts']]
                gt_faces_list = [f.to(accelerator.device) for f in batch['gt_faces']]

                optimizer.zero_grad(set_to_none=True)

                # Stage 1 forward (WITH mesh extraction, differentiable positions)
                base_verts, base_faces, base_normals, s1_aux = \
                    accelerator.unwrap_model(stage1)(scan_points, scan_normals)

                if base_faces[0].shape[0] == 0:
                    continue

                # Stage 1 loss (geometric regularization)
                loss_s1, loss_dict_s1 = compute_stage1_loss(
                    s1_aux['seed_positions'], s1_aux['seed_normals'],
                    scan_points, scan_normals, scan_curvature,
                    config['stage1'],
                    extracted_faces=base_faces[0],
                    extracted_verts=base_verts[0],
                )

                # Stage 2 forward
                f_verts, f_faces, disp, trans_feat, vertex_features, kl_loss = \
                    accelerator.unwrap_model(stage2)(
                        base_verts, base_faces, base_normals,
                        scan_points, scan_normals=scan_normals,
                    )

                g_verts = gt_verts_list[0].unsqueeze(0)
                g_faces = gt_faces_list[0].unsqueeze(0)

                if f_faces.max() >= f_verts.shape[1] or g_faces.max() >= g_verts.shape[1]:
                    if accelerator.is_main_process:
                        print("[Error] Invalid face indices, skipping")
                    continue

                f_faces_expanded = f_faces.unsqueeze(0)

                # Stage 2 rendering loss
                loss_s2 = _compute_stage2_rendering_loss(
                    f_verts, f_faces, f_faces_expanded,
                    g_verts, g_faces,
                    trans_feat, kl_loss, disp,
                    renderer, num_view_chunks, view_chunk,
                    config, epoch, accelerator,
                )

                if loss_s2 is None:
                    continue

                # Joint loss
                loss_total = loss_s2 + lambda_s1 * loss_s1

                accelerator.backward(loss_total)
                torch.nn.utils.clip_grad_norm_(stage1.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(stage2.parameters(), max_norm=1.0)
                optimizer.step()

                if accelerator.is_main_process and step % config['train']['log_interval'] == 0:
                    log_data = {
                        **loss_dict_s1,
                        's2/rendering_loss': loss_s2.item(),
                        'joint/total_loss': loss_total.item(),
                        'joint/lambda_s1': lambda_s1,
                        'joint/lr_s1': optimizer.param_groups[0]['lr'],
                        'joint/lr_s2': optimizer.param_groups[1]['lr'],
                        'joint/num_base_faces': base_faces[0].shape[0],
                        'joint/num_seeds': s1_aux['num_seeds'],
                    }
                    if not args.no_wandb:
                        accelerator.log(log_data, step=step)
                    pbar.set_postfix(
                        s1=f"{loss_s1.item():.3f}",
                        s2=f"{loss_s2.item():.3f}",
                        faces=base_faces[0].shape[0],
                    )

                step += 1

            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    if accelerator.is_main_process:
                        print(f"[OOM] Phase 3 step {step}, skipping")
                    continue
                raise

        scheduler.step()

        if (epoch + 1) % config['train']['save_interval'] == 0 and accelerator.is_main_process:
            s1_unwrap = accelerator.unwrap_model(stage1)
            s2_unwrap = accelerator.unwrap_model(stage2)
            save_path = os.path.join(ckpt_dir, f"joint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'stage1_state_dict': s1_unwrap.state_dict(),
                'stage2_state_dict': s2_unwrap.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, save_path)
            print(f"Saved: {save_path}")

    if accelerator.is_main_process:
        print(f"Phase 3 complete. Checkpoints in {ckpt_dir}")

    accelerator.end_training()


def _compute_stage2_rendering_loss(
    f_verts, f_faces, f_faces_expanded,
    g_verts, g_faces,
    trans_feat, kl_loss, disp,
    renderer, num_view_chunks, view_chunk,
    config, epoch, accelerator,
):
    """Compute Stage 2 rendering loss (view-chunked).

    Follows the same pattern as train_stage2.py lines 355-534.
    Returns total loss scalar or None if rendering fails.
    """
    device = accelerator.device

    # Non-render losses
    loss_lap = torch.tensor(0.0, device=device)
    w_lap = config['loss'].get('w_laplacian', 0.0)
    if w_lap > 0:
        loss_lap = compute_uniform_laplacian_l1(f_verts[0], f_faces)

    loss_kl = torch.tensor(0.0, device=device)
    if kl_loss is not None:
        vae_beta = config['loss'].get('vae_beta', 0.001)
        loss_kl = vae_beta * kl_loss

    w_mat = config['loss'].get('w_mat', 0.001)
    w_render = config['loss']['w_render']

    loss_non_render = w_lap * loss_lap + loss_kl

    # Feature transform regularization
    if trans_feat is not None and w_mat > 0:
        loss_mat = feature_transform_regularizer(trans_feat)
        loss_non_render = loss_non_render + w_mat * loss_mat

    # View-chunked rendering loss
    loss_render_total = torch.tensor(0.0, device=device)

    for vc in range(num_view_chunks):
        try:
            render_out = renderer(
                f_verts, f_faces_expanded,
                g_verts, g_faces,
            )
        except Exception:
            continue

        pred_normals = render_out['pred_normals']
        gt_normals = render_out['gt_normals']
        pred_mask = render_out['pred_mask']
        gt_mask = render_out['gt_mask']

        # AND mask for valid pixel comparison
        and_mask = (pred_mask * gt_mask).detach()
        valid_pixels = and_mask.sum().clamp(min=1.0)

        loss_chunk = torch.tensor(0.0, device=device)

        # Depth L1
        w_depth = config['loss'].get('w_depth_l1', 0.0)
        if w_depth > 0 and 'pred_depth' in render_out:
            pred_d = render_out['pred_depth']
            gt_d = render_out['gt_depth']
            loss_chunk = loss_chunk + w_depth * (
                (pred_d - gt_d).abs() * and_mask
            ).sum() / valid_pixels

        # Mask loss
        w_mask = config['loss'].get('w_mask', 0.0)
        if w_mask > 0:
            loss_chunk = loss_chunk + w_mask * F.binary_cross_entropy(
                pred_mask.clamp(1e-6, 1 - 1e-6), gt_mask, reduction='mean'
            )

        # Normal L1
        w_nl1 = config['loss'].get('w_normal_l1', 0.0)
        if w_nl1 > 0:
            loss_chunk = loss_chunk + w_nl1 * (
                (pred_normals - gt_normals).abs() * and_mask.unsqueeze(-1)
            ).sum() / (valid_pixels * 3)

        # Normal SSIM
        w_nssim = config['loss'].get('w_normal_ssim', 0.0)
        if w_nssim > 0 and HAS_SSIM:
            p_img = pred_normals.permute(0, 3, 1, 2)
            g_img = gt_normals.permute(0, 3, 1, 2)
            ssim_val = ssim(p_img, g_img, data_range=2.0, size_average=True)
            loss_chunk = loss_chunk + w_nssim * (1.0 - ssim_val)

        # Normal Geo
        normal_loss_type = config['loss'].get('normal_loss_type', 'l1')
        w_ngeo = config['loss'].get('w_normal_geo', 0.0)
        if normal_loss_type == 'geo' and w_ngeo > 0:
            loss_geo = compute_normal_loss_geo(pred_normals, gt_normals, and_mask)
            loss_chunk = loss_chunk + w_ngeo * loss_geo

        loss_render_total = loss_render_total + loss_chunk / num_view_chunks

    total = loss_non_render + w_render * loss_render_total

    if torch.isnan(total) or torch.isinf(total):
        return None

    return total


def main():
    parser = argparse.ArgumentParser(description="Joint Stage 1 + Stage 2 Training")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--phase', type=int, required=True, choices=[1, 2, 3])
    parser.add_argument('--stage1-ckpt', type=str, default=None)
    parser.add_argument('--stage2-ckpt', type=str, default=None)
    parser.add_argument('--no-wandb', action='store_true')
    args = parser.parse_args()

    config = load_config(args.config)

    if args.phase == 1:
        train_phase1(config, args)
    elif args.phase == 2:
        train_phase2(config, args)
    elif args.phase == 3:
        train_phase3(config, args)


if __name__ == '__main__':
    main()
