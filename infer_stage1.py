"""Stage 1 inference: point cloud → base mesh.

Usage:
    # From a preprocessed .pt file (same format as dataset)
    python infer_stage1.py \
        --checkpoint checkpoints/stage1_final.pth \
        --config configs/config_joint.yaml \
        --input /path/to/sample.pt \
        --output output_base_mesh.obj

    # From a raw .ply point cloud
    python infer_stage1.py \
        --checkpoint checkpoints/stage1_final.pth \
        --config configs/config_joint.yaml \
        --input /path/to/scan.ply \
        --output output_base_mesh.obj \
        --num-seeds 1000
"""

import argparse
import os
import torch
import numpy as np
import trimesh

from models.stage1_pipeline import Stage1Pipeline
from utils.train_helpers import load_config


def load_point_cloud(path, point_num=819200):
    """Load point cloud from .pt (dataset format) or .ply/.obj file."""
    ext = os.path.splitext(path)[1].lower()

    if ext == '.pt':
        data = torch.load(path, weights_only=False)
        verts = data['gt_verts'].numpy()
        faces = data['gt_faces'].numpy()
        # Normalize (same as dataset)
        bbox_min, bbox_max = verts.min(0), verts.max(0)
        center = (bbox_min + bbox_max) / 2
        verts = verts - center
        scale = max(np.linalg.norm(verts, axis=1).max(), 1e-6)
        verts = verts / scale
        # Sample surface
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        points, face_idx = mesh.sample(point_num, return_index=True)
        normals = mesh.face_normals[face_idx]
        return (
            torch.from_numpy(points.astype(np.float32)),
            torch.from_numpy(normals.astype(np.float32)),
        )
    else:
        # .ply or .obj
        cloud = trimesh.load(path)
        if isinstance(cloud, trimesh.Trimesh):
            points, face_idx = cloud.sample(point_num, return_index=True)
            normals = cloud.face_normals[face_idx]
        elif isinstance(cloud, trimesh.PointCloud):
            points = np.array(cloud.vertices)
            normals = np.zeros_like(points)  # no normals available
            if len(points) > point_num:
                idx = np.random.choice(len(points), point_num, replace=False)
                points, normals = points[idx], normals[idx]
        else:
            raise ValueError(f"Unsupported format: {ext}")
        return (
            torch.from_numpy(points.astype(np.float32)),
            torch.from_numpy(normals.astype(np.float32)),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='stage1_output.obj')
    parser.add_argument('--num-seeds', type=int, default=None)
    parser.add_argument('--point-num', type=int, default=819200)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--export-seeds', action='store_true',
                        help='Also export seed point cloud as .ply')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device)

    # Load model
    stage1 = Stage1Pipeline(config['stage1'])
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt.get('stage1_state_dict', ckpt))
    stage1.load_state_dict(state)
    stage1 = stage1.to(device).eval()
    print(f"Loaded Stage 1 from {args.checkpoint}")

    # Load input
    scan_points, scan_normals = load_point_cloud(args.input, args.point_num)
    scan_points = scan_points.unsqueeze(0).to(device)
    scan_normals = scan_normals.unsqueeze(0).to(device)
    print(f"Input: {scan_points.shape[1]} points")

    # Inference
    with torch.no_grad():
        base_verts, base_faces, base_normals, aux = stage1(
            scan_points, scan_normals, num_seeds=args.num_seeds
        )

    v = base_verts[0].cpu().numpy()
    f = base_faces[0].cpu().numpy()
    print(f"Output: {len(v)} vertices, {len(f)} faces")

    # Export mesh
    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
    mesh.export(args.output)
    print(f"Saved: {args.output}")

    # Optionally export seeds
    if args.export_seeds:
        seeds_path = args.output.replace('.obj', '_seeds.ply')
        seeds = aux['seed_positions'][0].cpu().numpy()
        trimesh.PointCloud(seeds).export(seeds_path)
        print(f"Saved seeds: {seeds_path} ({len(seeds)} points)")


if __name__ == '__main__':
    main()
