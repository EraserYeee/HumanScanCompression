"""
Demo: Sharp Edge Sampling vs Uniform Sampling on a mesh.

Usage:
    python demo_sharp_sampling.py --input mesh.obj \
        --point_num 819200 --angle_threshold 30 --sharp_ratio 0.5 \
        --output_dir demo_output

Outputs (PLY point clouds):
    uniform.ply      – 100% uniform surface sampling
    sharp_only.ply   – 100% sharp-edge sampling (with fallback uniform fill)
    mixed.ply        – sharp_ratio% sharp + rest uniform
"""

import argparse
import os
import time
import numpy as np
import trimesh
import open3d as o3d


# ──────────────────────────────────────────────────────────────────────
#  Core: sharp edge detection + edge-interpolation sampling (trimesh)
# ──────────────────────────────────────────────────────────────────────

def detect_sharp_edges(mesh: trimesh.Trimesh, angle_threshold_deg: float):
    """Return sharp edge vertex-pairs, per-edge face-pair indices, and angles.

    Returns:
        sharp_edge_verts : (E, 2)    vertex indices for each sharp edge
        sharp_face_pairs : (E, 2)    adjacent face indices
        sharp_angles_deg : (E,)      dihedral angle in degrees
    """
    angles = mesh.face_adjacency_angles            # (A,)  radians
    threshold_rad = np.deg2rad(angle_threshold_deg)
    mask = angles >= threshold_rad

    sharp_edge_verts = mesh.face_adjacency_edges[mask]   # (E, 2)
    sharp_face_pairs = mesh.face_adjacency[mask]         # (E, 2)
    sharp_angles_deg = np.rad2deg(angles[mask])          # (E,)
    return sharp_edge_verts, sharp_face_pairs, sharp_angles_deg


def sample_on_sharp_edges(
    mesh: trimesh.Trimesh,
    sharp_edge_verts: np.ndarray,
    sharp_face_pairs: np.ndarray,
    target_num: int,
):
    """Interpolate points along sharp edges (Dora-style).

    Returns:
        points  : (N, 3)
        normals : (N, 3)   averaged normals of the two adjacent faces
    """
    verts = mesh.vertices
    face_normals = mesh.face_normals

    E = len(sharp_edge_verts)
    if E == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    # Unique sharp vertices as seed points
    unique_idx = np.unique(sharp_edge_verts.ravel())
    known_pts = verts[unique_idx]                              # (U, 3)
    known_norms = mesh.vertex_normals[unique_idx]              # (U, 3)

    num_known = len(known_pts)
    num_need = max(target_num - num_known, 0)

    # Edge normals = average of the two adjacent face normals
    n1 = face_normals[sharp_face_pairs[:, 0]]
    n2 = face_normals[sharp_face_pairs[:, 1]]
    edge_normals = 0.5 * (n1 + n2)
    norms_len = np.linalg.norm(edge_normals, axis=1, keepdims=True).clip(1e-8)
    edge_normals = edge_normals / norms_len

    start = verts[sharp_edge_verts[:, 0]]
    end   = verts[sharp_edge_verts[:, 1]]

    interp_pts_list = []
    interp_nrm_list = []

    if num_need > 0 and E > 0:
        if num_need >= E:
            per_edge = num_need // E
            remainder = num_need % E

            # Uniform interpolation per edge
            for j in range(1, per_edge + 1):
                t = j / (per_edge + 1)
                pts = (1 - t) * start + t * end           # (E, 3)
                interp_pts_list.append(pts)
                interp_nrm_list.append(edge_normals)

            # Distribute remainder randomly
            if remainder > 0:
                rng = np.random.default_rng()
                sel = rng.choice(E, remainder, replace=False)
                t = np.random.rand(remainder, 1).astype(np.float32)
                pts = (1 - t) * start[sel] + t * end[sel]
                interp_pts_list.append(pts)
                interp_nrm_list.append(edge_normals[sel])
        else:
            # Fewer points needed than edges → random subset
            rng = np.random.default_rng()
            sel = rng.choice(E, num_need, replace=False)
            t = np.random.rand(num_need, 1).astype(np.float32)
            pts = (1 - t) * start[sel] + t * end[sel]
            interp_pts_list.append(pts)
            interp_nrm_list.append(edge_normals[sel])

    all_pts = [known_pts]
    all_nrm = [known_norms]
    if interp_pts_list:
        all_pts.append(np.concatenate(interp_pts_list, axis=0))
        all_nrm.append(np.concatenate(interp_nrm_list, axis=0))

    all_pts = np.concatenate(all_pts, axis=0).astype(np.float32)
    all_nrm = np.concatenate(all_nrm, axis=0).astype(np.float32)

    # Trim or pad to exact target_num
    if len(all_pts) > target_num:
        idx = np.random.choice(len(all_pts), target_num, replace=False)
        all_pts = all_pts[idx]
        all_nrm = all_nrm[idx]

    return all_pts, all_nrm


def sharp_edge_sample(
    mesh: trimesh.Trimesh,
    total_points: int,
    angle_threshold_deg: float = 30.0,
    sharp_ratio: float = 0.5,
    min_sharp_fallback: float = 0.05,
):
    """Combined sharp + uniform sampling (to be used in __getitem__).

    Returns:
        points  : (total_points, 3)
        normals : (total_points, 3)
        stats   : dict with diagnostic info
    """
    stats = {}

    t0 = time.time()
    sharp_edge_verts, sharp_face_pairs, sharp_angles = detect_sharp_edges(
        mesh, angle_threshold_deg
    )
    stats['detect_time'] = time.time() - t0
    stats['num_sharp_edges'] = len(sharp_edge_verts)
    stats['angle_mean'] = float(sharp_angles.mean()) if len(sharp_angles) > 0 else 0.0
    stats['angle_max'] = float(sharp_angles.max()) if len(sharp_angles) > 0 else 0.0

    num_sharp_target = int(total_points * sharp_ratio)
    num_uniform_target = total_points - num_sharp_target

    # Sharp edge sampling
    t1 = time.time()
    sharp_pts, sharp_nrm = sample_on_sharp_edges(
        mesh, sharp_edge_verts, sharp_face_pairs, num_sharp_target
    )
    stats['sharp_sample_time'] = time.time() - t1
    stats['num_sharp_sampled'] = len(sharp_pts)

    # Fallback: if sharp points are too few, fill with uniform
    if len(sharp_pts) < num_sharp_target * min_sharp_fallback:
        stats['sharp_fallback'] = True
        num_uniform_target = total_points
        sharp_pts = np.zeros((0, 3), dtype=np.float32)
        sharp_nrm = np.zeros((0, 3), dtype=np.float32)
    else:
        stats['sharp_fallback'] = False
        # If we didn't get enough sharp points, give the rest to uniform
        actual_sharp = len(sharp_pts)
        num_uniform_target = total_points - actual_sharp

    # Uniform surface sampling
    t2 = time.time()
    uniform_pts, face_idx = trimesh.sample.sample_surface(mesh, num_uniform_target)
    uniform_pts = uniform_pts.astype(np.float32)
    uniform_nrm = mesh.face_normals[face_idx].astype(np.float32)
    stats['uniform_sample_time'] = time.time() - t2

    # Concatenate
    if len(sharp_pts) > 0:
        points = np.concatenate([sharp_pts, uniform_pts], axis=0)
        normals = np.concatenate([sharp_nrm, uniform_nrm], axis=0)
    else:
        points = uniform_pts
        normals = uniform_nrm

    stats['total_points'] = len(points)
    return points, normals, stats


# ──────────────────────────────────────────────────────────────────────
#  I/O helpers
# ──────────────────────────────────────────────────────────────────────

def save_ply(path, points, normals=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    # Color by z-height for visualization
    z = points[:, 2]
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
    colors = np.stack([z_norm, 0.4 * np.ones_like(z_norm), 1.0 - z_norm], axis=1)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(path, pcd, write_ascii=False)


def normalize_to_unit_sphere(vertices):
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    center = (bbox_min + bbox_max) / 2
    vertices = vertices - center
    scale = np.max(np.linalg.norm(vertices, axis=1))
    if scale < 1e-6:
        scale = 1.0
    return vertices / scale


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sharp Edge Sampling Demo")
    parser.add_argument("--input", type=str, required=True, help="Input mesh path (.obj / .ply / .stl)")
    parser.add_argument("--point_num", type=int, default=819200)
    parser.add_argument("--angle_threshold", type=float, default=30.0, help="Dihedral angle threshold (degrees)")
    parser.add_argument("--sharp_ratio", type=float, default=0.5, help="Fraction of points from sharp edges")
    parser.add_argument("--output_dir", type=str, default="demo_sharp_output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load & normalize
    print(f"Loading mesh: {args.input}")
    t_load = time.time()
    mesh = trimesh.load(args.input, force='mesh', process=False)
    verts = normalize_to_unit_sphere(mesh.vertices)
    mesh = trimesh.Trimesh(vertices=verts, faces=mesh.faces, process=False)
    t_load = time.time() - t_load
    print(f"  Loaded: V={len(mesh.vertices)}, F={len(mesh.faces)}")
    print(f"  Load + normalize time: {t_load:.3f}s")
    print()

    # ── 1. Uniform only ──────────────────────────────────────────────
    print("=" * 60)
    print("[1] Uniform surface sampling")
    t0 = time.time()
    uniform_pts, face_idx = trimesh.sample.sample_surface(mesh, args.point_num)
    uniform_pts = uniform_pts.astype(np.float32)
    uniform_nrm = mesh.face_normals[face_idx].astype(np.float32)
    t_uniform = time.time() - t0
    print(f"  Points: {len(uniform_pts)}")
    print(f"  Time:   {t_uniform:.3f}s")
    out_path = os.path.join(args.output_dir, "uniform.ply")
    save_ply(out_path, uniform_pts, uniform_nrm)
    print(f"  Saved:  {out_path}")
    print()

    # ── 2. Sharp edge only (ratio=1.0) ───────────────────────────────
    print("=" * 60)
    print("[2] Sharp edge sampling only (ratio=1.0)")
    sharp_only_pts, sharp_only_nrm, stats_sharp = sharp_edge_sample(
        mesh, args.point_num,
        angle_threshold_deg=args.angle_threshold,
        sharp_ratio=1.0,
        min_sharp_fallback=0.0,
    )
    print(f"  Sharp edges detected: {stats_sharp['num_sharp_edges']}")
    if stats_sharp['num_sharp_edges'] > 0:
        print(f"  Angle range: mean={stats_sharp['angle_mean']:.1f}°, max={stats_sharp['angle_max']:.1f}°")
    print(f"  Sharp points sampled: {stats_sharp['num_sharp_sampled']}")
    print(f"  Uniform fill points:  {stats_sharp['total_points'] - stats_sharp['num_sharp_sampled']}")
    print(f"  Timing:")
    print(f"    detect sharp edges:  {stats_sharp['detect_time']:.3f}s")
    print(f"    sample sharp points: {stats_sharp['sharp_sample_time']:.3f}s")
    print(f"    sample uniform fill: {stats_sharp['uniform_sample_time']:.3f}s")
    total_sharp = stats_sharp['detect_time'] + stats_sharp['sharp_sample_time'] + stats_sharp['uniform_sample_time']
    print(f"    total:               {total_sharp:.3f}s")
    out_path = os.path.join(args.output_dir, "sharp_only.ply")
    save_ply(out_path, sharp_only_pts, sharp_only_nrm)
    print(f"  Saved:  {out_path}")
    print()

    # ── 3. Mixed (sharp + uniform) ───────────────────────────────────
    print("=" * 60)
    print(f"[3] Mixed sampling (sharp_ratio={args.sharp_ratio})")
    mixed_pts, mixed_nrm, stats_mix = sharp_edge_sample(
        mesh, args.point_num,
        angle_threshold_deg=args.angle_threshold,
        sharp_ratio=args.sharp_ratio,
    )
    print(f"  Sharp edges detected: {stats_mix['num_sharp_edges']}")
    print(f"  Sharp points:         {stats_mix['num_sharp_sampled']}")
    print(f"  Uniform points:       {stats_mix['total_points'] - stats_mix['num_sharp_sampled']}")
    print(f"  Fallback to uniform:  {stats_mix['sharp_fallback']}")
    print(f"  Timing:")
    print(f"    detect sharp edges:  {stats_mix['detect_time']:.3f}s")
    print(f"    sample sharp points: {stats_mix['sharp_sample_time']:.3f}s")
    print(f"    sample uniform fill: {stats_mix['uniform_sample_time']:.3f}s")
    total_mix = stats_mix['detect_time'] + stats_mix['sharp_sample_time'] + stats_mix['uniform_sample_time']
    print(f"    total:               {total_mix:.3f}s")
    out_path = os.path.join(args.output_dir, "mixed.ply")
    save_ply(out_path, mixed_pts, mixed_nrm)
    print(f"  Saved:  {out_path}")
    print()

    # ── Summary ──────────────────────────────────────────────────────
    print("=" * 60)
    print("Summary")
    print(f"  Mesh:            V={len(mesh.vertices)}, F={len(mesh.faces)}")
    print(f"  point_num:       {args.point_num}")
    print(f"  angle_threshold: {args.angle_threshold}°")
    print(f"  sharp_ratio:     {args.sharp_ratio}")
    print(f"  sharp_edges:     {stats_mix['num_sharp_edges']}")
    print()
    print(f"  Uniform only:    {t_uniform:.3f}s")
    print(f"  Sharp only:      {total_sharp:.3f}s")
    print(f"  Mixed:           {total_mix:.3f}s")
    print()
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
