import os
import glob
import argparse
import json
import pickle
import random
import torch
import trimesh
import numpy as np
import pymeshlab
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def load_smplx_scale(smplx_dir: str, mesh_id: str) -> float:
    """Load scale from smplx_param.pkl for a given THuman mesh id."""
    pkl_path = os.path.join(smplx_dir, mesh_id, "smplx_param.pkl")
    if not os.path.exists(pkl_path):
        return None
    params = np.load(pkl_path, allow_pickle=True)
    scale = params["scale"]
    if hasattr(scale, "__len__"):
        scale = float(scale[0])
    return float(scale)


def repair_mesh_pymeshlab(verts: np.ndarray, faces: np.ndarray):
    """
    Run basic mesh repair filters via pymeshlab.
    No remeshing, no component removal — only topological/geometric fixes.
    """
    ms = pymeshlab.MeshSet()
    m = pymeshlab.Mesh(
        vertex_matrix=verts.astype(np.float64),
        face_matrix=faces.astype(np.int32),
    )
    ms.add_mesh(m)

    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_null_faces()

    try:
        ms.meshing_repair_non_manifold_edges(method=0)
    except Exception:
        pass
    try:
        ms.meshing_repair_non_manifold_vertices()
    except Exception:
        pass

    out = ms.current_mesh()
    return (
        out.vertex_matrix().astype(np.float32),
        out.face_matrix().astype(np.int64),
    )


def process_mesh(args):
    """
    Process a single THuman mesh:
    1. Load OBJ
    2. Divide vertices by smplx scale (restore to ~1.7m)
    3. Keep largest connected component
    4. Repair (no remesh)
    5. Save as .pt
    """
    path, output_dir, smplx_dir, apply_scale, do_repair = args
    try:
        mesh = trimesh.load(path, process=False, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) == 0:
                return None
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

        verts = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int64)

        # --- Scale restoration + center to origin ---
        if apply_scale and smplx_dir:
            parent_dir = os.path.basename(os.path.dirname(path))
            mesh_id = parent_dir
            scale = load_smplx_scale(smplx_dir, mesh_id)
            if scale is not None and scale > 0:
                verts = verts / scale
            else:
                print(f"[Warning] No valid scale for {mesh_id}, skipping scale restoration")

        vmin = verts.min(axis=0)
        vmax = verts.max(axis=0)
        center = (vmin + vmax) / 2.0
        verts = verts - center

        # --- Repair (no remesh) ---
        if do_repair:
            verts, faces = repair_mesh_pymeshlab(verts, faces)

        gt_verts = torch.from_numpy(verts).float()
        gt_faces = torch.from_numpy(faces).long()

        # Use mesh id (parent folder name) as filename
        parent_dir = os.path.basename(os.path.dirname(path))
        save_filename = f"{parent_dir}.pt"
        save_path = os.path.join(output_dir, save_filename)

        torch.save(
            {
                "gt_verts": gt_verts,
                "gt_faces": gt_faces,
                "original_path": path,
            },
            save_path,
        )

        return {"pt_path": save_filename}

    except Exception as e:
        print(f"[Error] Failed to process {path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess THuman2.1 OBJ meshes to PT files: scale restore + clean + repair."
    )
    parser.add_argument(
        "--src_dir",
        type=str,
        required=True,
        help="THuman2.1 model directory (e.g. data/official/THuman2.1/model)",
    )
    parser.add_argument(
        "--smplx_dir",
        type=str,
        default=None,
        help="THuman2.1 smplx directory (e.g. data/official/THuman2.1/smplx)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for .pt files and index json",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Fraction of data reserved for test set (default: 0.1 = 10%%)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for train/test split"
    )
    parser.add_argument(
        "--apply_scale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Divide vertices by smplx scale to restore real-world size",
    )
    parser.add_argument(
        "--repair",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run pymeshlab repair filters (no remeshing)",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of worker processes"
    )
    args = parser.parse_args()

    if args.apply_scale and args.smplx_dir is None:
        parser.error("--smplx_dir is required when --apply_scale is enabled")

    os.makedirs(args.out_dir, exist_ok=True)

    # THuman2.1 layout: src_dir/<id>/<id>.obj
    print(f"Scanning {args.src_dir} for THuman meshes...")
    mesh_dirs = sorted(
        [
            d
            for d in os.listdir(args.src_dir)
            if os.path.isdir(os.path.join(args.src_dir, d))
        ]
    )
    files = []
    for d in mesh_dirs:
        obj_path = os.path.join(args.src_dir, d, f"{d}.obj")
        if os.path.exists(obj_path):
            files.append(obj_path)

    if len(files) == 0:
        print("No THuman OBJ files found.")
        return

    print(f"Found {len(files)} meshes. Processing with {args.workers} workers...")

    process_args = [
        (f, args.out_dir, args.smplx_dir, args.apply_scale, args.repair)
        for f in files
    ]

    results = []
    with Pool(processes=args.workers) as pool:
        for res in tqdm(
            pool.imap_unordered(process_mesh, process_args), total=len(files)
        ):
            if res is not None:
                results.append(res)

    # --- Train / Test split ---
    results.sort(key=lambda x: x["pt_path"])
    random.seed(args.seed)
    indices = list(range(len(results)))
    random.shuffle(indices)

    n_test = max(1, int(len(results) * args.test_ratio))
    test_indices = set(indices[:n_test])

    train_results = [results[i] for i in range(len(results)) if i not in test_indices]
    test_results = [results[i] for i in range(len(results)) if i in test_indices]

    train_json = os.path.join(args.out_dir, "train.json")
    test_json = os.path.join(args.out_dir, "test.json")
    with open(train_json, "w") as f:
        json.dump(train_results, f, indent=4)
    with open(test_json, "w") as f:
        json.dump(test_results, f, indent=4)

    print(f"Successfully processed {len(results)} / {len(files)} samples.")
    print(f"Train: {len(train_results)} | Test: {len(test_results)}")
    print(f"Saved: {train_json}")
    print(f"Saved: {test_json}")


if __name__ == "__main__":
    main()
