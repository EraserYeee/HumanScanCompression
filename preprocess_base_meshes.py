import os
import glob
import argparse
import torch
import numpy as np
import pymeshlab
from multiprocessing import Pool
from tqdm import tqdm
import json
import random


def get_pymeshlab_mesh_set(verts_np, faces_np):
    ms = pymeshlab.MeshSet()
    m = pymeshlab.Mesh(vertex_matrix=verts_np, face_matrix=faces_np)
    ms.add_mesh(m)
    return ms


def simplify_mesh_pymeshlab(verts_np, faces_np, target_faces):
    """
    High-quality mesh simplification via QEM with topology/normal preservation.
    Includes pre-repair and post-cleanup to avoid single-sided faces, self-intersections,
    and degenerate thin structures.
    """
    try:
        ms = get_pymeshlab_mesh_set(
            verts_np.astype(np.float64), faces_np.astype(np.int32)
        )

        # --- Pre-repair ---
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

        # --- QEM decimation with topology preservation ---
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=target_faces,
            targetperc=0.0,
            qualitythr=0.8,
            preservenormal=True,
            preservetopology=True,
            optimalplacement=True,
            planarquadric=True,
        )

        # --- Post-cleanup ---
        ms.meshing_remove_folded_faces()
        ms.meshing_remove_null_faces()
        ms.meshing_remove_duplicate_faces()
        try:
            ms.meshing_repair_non_manifold_edges(method=0)
        except Exception:
            pass
        try:
            ms.meshing_repair_non_manifold_vertices()
        except Exception:
            pass
        ms.meshing_remove_unreferenced_vertices()

        m = ms.current_mesh()
        new_verts = m.vertex_matrix().astype(np.float32)
        new_faces = m.face_matrix().astype(np.int64)

        return new_verts, new_faces

    except Exception as e:
        print(f"[Warning] Simplification failed: {e}")
        return verts_np, faces_np


def process_single_pt_file(args):
    """
    Load a GT .pt file and generate multiple base meshes at random face counts.
    """
    pt_path, output_dir, min_faces, max_faces, num_variants = args

    try:
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        gt_verts = data["gt_verts"].numpy()
        gt_faces = data["gt_faces"].numpy()

        filename = os.path.basename(pt_path)
        name, _ = os.path.splitext(filename)

        mesh_out_dir = os.path.join(output_dir, name)
        os.makedirs(mesh_out_dir, exist_ok=True)

        results = []

        target_counts = [random.randint(min_faces, max_faces) for _ in range(num_variants)]

        for i, target_f in enumerate(target_counts):
            base_verts, base_faces = simplify_mesh_pymeshlab(gt_verts, gt_faces, target_f)

            save_name = f"{name}_base_{i}_f{target_f}.pt"
            save_path = os.path.join(mesh_out_dir, save_name)

            torch.save(
                {
                    "base_verts": torch.from_numpy(base_verts),
                    "base_faces": torch.from_numpy(base_faces),
                    "gt_path": pt_path,
                    "target_faces": target_f,
                },
                save_path,
            )

            results.append(
                {
                    "base_pt_path": os.path.join(name, save_name),
                    "gt_pt_path": os.path.basename(pt_path),
                    "base_faces": int(base_faces.shape[0]),
                }
            )

        return results

    except Exception as e:
        print(f"[Error] Failed to process {pt_path}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Generate simplified base meshes from GT .pt files for training."
    )
    parser.add_argument(
        "--pt_dir",
        type=str,
        required=True,
        help="Directory containing GT .pt files (from preprocess_data.py)",
    )
    parser.add_argument(
        "--base_output_dir",
        type=str,
        required=True,
        help="Output directory for base meshes",
    )
    parser.add_argument(
        "--index_json",
        type=str,
        default=None,
        help="Optional path to a single split index JSON (e.g. train.json). If not set, will look for pt_dir/train.json and pt_dir/test.json.",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of worker processes"
    )
    parser.add_argument(
        "--min_faces", type=int, default=2000, help="Min target face count"
    )
    parser.add_argument(
        "--max_faces", type=int, default=3000, help="Max target face count"
    )
    parser.add_argument(
        "--num_variants",
        type=int,
        default=5,
        help="Number of random simplification variants per mesh",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.base_output_dir, exist_ok=True)
    print("Sleeping for 40 minutes before starting to process base meshes...")
    import time
    time.sleep(30 * 60)
    print("Woke up! Starting base mesh processing.")

    def resolve_pt_files_from_index(index_path: str):
        with open(index_path, "r") as f:
            entries = json.load(f)
        pt_files = []
        for e in entries:
            p = os.path.join(args.pt_dir, e["pt_path"])
            if os.path.exists(p):
                pt_files.append(p)
        return pt_files

    def run_split(split_name: str, pt_files: list[str]):
        if len(pt_files) == 0:
            print(f"[{split_name}] No .pt files found, skipping.")
            return

        tasks = [
            (p, args.base_output_dir, args.min_faces, args.max_faces, args.num_variants)
            for p in pt_files
        ]

        all_results = []
        with Pool(processes=args.workers) as pool:
            for res in tqdm(
                pool.imap_unordered(process_single_pt_file, tasks),
                total=len(tasks),
                desc=f"Generating base meshes ({split_name})",
            ):
                if res:
                    all_results.extend(res)

        base_json = os.path.join(args.base_output_dir, f"{split_name}_base.json")
        with open(base_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"[{split_name}] Generated {len(all_results)} base meshes for {len(pt_files)} GT models.")
        print(f"[{split_name}] Index saved to: {base_json}")

    # Mode A: user provides a single split index_json
    if args.index_json and os.path.exists(args.index_json):
        split_name = os.path.splitext(os.path.basename(args.index_json))[0]
        pt_files = resolve_pt_files_from_index(args.index_json)
        print(f"Loaded {len(pt_files)} entries from {args.index_json}")
        run_split(split_name, pt_files)
        return

    # Mode B: follow preprocess_data.py output (train.json + test.json)
    train_index = os.path.join(args.pt_dir, "train.json")
    test_index = os.path.join(args.pt_dir, "test.json")
    if os.path.exists(train_index) or os.path.exists(test_index):
        if os.path.exists(train_index):
            run_split("train", resolve_pt_files_from_index(train_index))
        else:
            print("[train] train.json not found, skipping.")

        if os.path.exists(test_index):
            run_split("test", resolve_pt_files_from_index(test_index))
        else:
            print("[test] test.json not found, skipping.")
        return

    # Mode C: fallback - scan pt_dir for *.pt and treat as train
    pt_files = sorted(glob.glob(os.path.join(args.pt_dir, "*.pt")))
    print(f"Found {len(pt_files)} .pt files in {args.pt_dir} (no train/test json found)")
    run_split("train", pt_files)


if __name__ == "__main__":
    main()
