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


def simplify_mesh_adaptive(verts_np, faces_np, target_faces, curvature_smooth_iters=10):
    """
    Curvature-adaptive mesh simplification via quality-weighted QEM.

    Computes per-vertex absolute curvature (|k1| + |k2|) on the input mesh,
    smooths it, then uses it as vertex quality for QEM decimation.
    High-curvature regions (fingers, face, creases) are preserved preferentially;
    flat regions (torso, limbs) are simplified more aggressively.

    All quality manipulation is done through pymeshlab filters (no direct array access),
    for compatibility across pymeshlab versions.
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

        # --- Compute absolute curvature (|k1| + |k2|) -> vertex quality ---
        ms.compute_scalar_by_discrete_curvature_per_vertex(curvaturetype=3)

        # --- Smooth curvature field to reduce scan noise ---
        # 10 iterations of Laplacian smoothing attenuates outlier spikes well
        for _ in range(curvature_smooth_iters):
            ms.apply_scalar_smoothing_per_vertex()

        # --- Normalize quality in-place via filter (sqrt + floor) ---
        try:
            # ms.compute_scalar_by_function_per_vertex(q="sqrt(max(q, 0)) + 0.1")
            ms.compute_scalar_by_function_per_vertex(q="max(q, 0)+0.1")
        except Exception:
            pass  # raw smoothed curvature still works fine with qualityweight

        # --- Quality-weighted QEM decimation ---
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=target_faces,
            targetperc=0.0,
            qualitythr=0.8,
            preservenormal=True,
            preservetopology=True,
            optimalplacement=True,
            planarquadric=True,
            qualityweight=True,
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
        print(f"[Warning] Adaptive simplification failed: {e}, falling back to standard QEM")
        return simplify_mesh_pymeshlab(verts_np, faces_np, target_faces)


def process_single_pt_file(args):
    """
    Load a GT .pt file and generate multiple base meshes at random face counts.
    """
    pt_path, output_dir, min_faces, max_faces, num_variants, method = args

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

        simplify_fn = simplify_mesh_adaptive if method == "adaptive" else simplify_mesh_pymeshlab

        for i, target_f in enumerate(target_counts):
            base_verts, base_faces = simplify_fn(gt_verts, gt_faces, target_f)

            save_name = f"{name}_base_{i}_f{target_f}.pt"
            save_path = os.path.join(mesh_out_dir, save_name)

            torch.save(
                {
                    "base_verts": torch.from_numpy(base_verts),
                    "base_faces": torch.from_numpy(base_faces),
                    "gt_path": pt_path,
                    "target_faces": target_f,
                    "method": method,
                },
                save_path,
            )

            results.append(
                {
                    "base_pt_path": os.path.join(name, save_name),
                    "gt_pt_path": os.path.basename(pt_path),
                    "base_faces": int(base_faces.shape[0]),
                    "target_faces": target_f,
                }
            )

        return results

    except Exception as e:
        print(f"[Error] Failed to process {pt_path}: {e}")
        return []


def _save_obj(verts, faces, path):
    """Write a triangle mesh to OBJ (1-indexed faces)."""
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")


def export_sample_objs(all_results, base_output_dir, pt_dir, export_dir,
                       num_export=10, method="adaptive"):
    """Export random base mesh + GT scan pairs as OBJ for visual inspection.

    For each sample exports:
      - {name}_gt.obj                  : GT scan mesh
      - {name}_{method}_{F}f.obj       : base mesh from chosen method
      - {name}_qem_{F}f.obj            : standard QEM base (for comparison, if method != qem)
    """
    os.makedirs(export_dir, exist_ok=True)

    samples = random.sample(all_results, min(num_export, len(all_results)))

    print(f"\nExporting {len(samples)} sample pairs to {export_dir}/")
    for i, entry in enumerate(samples):
        gt_pt = os.path.join(pt_dir, entry["gt_pt_path"])
        base_pt = os.path.join(base_output_dir, entry["base_pt_path"])

        gt_data = torch.load(gt_pt, map_location="cpu", weights_only=False)
        gt_v = gt_data["gt_verts"].numpy()
        gt_f = gt_data["gt_faces"].numpy()

        base_data = torch.load(base_pt, map_location="cpu", weights_only=False)
        base_v = base_data["base_verts"].numpy()
        base_f = base_data["base_faces"].numpy()
        target_f = entry["target_faces"]

        name = os.path.splitext(entry["gt_pt_path"])[0]
        prefix = f"{i:02d}_{name}"

        # GT scan mesh
        _save_obj(gt_v, gt_f, os.path.join(export_dir, f"{prefix}_gt.obj"))

        # Base mesh from chosen method
        _save_obj(
            base_v, base_f,
            os.path.join(export_dir, f"{prefix}_{method}_{base_f.shape[0]}f.obj"),
        )

        # Comparison: generate standard QEM at same target face count
        if method != "qem":
            qem_v, qem_f = simplify_mesh_pymeshlab(gt_v, gt_f, target_f)
            _save_obj(
                qem_v, qem_f,
                os.path.join(export_dir, f"{prefix}_qem_{qem_f.shape[0]}f.obj"),
            )
            extra = f", QEM {qem_f.shape[0]}f"
        else:
            extra = ""

        print(
            f"  [{i+1}/{len(samples)}] {name}: "
            f"GT {gt_f.shape[0]}f, {method} {base_f.shape[0]}f{extra}"
        )


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
        "--workers", type=int, default=4, help="Number of worker processes"
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
    parser.add_argument(
        "--method",
        type=str,
        default="qem",
        choices=["qem", "adaptive"],
        help="Simplification method: 'qem' (standard) or 'adaptive' (curvature-weighted QEM)",
    )
    parser.add_argument(
        "--export_dir",
        type=str,
        default="export_samples",
        help="Directory to export sample OBJ pairs for visual inspection",
    )
    parser.add_argument(
        "--num_export",
        type=int,
        default=10,
        help="Number of random sample pairs to export as OBJ (0 to skip)",
    )
    parser.add_argument(
        "--early_export_every",
        type=int,
        default=0,
        help=(
            "If >0, export --num_export sample OBJs every N completed .pt files "
            "(imap checkpoint; 0 = only export after all processing in main, as before)"
        ),
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.base_output_dir, exist_ok=True)


    def resolve_pt_files_from_index(index_path: str):
        with open(index_path, "r") as f:
            entries = json.load(f)
        pt_files = []
        for e in entries:
            p = os.path.join(args.pt_dir, e["pt_path"])
            if os.path.exists(p):
                pt_files.append(p)
        return pt_files

    def run_split(split_name: str, pt_files: list, args):
        if len(pt_files) == 0:
            print(f"[{split_name}] No .pt files found, skipping.")
            return []

        tasks = [
            (p, args.base_output_dir, args.min_faces, args.max_faces,
             args.num_variants, args.method)
            for p in pt_files
        ]

        all_results = []
        completed = 0
        with Pool(processes=args.workers) as pool:
            for res in tqdm(
                pool.imap_unordered(process_single_pt_file, tasks),
                total=len(tasks),
                desc=f"Generating base meshes ({split_name}, method={args.method})",
            ):
                completed += 1
                if res:
                    all_results.extend(res)
                if (
                    args.early_export_every > 0
                    and args.num_export > 0
                    and completed % args.early_export_every == 0
                    and len(all_results) > 0
                ):
                    export_sample_objs(
                        all_results,
                        args.base_output_dir,
                        args.pt_dir,
                        args.export_dir,
                        num_export=args.num_export,
                        method=args.method,
                    )
                    print(
                        f"[{split_name}] Early export after {completed} / {len(tasks)} "
                        f".pt files -> {args.export_dir}/"
                    )

        base_json = os.path.join(args.base_output_dir, f"{split_name}_base.json")
        with open(base_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"[{split_name}] Generated {len(all_results)} base meshes for {len(pt_files)} GT models.")
        print(f"[{split_name}] Index saved to: {base_json}")
        return all_results

    # --- Process all splits, collecting results for export ---
    collected_results = []

    # Mode A: user provides a single split index_json
    if args.index_json and os.path.exists(args.index_json):
        split_name = os.path.splitext(os.path.basename(args.index_json))[0]
        pt_files = resolve_pt_files_from_index(args.index_json)
        print(f"Loaded {len(pt_files)} entries from {args.index_json}")
        collected_results.extend(run_split(split_name, pt_files, args))

    # Mode B: follow preprocess_data.py output (train.json + test.json)
    elif os.path.exists(os.path.join(args.pt_dir, "train.json")) or \
         os.path.exists(os.path.join(args.pt_dir, "test.json")):
        train_index = os.path.join(args.pt_dir, "train.json")
        test_index = os.path.join(args.pt_dir, "test.json")
        if os.path.exists(train_index):
            collected_results.extend(run_split("train", resolve_pt_files_from_index(train_index), args))
        else:
            print("[train] train.json not found, skipping.")
        if os.path.exists(test_index):
            collected_results.extend(run_split("test", resolve_pt_files_from_index(test_index), args))
        else:
            print("[test] test.json not found, skipping.")

    # Mode C: fallback - scan pt_dir for *.pt and treat as train
    else:
        pt_files = sorted(glob.glob(os.path.join(args.pt_dir, "*.pt")))
        print(f"Found {len(pt_files)} .pt files in {args.pt_dir} (no train/test json found)")
        collected_results.extend(run_split("train", pt_files, args))

    # --- Export sample OBJ pairs for visual inspection ---
    if args.num_export > 0 and collected_results:
        export_sample_objs(
            collected_results,
            args.base_output_dir,
            args.pt_dir,
            args.export_dir,
            num_export=args.num_export,
            method=args.method,
        )


if __name__ == "__main__":
    main()
