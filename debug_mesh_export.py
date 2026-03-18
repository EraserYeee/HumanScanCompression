import os
import json
import argparse
import random

import torch
import numpy as np
import trimesh


def load_pt_mesh(pt_path: str):
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    if "gt_verts" in data and "gt_faces" in data:
        v = data["gt_verts"].cpu().numpy()
        f = data["gt_faces"].cpu().numpy()
    elif "base_verts" in data and "base_faces" in data:
        v = data["base_verts"].cpu().numpy()
        f = data["base_faces"].cpu().numpy()
    else:
        raise KeyError(f"Unsupported pt format: {pt_path}, keys={list(data.keys())}")
    return v.astype(np.float32), f.astype(np.int64)


def export_obj(verts, faces, out_path: str):
    m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    m.export(out_path, include_normals=False, include_texture=False)


def main():
    parser = argparse.ArgumentParser("Export GT/Base OBJ pairs for quick inspection")
    parser.add_argument("--pt_dir", type=str, required=True, help="Directory containing GT .pt and train.json/test.json")
    parser.add_argument("--base_dir", type=str, required=True, help="Directory containing base meshes and train_base.json/test_base.json")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Which split to export")
    parser.add_argument("--num", type=int, default=8, help="How many GT/Base pairs to export")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out_dir", type=str, default="./tempout", help="Output directory for OBJ pairs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    split_index = os.path.join(args.pt_dir, f"{args.split}.json")
    base_index = os.path.join(args.base_dir, f"{args.split}_base.json")
    if not os.path.exists(split_index):
        raise FileNotFoundError(f"Missing split index: {split_index}")
    if not os.path.exists(base_index):
        raise FileNotFoundError(f"Missing base index: {base_index}")

    with open(split_index, "r") as f:
        gt_entries = json.load(f)
    with open(base_index, "r", encoding="utf-8") as f:
        base_entries = json.load(f)

    # base_entries: many variants per GT; group by gt_pt_path
    base_by_gt = {}
    for e in base_entries:
        base_by_gt.setdefault(e["gt_pt_path"], []).append(e)

    # only keep GTs that have base variants
    gt_entries = [e for e in gt_entries if e["pt_path"] in base_by_gt]
    if len(gt_entries) == 0:
        raise RuntimeError("No GT entries found that have matching base meshes.")

    random.seed(args.seed)
    sample = gt_entries if len(gt_entries) <= args.num else random.sample(gt_entries, args.num)

    for item in sample:
        gt_rel = item["pt_path"]
        name = os.path.splitext(os.path.basename(gt_rel))[0]

        gt_pt = os.path.join(args.pt_dir, gt_rel)
        base_candidates = base_by_gt[gt_rel]
        base_item = random.choice(base_candidates)
        base_pt = os.path.join(args.base_dir, base_item["base_pt_path"])

        gt_v, gt_f = load_pt_mesh(gt_pt)
        base_v, base_f = load_pt_mesh(base_pt)

        gt_out = os.path.join(args.out_dir, f"{name}_gt.obj")
        base_out = os.path.join(args.out_dir, f"{name}_base.obj")

        export_obj(gt_v, gt_f, gt_out)
        export_obj(base_v, base_f, base_out)

    print(f"Exported {len(sample)} GT/Base pairs to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()