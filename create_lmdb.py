"""
Pack GT .pt and base .pt files into LMDB databases that stage2_dataset.py can read.

Produces one LMDB per split.  Key conventions (must match ScanToMeshDataset):
  - "__meta__"             -> { length, has_base }
  - "gt_{i}"               -> dict loaded from GT .pt  (i = index in split json)
  - "base_{rel_path}"      -> dict loaded from base .pt (rel_path from *_base.json)

Usage example:
  python create_lmdb.py \
      --pt_dir  /path/to/gt_pt \
      --base_dir /path/to/base_pt \
      --out_dir /path/to/lmdb_out
  -> creates  /path/to/lmdb_out/train.lmdb  and  /path/to/lmdb_out/test.lmdb
"""

import os
import json
import pickle
import argparse

import lmdb
import torch
from tqdm import tqdm


def pack_split(
    pt_dir: str,
    base_dir: str | None,
    split: str,
    out_path: str,
    map_size: int,
):
    split_json = os.path.join(pt_dir, f"{split}.json")
    if not os.path.exists(split_json):
        print(f"[{split}] {split_json} not found, skipping.")
        return

    with open(split_json, "r") as f:
        gt_list = json.load(f)

    base_list = []
    if base_dir:
        base_json = os.path.join(base_dir, f"{split}_base.json")
        if os.path.exists(base_json):
            with open(base_json, "r") as f:
                base_list = json.load(f)
        else:
            print(f"[{split}] {base_json} not found, packing GT only.")

    print(f"[{split}] GT: {len(gt_list)} | Base: {len(base_list)} -> {out_path}")

    env = lmdb.open(out_path, map_size=map_size)
    with env.begin(write=True) as txn:
        meta = {"length": len(gt_list), "has_base": len(base_list) > 0}
        txn.put("__meta__".encode(), pickle.dumps(meta))

        print(f"[{split}] Packing GT...")
        for i, entry in enumerate(tqdm(gt_list, desc=f"{split} GT")):
            pt_path = os.path.join(pt_dir, entry["pt_path"])
            try:
                data = torch.load(pt_path, map_location="cpu", weights_only=False)
                txn.put(f"gt_{i}".encode(), pickle.dumps(data))
            except Exception as e:
                print(f"  [Error] GT {pt_path}: {e}")

        if base_list:
            print(f"[{split}] Packing Base...")
            for entry in tqdm(base_list, desc=f"{split} Base"):
                rel_path = entry["base_pt_path"]
                full_path = os.path.join(base_dir, rel_path)
                try:
                    data = torch.load(full_path, map_location="cpu", weights_only=False)
                    txn.put(f"base_{rel_path}".encode(), pickle.dumps(data))
                except Exception as e:
                    print(f"  [Error] Base {full_path}: {e}")

    env.close()
    print(f"[{split}] Done.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Pack GT + Base .pt files into LMDB (one per split) for stage2_dataset.py"
    )
    parser.add_argument(
        "--pt_dir",
        type=str,
        required=True,
        help="Directory with GT .pt files and train.json / test.json",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=None,
        help="Directory with base meshes and train_base.json / test_base.json",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory (will contain train.lmdb / test.lmdb)",
    )
    parser.add_argument(
        "--map_size_gb",
        type=float,
        default=100.0,
        help="LMDB map size in GB (default: 100)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,test",
        help="Comma-separated splits to pack (default: train,test)",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    map_size = int(args.map_size_gb * 1024 * 1024 * 1024)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    for split in splits:
        out_path = os.path.join(args.out_dir, f"{split}.lmdb")
        pack_split(args.pt_dir, args.base_dir, split, out_path, map_size)

    print("All done.")


if __name__ == "__main__":
    main()
