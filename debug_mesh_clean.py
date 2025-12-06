import pymeshlab
import argparse
import os
import time

def clean_and_simplify_pymeshlab(input_path, output_dir, target_verts):
    t0 = time.time()
    
    # Initialize MeshSet
    ms = pymeshlab.MeshSet()
    print(f"[PyMeshLab] Loading {input_path}...")
    ms.load_new_mesh(input_path)
    
    m = ms.current_mesh()
    print(f"  Initial: V={m.vertex_number()}, F={m.face_number()}")
    
    # 1. Merge Close Vertices (Dedup)
    # threshold=0 means exact match (usually, or very small epsilon)
    # You can set a small threshold if needed, e.g. pymeshlab.Percentage(0.0001)
    print("  [Step 1] Merging close vertices...")
    ms.meshing_merge_close_vertices()
    
    # 2. Remove Small Components (Denoise)
    # Remove components smaller than 20% of the bounding box diagonal? 
    # Or use a fixed percentage. Let's try 10-20%.
    # mincomponentdiag: The diameter of the connected component is calculated as the 
    # diagonal of its bounding box.
    print("  [Step 2] Removing small connected components (by diameter)...")
    ms.meshing_remove_connected_component_by_diameter()
    
    # 3. Repair Non-Manifold Edges
    print("  [Step 3] Repairing non-manifold edges...")
    ms.meshing_repair_non_manifold_edges()
    
    m = ms.current_mesh()
    print(f"  After Clean: V={m.vertex_number()}, F={m.face_number()}")
    
    # Save Cleaned
    name = os.path.splitext(os.path.basename(input_path))[0]
    clean_path = os.path.join(output_dir, f"{name}_cleaned.obj")
    ms.save_current_mesh(clean_path)
    print(f"  Exported cleaned mesh to {clean_path}")
    
    # 4. Simplify
    target_faces = target_verts
    print(f"  [Step 4] Decimating to {target_faces} faces (~{target_verts} verts)...")
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces)
    
    m = ms.current_mesh()
    print(f"  Final: V={m.vertex_number()}, F={m.face_number()}")
    
    # Save Simplified
    simple_path = os.path.join(output_dir, f"{name}_simplified_{target_verts}.obj")
    ms.save_current_mesh(simple_path)
    print(f"  Exported simplified mesh to {simple_path}")
    
    print(f"[PyMeshLab] Total Time: {time.time() - t0:.3f}s")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="Path to input OBJ/PLY")
    parser.add_argument('--output_dir', type=str, default="debug_clean_results")
    parser.add_argument('--target_verts', type=int, default=1600)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        clean_and_simplify_pymeshlab(args.input, args.output_dir, args.target_verts)
        print("Done!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
