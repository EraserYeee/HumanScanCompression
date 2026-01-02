import torch
import numpy as np

class BarycentricSubdivision:
    """
    基于重心坐标采样的网格细分工具类。
    复刻 NGF 中的 sample_uniform_bary 和 build_triangulated_faces 逻辑。
    """
    def __init__(self, device='cuda'):
        self.uv_cache = {}
        self.face_cache = {} # cache faces for each (rate, num_triangles)
        self.merge_idx_cache = {} # Cache merge indices for seams stitching
        self.device = device

    def sample_uniform_bary(self, rate: int, num_triangles: int):
        """
        在重心坐标系中均匀采样。
        
        Args:
            rate: 细分等级 (每条边上的分段数)
            num_triangles: 三角形数量 Nt
            
        Returns:
            A, B: (Nt * K,) 张量，表示重心坐标。 C = 1 - A - B。
                  K = (rate+1)(rate+2)/2 是每个三角形内的采样点数。
        """
        cache_key = (rate, num_triangles)
        if cache_key in self.uv_cache:
            return self.uv_cache[cache_key]

        # Create uniform grid in [0,1] x [0,1]
        # 注意：NGF 中 steps=rate 其实是生成 rate 个点，
        # 但如果是细分，通常是 rate+1 个点 (0/r, 1/r, ..., r/r)
        # 我们严格照搬 ngf.py: U = linspace(0, 1, steps=rate)
        
        # 修正：根据 build_triangulated_faces 的逻辑，
        # L = max(1, rate - 1) 是边的段数。
        # 采样点数应该覆盖整个三角形。
        # ngf.py 中的 sample_uniform_bary 用的是 steps=rate。
        
        U = torch.linspace(0.0, 1.0, steps=rate, device=self.device)
        V = torch.linspace(0.0, 1.0, steps=rate, device=self.device)
        U, V = torch.meshgrid(U, V, indexing='ij')
        
        # Flatten to 1D
        U, V = U.reshape(-1), V.reshape(-1)
        
        # Filter out points where A+B > 1 (outside triangle)
        # 这里的 <= 1.0 可能存在精度问题，通常加个小 epsilon 或者不做这一步？
        # ngf.py 是做了 valid_mask = (U + V) <= 1.0
        valid_mask = (U + V) <= (1.0 + 1e-6)
        A = U[valid_mask]
        B = V[valid_mask]
        
        # Repeat for all triangles
        # A: (K,) -> (Nt, K) -> (Nt*K,)
        K = A.shape[0]
        A = A.repeat(num_triangles)
        B = B.repeat(num_triangles)
        
        self.uv_cache[cache_key] = (A, B)
        return A, B

    def build_triangulated_faces(self, rate: int, num_triangles: int):
        """
        构建细分后的面片连接关系。
        
        Args:
            rate: 细分等级
            num_triangles: 原始三角形数量 Nt
            
        Returns:
            faces: (Nt * sub_faces_per_tri, 3)  IntTensor
        """
        cache_key = (rate, num_triangles)
        if cache_key in self.face_cache:
            return self.face_cache[cache_key]
            
        # 复刻 ngf.py build_triangulated_faces 逻辑
        L = max(1, rate - 1)  # 边的分段数 (NGF logic: rate 可能是点数，这里 L 是段数)
        
        # 计算每行的采样点数量
        # row 0: L+1 points
        # row 1: L points ...
        row_counts = [(L + 1 - r) for r in range(L + 1)]
        row_prefix = [0]
        for r in range(L + 1):
            row_prefix.append(row_prefix[-1] + row_counts[r])
        
        K = row_prefix[-1] # 每个三角形的总采样点数
        
        faces_list = []
        
        # 构建单个标准三角形的细分拓扑
        # 然后通过 offset 复制给所有 Nt 个三角形
        # 这样比 Python 循环快很多
        
        single_tri_faces = []
        for r in range(L):
            for c in range(L - r):
                # 当前行的点
                a = row_prefix[r] + c
                b = row_prefix[r] + c + 1
                # 下一行的点
                d = row_prefix[r + 1] + c
                e = row_prefix[r + 1] + c + 1
                
                # 上三角 (a, b, d)
                # 注意：为了保持 CCW 顺序 (a -> d -> b)
                # 原 (a, b, d) 是 CW
                single_tri_faces.append([a, d, b])
                
                # 下三角 (b, e, d) - 只有当不是该行最后一个小三角时才有
                # 注意：为了保持 CCW 顺序 (b -> d -> e)
                # 原 (b, e, d) 是 CW
                if c < L - r - 1:
                    single_tri_faces.append([b, d, e])
                    
        single_tri_faces = torch.tensor(single_tri_faces, dtype=torch.long, device=self.device) # (M, 3)
        num_sub_faces = single_tri_faces.shape[0]
        
        # 复制到所有三角形
        # offsets: [0, K, 2K, ..., (Nt-1)K]
        offsets = torch.arange(num_triangles, device=self.device) * K
        offsets = offsets.view(-1, 1, 1) # (Nt, 1, 1)
        
        # faces: (Nt, M, 3) = (1, M, 3) + (Nt, 1, 1)
        faces = single_tri_faces.unsqueeze(0) + offsets
        faces = faces.view(-1, 3) # (Nt * M, 3)
        
        self.face_cache[cache_key] = faces
        return faces

    def compute_merge_indices(self, base_faces, rate, base_verts=None):
        """
        Compute indices to merge duplicate subdivision points on shared edges/vertices.
        
        Strategy:
        1. Identify which subdivision points lie on Base Vertices or Base Edges.
           - Barycentric coords (A, B, C):
             - Vertex: one component is 1 (approx > 1-eps)
             - Edge: one component is 0 (approx < eps)
        2. Map these points to a unique global key.
           - Vertex points: map to Base Vertex Index directly.
           - Edge points: map to sorted Base Edge Vertex Indices (min(v1, v2), max(v1, v2)) + position on edge.
        3. Assign a unique global index to every subdivision point.
        
        Args:
            base_faces: (B, F, 3) IntTensor
            rate: int, subdivision rate
            base_verts: Optional (B, V, 3) for debugging or exact positioning
            
        Returns:
            merge_idx: (B, F*K) LongTensor. Index into a unique vertex array.
            num_unique_verts: int (per batch item approx, or max index + 1)
        """
        batch_size, F_count, _ = base_faces.shape
        
        # We need A, B for a single triangle (not repeated yet)
        # sample_uniform_bary returns repeated. Let's just get the pattern for one triangle.
        # We can call sample_uniform_bary(rate, 1)
        A_single, B_single = self.sample_uniform_bary(rate, 1) # (K,)
        C_single = 1.0 - A_single - B_single
        K = A_single.shape[0]
        
        # Expand for batch and faces
        # We process everything flattened as (B*F, 3) faces
        flat_base_faces = base_faces.view(-1, 3) # (B*F, 3)
        total_faces = flat_base_faces.shape[0]
        
        # Prepare Bary coords for all points: (B*F*K, 3)
        # A, B, C repeated
        bary_A = A_single.repeat(total_faces)
        bary_B = B_single.repeat(total_faces)
        bary_C = C_single.repeat(total_faces)
        
        # Identify types of points
        eps = 1e-4
        
        # Vertices mask
        is_v0 = bary_A > 1.0 - eps
        is_v1 = bary_B > 1.0 - eps
        is_v2 = bary_C > 1.0 - eps
        
        # Edges mask (points on edge but not vertices)
        # Edge 0: opposite to v0 -> A approx 0
        # Edge 1: opposite to v1 -> B approx 0
        # Edge 2: opposite to v2 -> C approx 0
        on_e0 = (bary_A < eps) & (~is_v1) & (~is_v2) # on edge v1-v2
        on_e1 = (bary_B < eps) & (~is_v0) & (~is_v2) # on edge v0-v2
        on_e2 = (bary_C < eps) & (~is_v0) & (~is_v1) # on edge v0-v1
        
        # Internal points
        is_internal = (~is_v0) & (~is_v1) & (~is_v2) & (~on_e0) & (~on_e1) & (~on_e2)
        
        # --- Construct Unique Keys ---
        # Key format:
        # We need a large integer key.
        # Max vertices V approx 6000.
        # Edge key: v_min * V + v_max
        # Point on edge key: EdgeKey * Rate + edge_index
        
        # But doing this in pure tensor ops requires careful index arithmetic.
        
        # 1. Base Vertex Indices for each sub-point
        # face_v: (B*F*K, 3)
        # We need to repeat face indices K times
        # flat_base_faces: (B*F, 3) -> repeat interleave -> (B*F*K, 3)
        face_v_indices = flat_base_faces.repeat_interleave(K, dim=0)
        v0_idx = face_v_indices[:, 0]
        v1_idx = face_v_indices[:, 1]
        v2_idx = face_v_indices[:, 2]
        
        # Total points
        N_points = total_faces * K
        
        # Initialize merge_idx with a unique ID for every point initially
        # Then we overwrite shared ones.
        # Ideally, we map to a new compact space.
        
        # Let's use a "Global Hash" strategy.
        # Max V is known from input (V_max).
        # We can assume V_max < 100,000.
        V_max = face_v_indices.max() + 1
        
        # --- Type 1: Vertex Points ---
        # Map to original vertex index.
        # These are global indices [0, V-1].
        # We keep them as is.
        final_id = torch.full((N_points,), -1, dtype=torch.long, device=self.device)
        
        final_id[is_v0] = v0_idx[is_v0]
        final_id[is_v1] = v1_idx[is_v1]
        final_id[is_v2] = v2_idx[is_v2]
        
        # --- Type 2: Edge Points ---
        # Edge defined by (u, v) with u < v.
        # Edge ID = u * V_max + v.
        # Position on edge:
        # For barycentric subdivision, points on edge are equally spaced.
        # We can use the coordinate value (e.g. B for edge v0-v2 where A=0) to discretize.
        # coord * rate -> int index [1, rate-1]
        
        # Edge v1-v2 (A=0): use B (or C). B goes from 1 to 0. C goes from 0 to 1.
        # Edge v0-v2 (B=0): use A (or C).
        # Edge v0-v1 (C=0): use A (or B).
        
        # We need consistent ordering. Edge (u, v). If we use linear interpolation parameter t from u to v.
        # p = (1-t)*u + t*v.
        # Identify u, v for each edge case.
        
        # Helper to compute edge key and local index
        def handle_edge(mask, u_idx, v_idx, t_param):
            if not mask.any(): return None
            
            # Use boolean indexing on u_idx and v_idx which are already repeated
            # The mask should be the same length as u_idx, v_idx, t_param
            
            u = u_idx[mask]
            v = v_idx[mask]
            t = t_param[mask]
            
            # Ensure u < v for unique edge ID
            swap = u > v
            u_final = torch.where(swap, v, u)
            v_final = torch.where(swap, u, v)
            
            # Adjust t to be relative to u_final
            # if swapped (u->v became v->u), t (from u) becomes 1-t (from v)
            t_final = torch.where(swap, 1.0 - t, t)
            
            # Discretize t to integer index [1, rate-1]
            # t is in (0, 1). t * rate -> float. Round to nearest int.
            edge_sub_idx = torch.round(t_final * rate).long()
            
            # Unique Key for this point on edge
            # We need to shift these keys to avoid collision with Base Vertices [0, V_max-1]
            # Strategy: Store (u, v, sub_idx) and use unique() later?
            # Or construct a high-precision int64 key?
            # V_max ~ 10^4. u*V_max + v ~ 10^8. sub_idx ~ 10.
            # Key = (u * V_max + v) * rate + sub_idx. Max ~ 10^9 < 2^63. Safe for int64.
            
            key = (u_final * V_max + v_final) * (rate + 1) + edge_sub_idx
            
            # We store this key temporarily. We need to remap it later.
            # To distinguish from vertex IDs, we can offset or just mark them.
            # Let's use a separate array for "raw keys"
            return key
            
        # Raw Keys for all points
        # Initialize with -1
        raw_keys = torch.full((N_points,), -1, dtype=torch.long, device=self.device)
        
        # Fill Vertices (use simple indices)
        # To avoid collision with edge keys (which are large), we keep vertex keys small.
        # We will run unique() on all keys at the end.
        raw_keys[is_v0] = v0_idx[is_v0]
        raw_keys[is_v1] = v1_idx[is_v1]
        raw_keys[is_v2] = v2_idx[is_v2]
        
        # Fill Edges
        # Edge 0: v1-v2. A=0. Param t varies along v1->v2.
        # v = v1 * B + v2 * C.  B+C=1. -> v = v1*(1-C) + v2*C.
        # t = C. (C goes 0 at v1 to 1 at v2).
        # To avoid shape mismatch, we need to ensure C, v1_idx, v2_idx match mask length.
        # on_e0 is (N_points,), v1_idx is (N_points,), C is (N_points,).
        
        # But wait, v1_idx is derived from flat_base_faces repeat_interleave
        # N_points = total_faces * K
        # on_e0 is (N_points,) boolean mask.
        
        keys_e0 = handle_edge(on_e0, face_v_indices[:, 1], face_v_indices[:, 2], bary_C)
        if keys_e0 is not None: raw_keys[on_e0] = keys_e0 + V_max # Offset to avoid collision with V indices
        
        # Edge 1: v0-v2. B=0.
        # v = v0 * A + v2 * C. t = C.
        keys_e1 = handle_edge(on_e1, face_v_indices[:, 0], face_v_indices[:, 2], bary_C)
        if keys_e1 is not None: raw_keys[on_e1] = keys_e1 + V_max
        
        # Edge 2: v0-v1. C=0.
        # v = v0 * A + v1 * B. t = B.
        keys_e2 = handle_edge(on_e2, face_v_indices[:, 0], face_v_indices[:, 1], bary_B)
        if keys_e2 is not None: raw_keys[on_e2] = keys_e2 + V_max
        
        # --- Type 3: Internal Points ---
        # Internal points are unique to the face (no sharing).
        # We can just assign a unique sequential ID starting after max possible edge key.
        # Or simpler: Just assign a unique ID based on their linear index in the array + a large offset.
        # Max possible Edge Key ~ V^2 * rate.
        # Let's just use (Total_Faces + Face_Idx) * K + k_idx ? 
        # Actually, linear index `torch.arange(N_points)` is unique.
        # To ensure no collision with Shared keys, we can offset them by a huge number.
        huge_offset = V_max * V_max * (rate + 2)
        
        # We assign raw_keys[internal] = huge_offset + linear_index
        linear_idx = torch.arange(N_points, device=self.device)
        raw_keys[is_internal] = huge_offset + linear_idx[is_internal]
        
        # --- Remap to contiguous range [0, M-1] ---
        # torch.unique is perfect for this.
        # return_inverse gives us exactly the indices we need for scatter/gather.
        # unique_vals, inverse_indices = torch.unique(raw_keys, sorted=True, return_inverse=True)
        # This might be slow if N is large.
        
        # Optimization:
        # We effectively want to merge points with same raw_keys.
        # scatter_mean needs indices [0, Num_Unique-1].
        
        # However, for Batching, we need to be careful.
        # Ideally, we do this PER BATCH ITEM because Base Vertex indices reset to 0 for each batch item?
        # WAIT. In the current pipeline, `base_faces` indices are usually relative to the batch item's vertex start?
        # NO. Usually `base_faces` is (B, F, 3) where values are [0, V-1].
        # So `v0_idx` etc. computed above are local to each batch item.
        # But we flattened B*F.
        # So (Batch 0, Face 0, v=1) has same index 1 as (Batch 1, Face 0, v=1).
        # We MUST add batch offsets to `v_idx` before computing keys!
        
        # Re-compute face_v_indices with Batch Offsets
        batch_ids = torch.arange(batch_size, device=self.device, dtype=torch.long).view(batch_size, 1, 1).expand(batch_size, F_count, 3).reshape(-1, 3)
        # We need V_count per batch.
        # If V is constant, V_max is V.
        # We can just shift v_idx by batch_id * V_max
        
        # But we already used `flat_base_faces` which has values [0, V-1].
        # We need to make them global: v_global = v_local + batch_id * V_max
        V_per_batch = base_faces.max() + 1 # Approximate
        
        # Re-calculate `raw_keys` with batch offset injection?
        # Actually, simpler:
        # Calculate `raw_keys` as above (which treats batches as overlapping in ID space).
        # Then add `batch_index * Huge_Batch_Offset` to `raw_keys`.
        
        batch_indices_flat = batch_ids.repeat_interleave(K, dim=0)[:, 0] # (B*F*K,)
        
        # Offset raw_keys by batch
        # huge_batch_stride > max possible raw_key
        huge_batch_stride = huge_offset + N_points + 1
        raw_keys = raw_keys + batch_indices_flat * huge_batch_stride
        
        # Now find unique
        _, inverse_indices = torch.unique(raw_keys, return_inverse=True)
        
        # inverse_indices is (B*F*K,) with values [0, Num_Unique-1]
        # This is exactly what we need for scatter_mean.
        
        return inverse_indices.view(batch_size, -1)
