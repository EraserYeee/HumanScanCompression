import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_points, knn_gather

def compute_rotation_matrices(normals: torch.Tensor) -> torch.Tensor:
    """
    根据法线计算每个顶点的局部旋转矩阵 R (B, V, 3, 3)。
    
    坐标系定义:
    Z轴 = Normal (法线方向)
    X轴 = 在切平面上，尽量指向固定的参考方向 (如 [1, 0, 0] 或 [0, 1, 0])
    Y轴 = Z x X (确保右手坐标系)

    Args:
        normals: (B, V, 3) 归一化的法线向量

    Returns:
        R: (B, V, 3, 3) 旋转矩阵，使得 p_local = (p_global - center) @ R^T
            即 R 的行向量分别是 X, Y, Z 轴
    """
    B, V, _ = normals.shape
    
    # 1. Z轴: 法线方向
    z_axis = F.normalize(normals, dim=-1) # (B, V, 3)
    
    # 2. 构造辅助向量来计算 X 轴
    # 策略: 优先使用 [0, 1, 0] (Up)，如果法线与 Up 平行，则使用 [1, 0, 0]
    up = torch.tensor([0.0, 1.0, 0.0], device=normals.device).view(1, 1, 3).expand(B, V, 3)
    
    # 计算点积绝对值，判断是否平行
    dot = torch.abs(torch.sum(z_axis * up, dim=-1, keepdim=True))
    
    # 如果接近平行 (>0.99)，使用 [1, 0, 0] 作为辅助向量
    fallback = torch.tensor([1.0, 0.0, 0.0], device=normals.device).view(1, 1, 3).expand(B, V, 3)
    aux_vector = torch.where(dot > 0.99, fallback, up)
    
    # 3. X轴: aux x Z (垂直于法线，位于切平面)
    x_axis = torch.cross(aux_vector, z_axis, dim=-1)
    x_axis = F.normalize(x_axis, dim=-1)
    
    # 4. Y轴: Z x X (确保正交)
    y_axis = torch.cross(z_axis, x_axis, dim=-1)
    # y_axis 已经是归一化的，因为 Z 和 X 正交且归一化
    
    # 5. 构建矩阵 R
    # 我们希望局部坐标 p_local = [x_local, y_local, z_local]
    # x_local = (p - c) dot X_axis
    # y_local = (p - c) dot Y_axis
    # z_local = (p - c) dot Z_axis
    # 所以 R 的行向量应该是 X, Y, Z
    
    # stack(dim=-2) -> (B, V, 3, 3)
    # R[b, v] = [ [x_0, x_1, x_2],
    #             [y_0, y_1, y_2],
    #             [z_0, z_1, z_2] ]
    R = torch.stack([x_axis, y_axis, z_axis], dim=-2)
    
    return R

class LocalPatchGrouper(nn.Module):
    """
    负责将 Scan Point Cloud (高分辨率点云) 关联到 Base Mesh (低分辨率网格) 的顶点上，
    并进行局部坐标系的标准化 (Canonicalization)。

    职责:
    1. KNN 搜索: 找到每个 Scan 点最近的 Base Mesh 顶点。
    2. 局部坐标变换: 
       - 计算每个 Base Mesh 顶点的切空间坐标系 (TBN矩阵)。
       - 将 Scan 点从世界坐标转换到该顶点的局部坐标系。
    """
    def __init__(self, use_global_coordinates=False):
        super().__init__()
        self.use_global_coordinates = use_global_coordinates

    # def forward(self, base_verts: torch.Tensor, base_normals: torch.Tensor, scan_points: torch.Tensor):
    #     """
    #     Args:
    #         base_verts: (B, V, 3) Base mesh 顶点 (锚点中心)
    #         base_normals: (B, V, 3) Base mesh 顶点法线 (锚点方向)
    #         scan_points: (B, P, 3) Scan 采样点云

    #     Returns:
    #         local_points: (B, P, 3) 转换到局部坐标系的点
    #         cluster_idx: (B, P) 每个点归属的顶点索引
    #     """
    #     B, P, _ = scan_points.shape
        
    #     # 1. KNN 搜索 (K=1)
    #     # dists: (B, P, 1), idx: (B, P, 1)
    #     knn_res = knn_points(scan_points, base_verts, K=1)
    #     cluster_idx = knn_res.idx.squeeze(-1) # (B, P)
        
    #     # 2. Gather 锚点信息 (中心位置 & 法线)
    #     # 创建 batch 索引用于 gather
    #     batch_indices = torch.arange(B, device=scan_points.device).view(-1, 1).expand(-1, P)
        
    #     # gather: 从 (B, V, 3) 中根据 (B, P) 的索引取值 -> (B, P, 3)
    #     # 注意: cluster_idx 的值域是 [0, V-1]
    #     anchor_pos = base_verts[batch_indices, cluster_idx]      # (B, P, 3)
        
    #     # 3. 计算相对位置 (Global Relative)
    #     delta_p = scan_points - anchor_pos # (B, P, 3)
        
    #     # 4. 计算并 Gather 旋转矩阵
    #     # 优化: 先为 V 个顶点算好 R，再 Gather 到 P 个点，比直接为 P 个点算 R 省显存
    #     R_verts = compute_rotation_matrices(base_normals) # (B, V, 3, 3)
        
    #     # Gather R: (B, P, 3, 3)
    #     R_points = R_verts[batch_indices, cluster_idx] 
        
    #     # 5. 应用旋转 (坐标变换)
    #     # p_local = R @ delta_p
    #     # R: (B, P, 3, 3), delta_p: (B, P, 3) -> (B, P, 3, 1)
    #     # matmul: (..., 3, 3) x (..., 3, 1) -> (..., 3, 1)
        
    #     p_local = torch.matmul(R_points, delta_p.unsqueeze(-1)).squeeze(-1)
        
    #     return p_local, cluster_idx

    def forward(self, base_verts: torch.Tensor, base_normals: torch.Tensor, scan_points: torch.Tensor):
        """
        Args:
            base_verts: (B, V, 3) Base mesh 顶点 (锚点中心)
            base_normals: (B, V, 3) Base mesh 顶点法线 (锚点方向)
            scan_points: (B, P, 3) Scan 采样点云

        Returns:
            local_points: (B, P, 3) 转换到局部坐标系的点
            cluster_idx: (B, P) 每个点归属的顶点索引
        """
        B, P, _ = scan_points.shape
        
        # 1. KNN 搜索 (K=1)
        # dists: (B, P, 1), idx: (B, P, 1)
        knn_res = knn_points(scan_points, base_verts, K=1)
        cluster_idx = knn_res.idx.squeeze(-1) # (B, P)
        
        # 2. Gather 锚点信息 (中心位置 & 法线)
        # 创建 batch 索引用于 gather
        batch_indices = torch.arange(B, device=scan_points.device).view(-1, 1).expand(-1, P)
        
        # gather: 从 (B, V, 3) 中根据 (B, P) 的索引取值 -> (B, P, 3)
        # 注意: cluster_idx 的值域是 [0, V-1]
        anchor_pos = base_verts[batch_indices, cluster_idx]      # (B, P, 3)
        
        # 3. 计算相对位置 (Global Relative)
        delta_p = scan_points - anchor_pos # (B, P, 3)

        # 直接使用相对位置作为局部坐标 (不进行旋转)
        p_local = delta_p
        
        # # 4. 坐标变换
        # if self.use_global_coordinates:
        #     # 直接使用相对位置作为局部坐标 (不进行旋转)
        #     p_local = delta_p
        # else:
        #     # 计算并 Gather 旋转矩阵
        #     # 优化: 先为 V 个顶点算好 R，再 Gather 到 P 个点，比直接为 P 个点算 R 省显存
        #     R_verts = compute_rotation_matrices(base_normals) # (B, V, 3, 3)
            
        #     # Gather R: (B, P, 3, 3)
        #     R_points = R_verts[batch_indices, cluster_idx] 
            
        #     # 应用旋转 p_local = R @ delta_p
        #     # R: (B, P, 3, 3), delta_p: (B, P, 3) -> (B, P, 3, 1)
        #     # matmul: (..., 3, 3) x (..., 3, 1) -> (..., 3, 1)
        #     p_local = torch.matmul(R_points, delta_p.unsqueeze(-1)).squeeze(-1)
        
        return p_local, cluster_idx


class LocalPatchGrouperPadding(nn.Module):
    """
    
    """
    def __init__(self):
        super().__init__()

    def forward(self, base_verts: torch.Tensor, base_normals: torch.Tensor, scan_points: torch.Tensor, patch_num_points: int=64):
        """
        Args:
            base_verts: (B, V, 3) Base mesh 顶点 (锚点中心)
            base_normals: (B, V, 3) Base mesh 顶点法线 (锚点方向)
            scan_points: (B, P, 3) Scan 采样点云

        Returns:
            local_points: (B, P, 3) 转换到局部坐标系的点
            cluster_idx: (B, P) 每个点归属的顶点索引
        """
        # 类似于Patch Grouper流程，先找anchor，再对每个group做padding/cropping

        B, P, _ = scan_points.shape
        V = base_verts.shape[1]

        # 1. KNN 搜索 (K=1)
        knn_res = knn_points(scan_points, base_verts, K=1)
        cluster_idx = knn_res.idx.squeeze(-1)  # (B, P)

        # 2. Gather 锚点信息 (中心位置 & 法线)
        batch_indices = torch.arange(B, device=scan_points.device).view(-1, 1).expand(-1, P)
        anchor_pos = base_verts[batch_indices, cluster_idx]  # (B, P, 3)

        # 3. 计算相对位置
        delta_p = scan_points - anchor_pos  # (B, P, 3)

        # 4. 计算旋转矩阵
        R_verts = compute_rotation_matrices(base_normals)  # (B, V, 3, 3)
        R_points = R_verts[batch_indices, cluster_idx]     # (B, P, 3, 3)

        # 5. 应用旋转
        p_local = torch.matmul(R_points, delta_p.unsqueeze(-1)).squeeze(-1)  # (B, P, 3)

        # 6. 对每个vertex聚类点, padding/cropping到patch_num_points
        device = scan_points.device
        grouped_points = []
        grouped_idx = []

        for b in range(B):
            groups_b = []
            idx_b = []
            for v in range(V):
                # 选出属于anchor v的所有点的索引
                mask = (cluster_idx[b] == v)
                point_indices = torch.where(mask)[0]  # (N_v,)
                n_points = point_indices.shape[0]

                if n_points >= patch_num_points:
                    # 对anchor v所有点的p_local与base_verts距离排序,留下最近patch_num_points
                    p_local_v = p_local[b, point_indices, :]  # (n_points, 3)
                    scan_v = scan_points[b, point_indices, :]  # (n_points, 3)
                    anchor_pos_v = base_verts[b, v].unsqueeze(0)  # (1, 3)
                    dists = torch.norm(scan_v - anchor_pos_v, dim=1)  # (n_points,)
                    topk = torch.topk(dists, k=patch_num_points, largest=False)
                    selected_idx = point_indices[topk.indices]  # patch_num_points indices
                    groups_b.append(p_local[b, selected_idx, :].unsqueeze(0))  # (1, patch_num_points, 3)
                    idx_b.append(selected_idx.unsqueeze(0))  # (1, patch_num_points)
                elif n_points > 0:
                    # 采样不足: 重复采样直到数量满
                    repeat = patch_num_points // n_points
                    remainder = patch_num_points % n_points
                    selected = point_indices.repeat(repeat)
                    if remainder > 0:
                        idx_extra = torch.randperm(n_points, device=device)[:remainder]
                        extra = point_indices[idx_extra]
                        selected = torch.cat([selected, extra], dim=0)
                    groups_b.append(p_local[b, selected, :].unsqueeze(0))
                    idx_b.append(selected.unsqueeze(0))
                else:
                    # 没有点属于该group, 填充全0
                    groups_b.append(torch.zeros(1, patch_num_points, 3, device=device))
                    idx_b.append(torch.full((1, patch_num_points), -1, dtype=torch.long, device=device))
            # 每个V
            grouped_points.append(torch.cat(groups_b, dim=0)) # (V, patch_num_points, 3)
            grouped_idx.append(torch.cat(idx_b, dim=0))       # (V, patch_num_points)

        # 输出: (B, V, patch_num_points, 3), (B, V, patch_num_points)
        local_points_grouped = torch.stack(grouped_points, dim=0)
        grouped_idx = torch.stack(grouped_idx, dim=0)
        return local_points_grouped, grouped_idx


def _chunked_topk_knn(
    query: torch.Tensor,
    database: torch.Tensor,
    k: int,
    chunk_size: int = 128,
) -> torch.Tensor:
    """Fast KNN via chunked cdist + topk (avoids full-sort bottleneck).

    Args:
        query:    (B, V, 3)
        database: (B, P, 3)
        k:        number of nearest neighbours
        chunk_size: vertices per chunk (controls peak VRAM of distance matrix)

    Returns:
        idx: (B, V, K)  indices into database dim-1
    """
    B, V, _ = query.shape
    idx_chunks = []
    for start in range(0, V, chunk_size):
        q = query[:, start : start + chunk_size]          # (B, cs, 3)
        d = torch.cdist(q, database)                      # (B, cs, P)
        _, topk_idx = d.topk(k, dim=-1, largest=False)    # (B, cs, K)
        idx_chunks.append(topk_idx)
    return torch.cat(idx_chunks, dim=1)                    # (B, V, K)


class VertexKNNGrouper(nn.Module):
    """
    Vertex-side KNN grouper: each base mesh vertex gathers its K nearest
    scan points, producing fixed-size neighborhoods suitable for batched
    self-attention (no scatter / variable-length padding needed).

    Uses chunked cdist + topk instead of PyTorch3D knn_points to avoid
    the expensive full-sort on the P dimension.
    """
    def __init__(self, k: int = 512, knn_chunk_size: int = 128):
        super().__init__()
        self.k = k
        self.knn_chunk_size = knn_chunk_size

    def forward(
        self,
        base_verts: torch.Tensor,
        base_normals: torch.Tensor,
        scan_points: torch.Tensor,
        scan_normals: torch.Tensor | None = None,
    ):
        """
        Args:
            base_verts:   (B, V, 3)
            base_normals: (B, V, 3)  – unused here but kept for API compat
            scan_points:  (B, P, 3)
            scan_normals: (B, P, 3) or None

        Returns:
            grouped_features: (B, V, K, D)  D=3 or 6
            local_coords:     (B, V, K, 3)  relative positions (for PE)
        """
        K = min(self.k, scan_points.shape[1])

        with torch.no_grad():
            idx = _chunked_topk_knn(
                base_verts, scan_points, K, chunk_size=self.knn_chunk_size
            )  # (B, V, K)

        gathered_pts = knn_gather(scan_points, idx)            # (B, V, K, 3)
        local_coords = gathered_pts - base_verts.unsqueeze(2)

        if scan_normals is not None:
            gathered_normals = knn_gather(scan_normals, idx)   # (B, V, K, 3)
            grouped_features = torch.cat([local_coords, gathered_normals], dim=-1)
        else:
            grouped_features = local_coords

        return grouped_features, local_coords