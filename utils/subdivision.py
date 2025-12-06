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
