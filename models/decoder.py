import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.subdivision import BarycentricSubdivision

def positional_encoding(vector: torch.Tensor, extras: list[torch.Tensor], levels: int) -> torch.Tensor:
    """
    复刻 ngf.py 中的 positional_encoding
    Improved: Add PI factor to ensure better frequency coverage in [-1, 1]
    """
    result = list(extras) # Copy list
    
    # Use torch.linspace for frequencies ensures we cover the range evenly if needed, 
    # but 2^i is standard.
    # We add math.pi to ensure sin(x * pi) covers a full cycle in [-1, 1]
    
    for i in range(levels):
        k = (2.0 ** i) * math.pi
        result.append(torch.sin(k * vector))
        result.append(torch.cos(k * vector))
        
    return torch.cat(result, dim=-1)

class NeuralSubdivisionDecoder(nn.Module):
    """
    基于 NGF 逻辑的解码器。
    不改变 Base Mesh 的拓扑数据结构，而是通过重心坐标采样生成细分后的点云和面片。
    
    流程:
    1. 在 Base Mesh 的每个三角形面上生成密集采样点 (重心坐标 A, B)。
    2. 插值得到采样点的:
       - 粗糙位置 lp (Linear Position)
       - 特征 lf (Linear Feature)
       - 法线 ln (Linear Normal)
    3. 神经位移:
       - input = PosEnc(lp) + lf
       - displacement = MLP(input)
       - fine_pos = lp + displacement * ln
    """
    def __init__(self, feature_dim=128, hidden_dim=64, levels=8, rate=4, predict_offset=False):
        """
        Args:
            feature_dim: 输入特征维度
            hidden_dim: MLP 隐藏层维度
            levels: Positional Encoding 的层数 (fflevels)
            rate: 细分等级 (edge subdivision rate)
            predict_offset: If True, predict 3D offset (xyz) instead of scalar displacement
        """
        super().__init__()
        self.fflevels = levels
        self.rate = rate
        self.predict_offset = predict_offset
        self.subdivision = BarycentricSubdivision()
        
        # MLP Input Dim calculation
        # input = lf (feature_dim) + PosEnc(lp) (3 * 2 * levels)
        self.input_dim = feature_dim + 3 * 2 * levels
        
        # MLP Output Dim: 1 for scalar displacement (along normal), 3 for vector offset (xyz)
        out_dim = 3 if self.predict_offset else 1
        
        # MLP_Normal from ngf.py
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

        # Initialize the last layer to output near-zero values
        # This ensures the deformation starts from the base mesh
        nn.init.uniform_(self.mlp[-1].weight, -1e-5, 1e-5)
        nn.init.constant_(self.mlp[-1].bias, 0)

    def interpolate_barycentric(self, attrs, faces, A, B):
        """
        Args:
            attrs: (B, V, C) 顶点属性
            faces: (B, F, 3) 面片索引
            A, B: (F*K,) 重心坐标 (已被 repeat 适配了所有面)
            
        Returns:
            interpolated: (B, F*K, C)
        """
        # attrs: (B, V, C)
        # faces: (B, F, 3)
        
        # Gather face vertex attributes
        # (B, F, 3) -> (B, F, 3, C)
        # 需要 gather: attrs[b, faces[b, f, i], :]
        
        B_batch, V, C = attrs.shape
        _, F_count, _ = faces.shape
        
        # Flat indices for gather
        # batch_offset: (B, 1, 1)
        batch_offset = (torch.arange(B_batch, device=attrs.device) * V).view(-1, 1, 1)
        flat_faces = (faces + batch_offset).view(-1) # (B*F*3)
        flat_attrs = attrs.view(-1, C) # (B*V, C)
        
        face_attrs = flat_attrs[flat_faces].view(B_batch, F_count, 3, C) # (B, F, 3, C)
        
        # Reshape A, B for broadcasting
        # A, B input: (F*K,) -> need to reshape to match (B, F, K) logic?
        # subdivision.sample_uniform_bary 返回的是 (F*K,)，没有 Batch 维
        # 我们假设所有 Batch 共享相同的细分模式 (A, B 相同)
        
        # A: (F*K,) -> (F, K)
        # 这里的 F 是 faces count (num_triangles)
        # NGF 的逻辑是 A, B 长度为 Nt * K
        
        K = A.shape[0] // F_count 
        
        # A_reshaped: (1, F, K, 1)
        A_reshaped = A.view(1, F_count, K, 1)
        B_reshaped = B.view(1, F_count, K, 1)
        C_reshaped = 1.0 - A_reshaped - B_reshaped
        
        # face_attrs: (B, F, 3, C) -> (B, F, 1, 3, C) for broadcasting with K
        # Wait, we need to combine 3 vertices.
        # v0 = face_attrs[:, :, 0, :] # (B, F, C)
        # v0 * A ? No. A corresponds to points inside the triangle.
        
        # Correct logic:
        # Result shape: (B, F, K, C)
        # v0 * A + v1 * B + v2 * C
        
        v0 = face_attrs[:, :, 0, :].unsqueeze(2) # (B, F, 1, C)
        v1 = face_attrs[:, :, 1, :].unsqueeze(2) # (B, F, 1, C)
        v2 = face_attrs[:, :, 2, :].unsqueeze(2) # (B, F, 1, C)
        
        interp = v0 * A_reshaped + v1 * B_reshaped + v2 * C_reshaped # (B, F, K, C)
        
        return interp.view(B_batch, -1, C) # (B, F*K, C)

    def forward(self, base_verts, base_faces, vertex_features, base_normals):
        """
        Args:
            base_verts: (B, V, 3)
            base_faces: (B, F, 3)
            vertex_features: (B, V, D)
            base_normals: (B, V, 3)

        Returns:
            fine_verts: (B, V_fine, 3)
            fine_faces: (F_fine, 3) shared across batch (if topology is same)
            displacements: (B, V_fine, 1)
        """
        B, V, _ = base_verts.shape
        B = int(B) # Ensure B is int for view()
        _, num_faces, _ = base_faces.shape
        
        # 1. Generate Barycentric Coordinates
        # A, B: (F*K,)
        uv_A, uv_B = self.subdivision.sample_uniform_bary(self.rate, num_triangles=num_faces)
        
        # 2. Interpolate Attributes
        # lp: Linear Position (B, F*K, 3)
        lp = self.interpolate_barycentric(base_verts, base_faces, uv_A, uv_B)
        
        # lf: Linear Feature (B, F*K, D)
        lf = self.interpolate_barycentric(vertex_features, base_faces, uv_A, uv_B)
        
        # ln: Linear Normal (B, F*K, 3)
        ln = self.interpolate_barycentric(base_normals, base_faces, uv_A, uv_B)
        ln = F.normalize(ln, dim=-1, p=2)
        
        # 3. Neural Displacement
        # Positional Encoding on lp (Assumed normalized in Dataset)
        # lin = PosEnc(lp) + lf
        lin = positional_encoding(lp, [lf], self.fflevels) # (B, F*K, input_dim)
        
        # MLP Prediction
        # Reshape for MLP: (B * F * K, input_dim)
        lin_flat = lin.view(-1, self.input_dim)
        disp_flat = self.mlp(lin_flat) # (B*F*K, out_dim)
        
        if self.predict_offset:
            # Output is (B*F*K, 3) vector offset
            disp = disp_flat.view(B, -1, 3) # (B, F*K, 3)
            # fine_verts = lp + disp (add offset directly)
            fine_verts = lp + disp
        else:
            # Output is (B*F*K, 1) scalar displacement
            disp = disp_flat.view(B, -1, 1) # (B, F*K, 1)
            # fine_verts = lp + d * ln
            fine_verts = lp + disp * ln
        
        # 4. Build Topology (for rendering)
        # fine_faces: (F*sub_faces, 3)
        fine_faces = self.subdivision.build_triangulated_faces(self.rate, num_triangles=num_faces)
        
        return fine_verts, fine_faces, disp
